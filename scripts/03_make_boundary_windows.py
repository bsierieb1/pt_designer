#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create -499/+499 bp guide-search windows around desired tile boundaries."
    )
    parser.add_argument(
        "--locus-json",
        required=True,
        help="Path to locus.json from Step 1.",
    )
    parser.add_argument(
        "--tiles-tsv",
        required=True,
        help="Path to tiles_phasing_optimized.tsv from Step 2.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--boundary-flank",
        type=int,
        default=499,
        help="Window flank size around each desired boundary. Default: 499",
    )
    return parser.parse_args()


def load_locus_json(path: Path) -> dict:
    with path.open() as fh:
        data = json.load(fh)

    required = ["chrom", "start_1based", "end_1based"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"locus.json missing required keys: {missing}")

    return data


def load_tiles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    required = ["tile_id", "chrom", "start_1based", "end_1based"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Tiles TSV missing required columns: {missing}")

    df["start_1based"] = pd.to_numeric(df["start_1based"], errors="coerce")
    df["end_1based"] = pd.to_numeric(df["end_1based"], errors="coerce")
    bad = df["start_1based"].isna() | df["end_1based"].isna()
    if bad.any():
        raise ValueError(
            "Tiles TSV contains missing start/end coordinates:\n"
            + df.loc[bad, ["tile_id", "chrom", "start_1based", "end_1based"]].to_string(index=False)
        )

    df["tile_label"] = df["tile_id"].astype(str)
    tile_num = pd.to_numeric(df["tile_id"], errors="coerce")
    if tile_num.isna().any() or tile_num.duplicated().any():
        df = df.sort_values(["start_1based", "end_1based", "tile_label"]).reset_index(drop=True)
        df["tile_id"] = range(1, len(df) + 1)
    else:
        df["tile_id"] = tile_num.astype(int)
        df = df.sort_values("tile_id").reset_index(drop=True)

    df["start_1based"] = df["start_1based"].astype(int)
    df["end_1based"] = df["end_1based"].astype(int)

    return df


def make_boundary_window(
    chrom: str,
    tile_id: int,
    boundary_type: str,
    desired_boundary_1based: int,
    flank: int,
    search_start_1based: int,
    search_end_1based: int,
) -> dict:
    window_start_1based = max(search_start_1based, desired_boundary_1based - flank)
    window_end_1based = min(search_end_1based, desired_boundary_1based + flank)

    if window_start_1based > window_end_1based:
        raise ValueError(
            f"Invalid boundary window for tile {tile_id} {boundary_type}: "
            f"desired_boundary_1based={desired_boundary_1based}, "
            f"search_start_1based={search_start_1based}, "
            f"search_end_1based={search_end_1based}, flank={flank}"
        )

    return {
        "tile_id": tile_id,
        "chrom": chrom,
        "boundary_type": boundary_type,
        "desired_boundary_1based": desired_boundary_1based,
        "desired_boundary_0based": desired_boundary_1based - 1,
        "window_start_1based": window_start_1based,
        "window_end_1based": window_end_1based,
        "window_start_0based": window_start_1based - 1,
        "window_end_0based": window_end_1based,
        "window_length_bp": window_end_1based - window_start_1based + 1,
    }


def build_boundary_windows(
    tiles_df: pd.DataFrame,
    locus_start_1based: int,
    locus_end_1based: int,
    flank: int,
) -> pd.DataFrame:
    rows = []
    search_start_1based = min(locus_start_1based, int(tiles_df["start_1based"].min()))
    search_end_1based = max(locus_end_1based, int(tiles_df["end_1based"].max()))

    for _, row in tiles_df.iterrows():
        tile_id = int(row["tile_id"])
        chrom = row["chrom"]
        tile_start = int(row["start_1based"])
        tile_end = int(row["end_1based"])

        rows.append(
            make_boundary_window(
                chrom=chrom,
                tile_id=tile_id,
                boundary_type="left",
                desired_boundary_1based=tile_start,
                flank=flank,
                search_start_1based=search_start_1based,
                search_end_1based=search_end_1based,
            )
        )

        rows.append(
            make_boundary_window(
                chrom=chrom,
                tile_id=tile_id,
                boundary_type="right",
                desired_boundary_1based=tile_end,
                flank=flank,
                search_start_1based=search_start_1based,
                search_end_1based=search_end_1based,
            )
        )

    df = pd.DataFrame(rows)
    return df.sort_values(["tile_id", "boundary_type"]).reset_index(drop=True)


def write_boundary_bed(df: pd.DataFrame, outpath: Path) -> None:
    if df.empty:
        outpath.write_text("")
        return

    bed = df.copy()
    bed["name"] = (
        "tile"
        + bed["tile_id"].astype(str)
        + "_"
        + bed["boundary_type"].astype(str)
        + "_boundary"
    )
    bed["score"] = "."
    bed["strand"] = "."

    bed = bed[
        [
            "chrom",
            "window_start_0based",
            "window_end_0based",
            "name",
            "score",
            "strand",
        ]
    ]
    bed.to_csv(outpath, sep="\t", header=False, index=False)


def write_unique_boundary_bed(df: pd.DataFrame, outpath: Path) -> None:
    if df.empty:
        outpath.write_text("")
        return

    uniq = (
        df[
            [
                "chrom",
                "desired_boundary_1based",
                "desired_boundary_0based",
                "window_start_1based",
                "window_end_1based",
                "window_start_0based",
                "window_end_0based",
                "window_length_bp",
            ]
        ]
        .drop_duplicates()
        .sort_values(["chrom", "window_start_0based", "window_end_0based"])
        .reset_index(drop=True)
    )

    uniq["name"] = "boundary_" + (uniq.index + 1).astype(str)
    uniq["score"] = "."
    uniq["strand"] = "."

    bed = uniq[
        [
            "chrom",
            "window_start_0based",
            "window_end_0based",
            "name",
            "score",
            "strand",
        ]
    ]
    bed.to_csv(outpath, sep="\t", header=False, index=False)


def main() -> int:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    locus = load_locus_json(Path(args.locus_json))
    tiles_df = load_tiles(Path(args.tiles_tsv))

    locus_chrom = locus["chrom"]
    locus_start_1based = int(locus["start_1based"])
    locus_end_1based = int(locus["end_1based"])

    if not (tiles_df["chrom"] == locus_chrom).all():
        raise ValueError("Not all tiles are on the same chromosome as locus.json")

    boundary_df = build_boundary_windows(
        tiles_df=tiles_df,
        locus_start_1based=locus_start_1based,
        locus_end_1based=locus_end_1based,
        flank=args.boundary_flank,
    )

    summary = {
        "chrom": locus_chrom,
        "locus_start_1based": locus_start_1based,
        "locus_end_1based": locus_end_1based,
        "search_start_1based": int(min(locus_start_1based, tiles_df["start_1based"].min())),
        "search_end_1based": int(max(locus_end_1based, tiles_df["end_1based"].max())),
        "boundary_flank_bp": args.boundary_flank,
        "n_tiles": int(tiles_df.shape[0]),
        "n_boundary_windows": int(boundary_df.shape[0]),
        "n_unique_desired_boundaries": int(
            boundary_df["desired_boundary_1based"].nunique()
        ),
    }

    boundary_df.to_csv(outdir / "boundary_windows.tsv", sep="\t", index=False)
    write_boundary_bed(boundary_df, outdir / "boundary_windows.bed")
    write_unique_boundary_bed(boundary_df, outdir / "boundary_windows.unique.bed")
    (outdir / "boundary_window_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
