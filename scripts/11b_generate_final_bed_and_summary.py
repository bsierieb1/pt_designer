#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def first_existing(df: pd.DataFrame, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required columns; tried: {candidates}")
    return None


def series_first_nonnull(df: pd.DataFrame, candidates):
    found = [c for c in candidates if c in df.columns]
    if not found:
        return None
    s = None
    for c in found:
        cur = df[c]
        if s is None:
            s = cur.copy()
        else:
            s = s.fillna(cur)
    return s


def to_bed(df: pd.DataFrame) -> pd.DataFrame:
    chrom = series_first_nonnull(
        df,
        ["chrom", "tile_chrom", "left_chrom", "right_chrom"]
    )
    if chrom is None:
        raise ValueError(
            "Could not determine chromosome for BED export. "
            "Tried: chrom, tile_chrom, left_chrom, right_chrom"
        )

    start = series_first_nonnull(
        df,
        [
            "tile_start_1based",
            "start_1based",
            "left_boundary_start_1based",
            "left_protospacer_start_1based",
            "left_cut_site_1based",
            "left_pam_start_1based",
        ],
    )
    end = series_first_nonnull(
        df,
        [
            "tile_end_1based",
            "end_1based",
            "right_boundary_end_1based",
            "right_protospacer_end_1based",
            "right_cut_site_1based",
            "right_pam_end_1based",
        ],
    )

    if start is None or end is None:
        start_candidates = [
            c for c in [
                "left_boundary_start_1based",
                "right_boundary_start_1based",
                "left_protospacer_start_1based",
                "right_protospacer_start_1based",
                "left_cut_site_1based",
                "right_cut_site_1based",
                "left_pam_start_1based",
                "right_pam_start_1based",
            ] if c in df.columns
        ]
        end_candidates = [
            c for c in [
                "left_boundary_end_1based",
                "right_boundary_end_1based",
                "left_protospacer_end_1based",
                "right_protospacer_end_1based",
                "left_cut_site_1based",
                "right_cut_site_1based",
                "left_pam_end_1based",
                "right_pam_end_1based",
            ] if c in df.columns
        ]
        if not start_candidates or not end_candidates:
            raise ValueError(
                "Could not determine tile start/end columns for BED export from selected pairs."
            )
        start = df[start_candidates].apply(pd.to_numeric, errors="coerce").min(axis=1)
        end = df[end_candidates].apply(pd.to_numeric, errors="coerce").max(axis=1)

    start = pd.to_numeric(start, errors="coerce")
    end = pd.to_numeric(end, errors="coerce")

    if start.isna().any() or end.isna().any():
        bad = df.loc[start.isna() | end.isna()].head(5)
        raise ValueError(
            "BED export could not infer coordinates for some rows. "
            f"Example rows:\n{bad.to_string(index=False)}"
        )

    bed = pd.DataFrame({
        "chrom": chrom.astype(str),
        "start_0based": start.astype(int) - 1,
        "end_1based": end.astype(int),
        "name": df["tile_id"].astype(str),
        "score": pd.to_numeric(df.get("pair_score", 0), errors="coerce").fillna(0).round(0).astype(int),
        "strand": ".",
    })
    return bed


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate final BED and summary files.")
    ap.add_argument("--selected-pairs-final-tsv", required=True)
    ap.add_argument("--tile-pool-assignment-tsv", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.selected_pairs_final_tsv, sep="\t")
    pools = pd.read_csv(args.tile_pool_assignment_tsv, sep="\t")
    df = pairs.merge(pools[["tile_id", "pool", "pool_label"]], on="tile_id", how="left")

    df.to_csv(outdir / "final_tiles.tsv", sep="\t", index=False)

    full_bed = to_bed(df)
    full_bed.to_csv(outdir / "custom_targets_full.bed", sep="\t", index=False, header=False)
    to_bed(df[df["pool"] == "odd"]).to_csv(outdir / "custom_targets_odd.bed", sep="\t", index=False, header=False)
    to_bed(df[df["pool"] == "even"]).to_csv(outdir / "custom_targets_even.bed", sep="\t", index=False, header=False)

    print(f"[DONE] Wrote final BED and tile summary files to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
