#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter raw PAM candidates by PureTarget boundary orientation rules."
    )
    parser.add_argument(
        "--candidates-raw-tsv",
        required=True,
        help="Path to candidates_raw.tsv from Step 4.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )
    return parser.parse_args()


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    required = [
        "tile_id",
        "chrom",
        "boundary_type",
        "strand",
        "guide_seq",
        "pam_seq",
        "cut_site_1based",
        "desired_boundary_1based",
        "distance_cut_to_boundary_bp",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"candidates_raw.tsv missing required columns: {missing}")

    df["tile_id"] = df["tile_id"].astype(int)
    df["cut_site_1based"] = df["cut_site_1based"].astype(int)
    df["desired_boundary_1based"] = df["desired_boundary_1based"].astype(int)
    df["distance_cut_to_boundary_bp"] = df["distance_cut_to_boundary_bp"].astype(int)

    return df.sort_values(
        ["tile_id", "boundary_type", "strand", "cut_site_1based", "guide_seq"]
    ).reset_index(drop=True)


def apply_orientation_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # PureTarget bracketing logic:
    #   left boundary  -> plus strand
    #   right boundary -> minus strand
    valid_mask = (
        ((out["boundary_type"] == "left") & (out["strand"] == "+")) |
        ((out["boundary_type"] == "right") & (out["strand"] == "-"))
    )

    out["is_valid_orientation"] = valid_mask
    out["distance_abs_to_boundary_bp"] = out["distance_cut_to_boundary_bp"].abs()

    out["orientation_rule"] = ""
    out.loc[out["boundary_type"] == "left", "orientation_rule"] = "left_requires_plus"
    out.loc[out["boundary_type"] == "right", "orientation_rule"] = "right_requires_minus"

    out["orientation_status"] = "invalid"
    out.loc[out["is_valid_orientation"], "orientation_status"] = "valid"

    return out


def summarize_all(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append(
        {
            "metric": "n_candidates_total",
            "value": int(df.shape[0]),
        }
    )
    rows.append(
        {
            "metric": "n_candidates_valid_orientation",
            "value": int(df["is_valid_orientation"].sum()),
        }
    )
    rows.append(
        {
            "metric": "n_candidates_invalid_orientation",
            "value": int((~df["is_valid_orientation"]).sum()),
        }
    )

    return pd.DataFrame(rows)


def summarize_by_boundary_type(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for boundary_type, sub in df.groupby("boundary_type", sort=True):
        rows.append(
            {
                "boundary_type": boundary_type,
                "n_total": int(sub.shape[0]),
                "n_valid_orientation": int(sub["is_valid_orientation"].sum()),
                "n_invalid_orientation": int((~sub["is_valid_orientation"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_tile_and_boundary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tile_id, boundary_type), sub in df.groupby(["tile_id", "boundary_type"], sort=True):
        valid = sub[sub["is_valid_orientation"]].copy()

        rows.append(
            {
                "tile_id": int(tile_id),
                "boundary_type": boundary_type,
                "n_total": int(sub.shape[0]),
                "n_valid_orientation": int(valid.shape[0]),
                "n_invalid_orientation": int(sub.shape[0] - valid.shape[0]),
                "min_abs_distance_valid_bp": (
                    int(valid["distance_abs_to_boundary_bp"].min())
                    if not valid.empty else None
                ),
                "max_abs_distance_valid_bp": (
                    int(valid["distance_abs_to_boundary_bp"].max())
                    if not valid.empty else None
                ),
            }
        )
    return pd.DataFrame(rows)


def write_bed(df: pd.DataFrame, outpath: Path) -> None:
    if df.empty:
        outpath.write_text("")
        return

    bed = df.copy()
    bed["name"] = (
        "tile"
        + bed["tile_id"].astype(str)
        + "_"
        + bed["boundary_type"].astype(str)
        + "_"
        + bed["strand"].astype(str)
        + "_"
        + bed["guide_seq"].astype(str)
    )
    bed["score"] = "."
    bed = bed[
        [
            "chrom",
            "protospacer_start_0based",
            "protospacer_end_0based",
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

    raw_df = load_candidates(Path(args.candidates_raw_tsv))
    annotated_df = apply_orientation_rules(raw_df)

    valid_df = (
        annotated_df[annotated_df["is_valid_orientation"]]
        .copy()
        .sort_values(
            ["tile_id", "boundary_type", "distance_abs_to_boundary_bp", "cut_site_1based", "guide_seq"]
        )
        .reset_index(drop=True)
    )

    invalid_df = (
        annotated_df[~annotated_df["is_valid_orientation"]]
        .copy()
        .sort_values(
            ["tile_id", "boundary_type", "distance_abs_to_boundary_bp", "cut_site_1based", "guide_seq"]
        )
        .reset_index(drop=True)
    )

    all_summary_df = summarize_all(annotated_df)
    boundary_summary_df = summarize_by_boundary_type(annotated_df)
    tile_boundary_summary_df = summarize_by_tile_and_boundary(annotated_df)

    annotated_df.to_csv(outdir / "candidates_with_orientation.tsv", sep="\t", index=False)
    valid_df.to_csv(outdir / "candidates_oriented.tsv", sep="\t", index=False)
    invalid_df.to_csv(outdir / "candidates_invalid_orientation.tsv", sep="\t", index=False)

    write_bed(valid_df, outdir / "candidates_oriented.bed")

    all_summary_df.to_csv(outdir / "orientation_summary.tsv", sep="\t", index=False)
    boundary_summary_df.to_csv(outdir / "orientation_summary_by_boundary_type.tsv", sep="\t", index=False)
    tile_boundary_summary_df.to_csv(outdir / "orientation_summary_by_tile_boundary.tsv", sep="\t", index=False)

    print(all_summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
