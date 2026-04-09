#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def normalize_tile_id(df: pd.DataFrame) -> pd.DataFrame:
    if "tile_id" not in df.columns:
        raise ValueError("selected_pairs_final.tsv must contain tile_id")
    df = df.copy()
    df["tile_id_numeric"] = (
        df["tile_id"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    )
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Assign alternating tiles to odd/even pools.")
    ap.add_argument("--selected-pairs-final-tsv", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.selected_pairs_final_tsv, sep="\t")
    df = normalize_tile_id(df)
    df = df.sort_values(["tile_id_numeric"]).reset_index(drop=True)

    df["tile_order"] = range(1, len(df) + 1)
    df["pool"] = df["tile_order"].map(lambda x: "odd" if x % 2 == 1 else "even")
    df["pool_label"] = df["pool"].map({"odd": "Pool_Odd", "even": "Pool_Even"})

    out = df[["tile_id", "tile_id_numeric", "tile_order", "pool", "pool_label"]].copy()
    out.to_csv(outdir / "tile_pool_assignment.tsv", sep="\t", index=False)

    summary = (
        out.groupby("pool", dropna=False)
        .agg(n_tiles=("tile_id", "count"), min_tile=("tile_id_numeric", "min"), max_tile=("tile_id_numeric", "max"))
        .reset_index()
    )
    summary.to_csv(outdir / "tile_pool_assignment_summary.tsv", sep="\t", index=False)
    print(f"[DONE] Wrote {outdir / 'tile_pool_assignment.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
