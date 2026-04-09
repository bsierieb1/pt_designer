#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


REQUIRED_COLS = [
    "tile_id",
    "left_guide_seq",
    "right_guide_seq",
    "pair_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Globally select one guide pair per tile from Step 7 pair candidates. "
            "Uses a transparent dynamic-programming objective that rewards high pair scores "
            "and penalizes guide reuse across adjacent tiles."
        )
    )
    parser.add_argument(
        "--guide-pairs-tsv",
        required=True,
        help="Path to guide_pairs_top.tsv (or guide_pairs_all.tsv) from Step 7.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--max-candidates-per-tile",
        type=int,
        default=20,
        help="Maximum number of pair candidates retained per tile for global optimization. Default: 20",
    )
    parser.add_argument(
        "--adjacent-guide-reuse-penalty",
        type=float,
        default=40.0,
        help="Penalty if adjacent selected pairs reuse the exact same guide sequence. Default: 40.0",
    )
    parser.add_argument(
        "--adjacent-boundary-distance-weight",
        type=float,
        default=0.05,
        help=(
            "Weight for mismatch between adjacent shared-boundary guide distances. "
            "Uses |tile_i right distance - tile_i+1 left distance|. Default: 0.05"
        ),
    )
    parser.add_argument(
        "--duplicate-pair-penalty",
        type=float,
        default=15.0,
        help="Penalty if the exact same left/right pair is selected in adjacent tiles. Default: 15.0",
    )
    parser.add_argument(
        "--allow-missing-tiles",
        action="store_true",
        help="Allow optimization to continue if some tile_ids are missing from the input.",
    )
    return parser.parse_args()


def load_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Guide-pairs TSV missing required columns: {missing}")

    # Optional fields used by the transition objective
    optional_numeric = [
        "pair_rank_within_tile",
        "pair_mean_ontarget_score",
        "pair_min_ontarget_score",
        "pair_mean_offtarget_score",
        "pair_min_offtarget_score",
        "left_distance_abs_to_boundary_bp",
        "right_distance_abs_to_boundary_bp",
        "pair_total_distance_abs_to_boundary_bp",
        "left_final_rank_score",
        "right_final_rank_score",
    ]
    for col in optional_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["tile_id"] = df["tile_id"].astype(int)
    df["pair_score"] = pd.to_numeric(df["pair_score"], errors="coerce")
    df = df.dropna(subset=["pair_score"]).copy()

    sort_cols = [c for c in ["tile_id", "pair_rank_within_tile", "pair_score"] if c in df.columns]
    ascending = [True, True, False][: len(sort_cols)]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    else:
        df = df.sort_values(["tile_id", "pair_score"], ascending=[True, False]).reset_index(drop=True)
    return df


def trim_candidates(df: pd.DataFrame, max_candidates_per_tile: int) -> pd.DataFrame:
    parts = []
    for _, sub in df.groupby("tile_id", sort=True):
        if "pair_rank_within_tile" in sub.columns:
            sub = sub.sort_values(["pair_rank_within_tile", "pair_score"], ascending=[True, False])
        else:
            sub = sub.sort_values(["pair_score"], ascending=[False])
        parts.append(sub.head(max_candidates_per_tile).copy())
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def validate_tile_continuity(tile_ids: List[int], allow_missing_tiles: bool) -> None:
    if not tile_ids:
        raise ValueError("No tile candidates found for global optimization.")
    if allow_missing_tiles:
        return
    expected = list(range(min(tile_ids), max(tile_ids) + 1))
    if tile_ids != expected:
        raise ValueError(
            "Tile IDs are not continuous in the provided pair table. "
            f"Observed {tile_ids[:10]}{'...' if len(tile_ids) > 10 else ''}; expected continuous {expected[:10]}"
        )


def get_num(row: pd.Series, col: str, default: float = 0.0) -> float:
    if col not in row.index:
        return default
    val = row[col]
    if pd.isna(val):
        return default
    return float(val)


def transition_penalty(
    prev_row: pd.Series,
    curr_row: pd.Series,
    adjacent_guide_reuse_penalty: float,
    adjacent_boundary_distance_weight: float,
    duplicate_pair_penalty: float,
) -> Tuple[float, Dict[str, float]]:
    penalty = 0.0
    parts: Dict[str, float] = {}

    prev_guides = {str(prev_row.get("left_guide_seq", "")), str(prev_row.get("right_guide_seq", ""))}
    curr_guides = {str(curr_row.get("left_guide_seq", "")), str(curr_row.get("right_guide_seq", ""))}
    shared_guides = {g for g in prev_guides.intersection(curr_guides) if g}
    if shared_guides:
        p = adjacent_guide_reuse_penalty * len(shared_guides)
        penalty += p
        parts["adjacent_guide_reuse_penalty"] = p

    if (
        str(prev_row.get("left_guide_seq", "")) == str(curr_row.get("left_guide_seq", ""))
        and str(prev_row.get("right_guide_seq", "")) == str(curr_row.get("right_guide_seq", ""))
    ):
        penalty += duplicate_pair_penalty
        parts["duplicate_pair_penalty"] = duplicate_pair_penalty

    prev_right_dist = get_num(prev_row, "right_distance_abs_to_boundary_bp", default=0.0)
    curr_left_dist = get_num(curr_row, "left_distance_abs_to_boundary_bp", default=0.0)
    boundary_distance_delta = abs(prev_right_dist - curr_left_dist)
    if adjacent_boundary_distance_weight != 0:
        p = adjacent_boundary_distance_weight * boundary_distance_delta
        penalty += p
        parts["adjacent_boundary_distance_penalty"] = p
        parts["adjacent_boundary_distance_delta_bp"] = boundary_distance_delta

    return penalty, parts


def optimize_tiles(
    df: pd.DataFrame,
    adjacent_guide_reuse_penalty: float,
    adjacent_boundary_distance_weight: float,
    duplicate_pair_penalty: float,
) -> Tuple[pd.DataFrame, Dict]:
    tile_ids = sorted(df["tile_id"].unique().tolist())
    by_tile: Dict[int, pd.DataFrame] = {tile_id: sub.reset_index(drop=True) for tile_id, sub in df.groupby("tile_id", sort=True)}

    dp: Dict[Tuple[int, int], float] = {}
    backptr: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
    trans_meta: Dict[Tuple[int, int], Dict[str, float]] = {}

    first_tile = tile_ids[0]
    for j, row in by_tile[first_tile].iterrows():
        base_score = float(row["pair_score"])
        dp[(first_tile, j)] = base_score
        backptr[(first_tile, j)] = None
        trans_meta[(first_tile, j)] = {"transition_penalty": 0.0}

    for prev_tile, curr_tile in zip(tile_ids[:-1], tile_ids[1:]):
        prev_df = by_tile[prev_tile]
        curr_df = by_tile[curr_tile]

        for j, curr_row in curr_df.iterrows():
            best_score = None
            best_prev_key = None
            best_meta = None

            curr_pair_score = float(curr_row["pair_score"])
            for i, prev_row in prev_df.iterrows():
                prev_key = (prev_tile, i)
                if prev_key not in dp:
                    continue

                penalty, penalty_parts = transition_penalty(
                    prev_row=prev_row,
                    curr_row=curr_row,
                    adjacent_guide_reuse_penalty=adjacent_guide_reuse_penalty,
                    adjacent_boundary_distance_weight=adjacent_boundary_distance_weight,
                    duplicate_pair_penalty=duplicate_pair_penalty,
                )
                score = dp[prev_key] + curr_pair_score - penalty
                if best_score is None or score > best_score:
                    best_score = score
                    best_prev_key = prev_key
                    best_meta = {"transition_penalty": penalty, **penalty_parts}

            if best_score is not None:
                dp[(curr_tile, j)] = best_score
                backptr[(curr_tile, j)] = best_prev_key
                trans_meta[(curr_tile, j)] = best_meta or {"transition_penalty": 0.0}

    last_tile = tile_ids[-1]
    last_candidates = [(key, val) for key, val in dp.items() if key[0] == last_tile]
    if not last_candidates:
        raise RuntimeError("Global optimization failed to retain any complete path across tiles.")

    best_final_key, best_final_score = max(last_candidates, key=lambda kv: kv[1])

    selected_rows: List[pd.Series] = []
    key = best_final_key
    while key is not None:
        tile_id, idx = key
        row = by_tile[tile_id].iloc[idx].copy()
        meta = trans_meta.get(key, {"transition_penalty": 0.0})
        row["global_objective_to_here"] = dp[key]
        for mkey, mval in meta.items():
            row[mkey] = mval
        selected_rows.append(row)
        key = backptr.get(key)

    selected_rows = list(reversed(selected_rows))
    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected_df["global_selected_rank"] = range(1, len(selected_df) + 1)

    total_pair_score = float(selected_df["pair_score"].sum()) if not selected_df.empty else 0.0
    total_transition_penalty = float(selected_df.get("transition_penalty", pd.Series(dtype=float)).fillna(0).sum()) if not selected_df.empty else 0.0

    summary = {
        "n_tiles_optimized": int(len(tile_ids)),
        "n_pairs_considered": int(df.shape[0]),
        "n_pairs_selected": int(selected_df.shape[0]),
        "total_pair_score": round(total_pair_score, 6),
        "total_transition_penalty": round(total_transition_penalty, 6),
        "global_objective_score": round(float(best_final_score), 6),
        "adjacent_guide_reuse_penalty": float(adjacent_guide_reuse_penalty),
        "adjacent_boundary_distance_weight": float(adjacent_boundary_distance_weight),
        "duplicate_pair_penalty": float(duplicate_pair_penalty),
    }
    return selected_df, summary


def build_alternatives(selected_df: pd.DataFrame, all_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    rows: List[Dict] = []
    if selected_df.empty:
        return pd.DataFrame()

    for _, chosen in selected_df.iterrows():
        tile_id = int(chosen["tile_id"])
        sub = all_df[all_df["tile_id"] == tile_id].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["pair_score"], ascending=[False]).head(top_k)
        for rank, (_, row) in enumerate(sub.iterrows(), start=1):
            rows.append(
                {
                    "tile_id": tile_id,
                    "alternative_rank_within_tile": rank,
                    "is_selected_globally": bool(
                        str(row.get("left_guide_seq", "")) == str(chosen.get("left_guide_seq", ""))
                        and str(row.get("right_guide_seq", "")) == str(chosen.get("right_guide_seq", ""))
                    ),
                    **row.to_dict(),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pairs_df = load_pairs(Path(args.guide_pairs_tsv))
    pairs_df = trim_candidates(pairs_df, max_candidates_per_tile=args.max_candidates_per_tile)

    tile_ids = sorted(pairs_df["tile_id"].unique().tolist())
    validate_tile_continuity(tile_ids, allow_missing_tiles=args.allow_missing_tiles)

    selected_df, summary = optimize_tiles(
        df=pairs_df,
        adjacent_guide_reuse_penalty=args.adjacent_guide_reuse_penalty,
        adjacent_boundary_distance_weight=args.adjacent_boundary_distance_weight,
        duplicate_pair_penalty=args.duplicate_pair_penalty,
    )

    alternatives_df = build_alternatives(selected_df, pairs_df, top_k=min(5, args.max_candidates_per_tile))

    selected_df.to_csv(outdir / "guide_pairs_global_selected.tsv", sep="\t", index=False)
    pairs_df.to_csv(outdir / "guide_pairs_global_candidates_trimmed.tsv", sep="\t", index=False)
    alternatives_df.to_csv(outdir / "guide_pairs_global_alternatives.tsv", sep="\t", index=False)
    pd.DataFrame([summary]).to_csv(outdir / "global_optimization_summary.tsv", sep="\t", index=False)

    with (outdir / "global_optimization_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(pd.DataFrame([summary]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
