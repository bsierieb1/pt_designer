#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class Segment:
    start_idx: int
    end_idx: int
    chrom: str
    start_1based: int
    end_1based: int

    @property
    def length_bp(self) -> int:
        return self.end_1based - self.start_1based + 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate non-overlapping hotspot tile-plan candidates from target intervals."
    )
    p.add_argument("--targets-tsv", required=True, help="Normalized hotspot targets TSV.")
    p.add_argument("--locus-json", required=True, help="locus.json from Step 1, used for metadata validation.")
    p.add_argument("--outdir", required=True)
    p.add_argument("--target-tile-size", type=int, default=10000)
    p.add_argument("--max-tile-size", type=int, required=True)
    p.add_argument("--min-tile-gap-bp", type=int, default=500)
    p.add_argument("--max-extra-tiles", type=int, default=2)
    p.add_argument("--max-candidate-plans", type=int, default=60)
    p.add_argument("--max-partitions-per-index", type=int, default=200)
    return p.parse_args()


def load_targets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = ["target_id", "chrom", "start_1based", "end_1based"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"targets TSV missing required columns: {missing}")

    df = df.copy()
    df["target_id"] = df["target_id"].astype(int)
    df["chrom"] = df["chrom"].astype(str)
    df["start_1based"] = pd.to_numeric(df["start_1based"], errors="coerce").astype(int)
    df["end_1based"] = pd.to_numeric(df["end_1based"], errors="coerce").astype(int)
    bad = df["end_1based"] < df["start_1based"]
    if bad.any():
        raise ValueError("Targets contain end < start after normalization.")
    if df["chrom"].nunique() != 1:
        raise ValueError("Hotspot mode currently requires all targets to be on one chromosome.")
    return df.sort_values(["chrom", "start_1based", "end_1based", "target_id"]).reset_index(drop=True)


def make_segments(targets: pd.DataFrame, max_tile_size: int) -> Dict[int, List[Segment]]:
    rows = targets.to_dict("records")
    by_start: Dict[int, List[Segment]] = {i: [] for i in range(len(rows))}

    for i in range(len(rows)):
        chrom = str(rows[i]["chrom"])
        start = int(rows[i]["start_1based"])
        end = int(rows[i]["end_1based"])
        for j in range(i, len(rows)):
            if str(rows[j]["chrom"]) != chrom:
                break
            start = min(start, int(rows[j]["start_1based"]))
            end = max(end, int(rows[j]["end_1based"]))
            if end - start + 1 > max_tile_size:
                break
            by_start[i].append(Segment(i, j, chrom, start, end))
    return by_start


def segment_gap_ok(left: Segment, right: Segment, min_gap: int) -> bool:
    return right.start_1based - left.end_1based - 1 >= min_gap


def enumerate_partitions(
    targets: pd.DataFrame,
    max_tile_size: int,
    min_gap: int,
    max_extra_tiles: int,
    max_partitions_per_index: int,
) -> List[List[Segment]]:
    n = len(targets)
    by_start = make_segments(targets, max_tile_size)
    dp: List[List[List[Segment]]] = [[] for _ in range(n + 1)]
    dp[0] = [[]]

    for i in range(n):
        if not dp[i]:
            continue
        for prefix in dp[i]:
            prev = prefix[-1] if prefix else None
            for seg in by_start.get(i, []):
                if prev is not None and not segment_gap_ok(prev, seg, min_gap):
                    continue
                cand = prefix + [seg]
                dp[seg.end_idx + 1].append(cand)

        for k in range(i + 1, n + 1):
            if len(dp[k]) > max_partitions_per_index:
                dp[k] = sorted(dp[k], key=partition_sort_key)[:max_partitions_per_index]

    partitions = dp[n]
    if not partitions:
        raise ValueError(
            "No valid hotspot tiling could cover all targets under the current max tile size "
            f"and {min_gap} bp minimum tile gap."
        )

    min_tiles = min(len(p) for p in partitions)
    max_tiles = min_tiles + max(0, max_extra_tiles)
    partitions = [p for p in partitions if len(p) <= max_tiles]
    return sorted(partitions, key=partition_sort_key)


def partition_sort_key(partition: List[Segment]) -> Tuple[int, int, int]:
    total_span = sum(seg.length_bp for seg in partition)
    max_span = max((seg.length_bp for seg in partition), default=0)
    return len(partition), total_span, max_span


def plan_from_partition(
    partition: List[Segment],
    *,
    target_tile_size: int,
    max_tile_size: int,
    min_gap: int,
    strategy: str,
) -> Optional[pd.DataFrame]:
    rows = []
    tight_spans = [(seg.start_1based, seg.end_1based) for seg in partition]

    for idx, seg in enumerate(partition):
        tight_start, tight_end = tight_spans[idx]
        tight_len = tight_end - tight_start + 1
        if tight_len > max_tile_size:
            return None

        left_budget = max_tile_size
        right_budget = max_tile_size
        if idx > 0:
            prev_end = tight_spans[idx - 1][1]
            left_budget = max(0, (tight_start - prev_end - 1 - min_gap) // 2)
        if idx < len(partition) - 1:
            next_start = tight_spans[idx + 1][0]
            right_budget = max(0, (next_start - tight_end - 1 - min_gap) // 2)

        desired_len = max(tight_len, min(target_tile_size, max_tile_size))
        if strategy == "tight":
            desired_len = tight_len
        elif strategy == "max_balanced":
            desired_len = max_tile_size

        extra = min(max_tile_size - tight_len, max(0, desired_len - tight_len))
        if strategy in {"tight"}:
            left_pad = 0
            right_pad = 0
        elif strategy in {"balanced", "max_balanced"}:
            left_pad = min(left_budget, extra // 2)
            right_pad = min(right_budget, extra - left_pad)
            leftover = extra - left_pad - right_pad
            if leftover > 0:
                add_left = min(left_budget - left_pad, leftover)
                left_pad += add_left
                leftover -= add_left
            if leftover > 0:
                right_pad += min(right_budget - right_pad, leftover)
        elif strategy == "left_biased":
            left_pad = min(left_budget, extra)
            right_pad = min(right_budget, extra - left_pad)
        elif strategy == "right_biased":
            right_pad = min(right_budget, extra)
            left_pad = min(left_budget, extra - right_pad)
        else:
            raise ValueError(f"Unsupported hotspot padding strategy: {strategy}")

        start = tight_start - int(left_pad)
        end = tight_end + int(right_pad)
        length = end - start + 1
        if length > max_tile_size:
            return None
        rows.append(
            {
                "tile_id": idx + 1,
                "chrom": seg.chrom,
                "start_1based": int(start),
                "end_1based": int(end),
                "tile_start_1based": int(start),
                "tile_end_1based": int(end),
                "start_0based": int(start) - 1,
                "end_0based": int(end),
                "length_bp": int(length),
                "target_start_1based": int(tight_start),
                "target_end_1based": int(tight_end),
                "target_count": int(seg.end_idx - seg.start_idx + 1),
                "target_ids": ",".join(str(i + 1) for i in range(seg.start_idx, seg.end_idx + 1)),
            }
        )

    out = pd.DataFrame(rows)
    if not validate_plan_geometry(out, max_tile_size=max_tile_size, min_gap=min_gap):
        return None
    return out


def validate_plan_geometry(df: pd.DataFrame, *, max_tile_size: int, min_gap: int) -> bool:
    if df.empty:
        return False
    work = df.sort_values(["start_1based", "end_1based"]).reset_index(drop=True)
    lengths = work["end_1based"] - work["start_1based"] + 1
    if (lengths > max_tile_size).any():
        return False
    for i in range(len(work) - 1):
        gap = int(work.loc[i + 1, "start_1based"]) - int(work.loc[i, "end_1based"]) - 1
        if gap < min_gap:
            return False
    return True


def geometry_signature(df: pd.DataFrame) -> str:
    work = df.sort_values(["start_1based", "end_1based"]).reset_index(drop=True)
    return "|".join(
        f"{r.chrom}:{int(r.start_1based)}-{int(r.end_1based)}"
        for r in work.itertuples()
    )


def write_bed(df: pd.DataFrame, outpath: Path) -> None:
    if df.empty:
        outpath.write_text("")
        return
    bed = df[["chrom", "start_0based", "end_0based", "tile_id"]].copy()
    bed["score"] = "."
    bed["strand"] = "."
    bed.to_csv(outpath, sep="\t", header=False, index=False)


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.target_tile_size > args.max_tile_size:
        raise ValueError("Target tile size must be <= Max tile size in hotspot mode.")

    with Path(args.locus_json).open() as fh:
        locus = json.load(fh)

    targets = load_targets(Path(args.targets_tsv))
    if targets["chrom"].iloc[0] != locus.get("chrom"):
        raise ValueError("Target chromosome does not match locus.json chromosome.")

    partitions = enumerate_partitions(
        targets,
        max_tile_size=args.max_tile_size,
        min_gap=args.min_tile_gap_bp,
        max_extra_tiles=args.max_extra_tiles,
        max_partitions_per_index=args.max_partitions_per_index,
    )

    strategies = ["balanced", "left_biased", "right_biased", "max_balanced"]
    plans: List[Dict] = []
    plan_rows: List[pd.DataFrame] = []
    seen = set()

    for partition in partitions:
        for strategy in strategies:
            plan = plan_from_partition(
                partition,
                target_tile_size=args.target_tile_size,
                max_tile_size=args.max_tile_size,
                min_gap=args.min_tile_gap_bp,
                strategy=strategy,
            )
            if plan is None:
                continue
            sig = geometry_signature(plan)
            if sig in seen:
                continue
            seen.add(sig)
            pid = f"hotspot_{len(plans) + 1:03d}"
            plan = plan.copy()
            plan["candidate_plan_id"] = pid
            plan["strategy"] = strategy
            plan["strategy_details"] = f"n_tiles={len(partition)};target_tile_size={args.target_tile_size}"
            total_tile_bp = int(plan["length_bp"].sum())
            plans.append(
                {
                    "candidate_plan_id": pid,
                    "strategy": strategy,
                    "strategy_details": plan["strategy_details"].iloc[0],
                    "n_tiles": int(plan.shape[0]),
                    "total_tile_bp": total_tile_bp,
                    "max_tile_bp": int(plan["length_bp"].max()),
                    "geometry_signature": sig,
                }
            )
            plan_rows.append(plan)
            if len(plans) >= args.max_candidate_plans:
                break
        if len(plans) >= args.max_candidate_plans:
            break

    if not plans:
        raise ValueError("No valid hotspot tile plans were generated.")

    plans_df = pd.DataFrame(plans).sort_values(
        ["n_tiles", "total_tile_bp", "candidate_plan_id"],
        kind="stable",
    ).reset_index(drop=True)
    cand_df = pd.concat(plan_rows, ignore_index=True, sort=False)
    keep_ids = set(plans_df["candidate_plan_id"])
    cand_df = cand_df[cand_df["candidate_plan_id"].isin(keep_ids)].copy()

    baseline_id = plans_df.iloc[0]["candidate_plan_id"]
    baseline = cand_df[cand_df["candidate_plan_id"] == baseline_id].copy()
    baseline = baseline.sort_values(["tile_id"]).reset_index(drop=True)

    plans_df.to_csv(outdir / "candidate_plans_summary.tsv", sep="\t", index=False)
    cand_df.to_csv(outdir / "tile_plan_candidates.tsv", sep="\t", index=False)
    baseline.to_csv(outdir / "tiles_phasing_optimized.tsv", sep="\t", index=False)
    write_bed(baseline, outdir / "tiles_phasing_optimized.bed")
    pd.DataFrame(
        columns=[
            "left_tile_id",
            "right_tile_id",
            "chrom",
            "overlap_start_1based",
            "overlap_end_1based",
            "overlap_length_bp",
            "overlap_snp_count",
            "overlap_snp_density_per_kb",
        ]
    ).to_csv(outdir / "tile_overlaps_phasing_optimized.tsv", sep="\t", index=False)

    summary = {
        "mode": "hotspot",
        "n_targets": int(targets.shape[0]),
        "n_candidate_plans": int(plans_df.shape[0]),
        "min_n_tiles": int(plans_df["n_tiles"].min()),
        "max_n_tiles": int(plans_df["n_tiles"].max()),
        "target_tile_size": int(args.target_tile_size),
        "max_tile_size": int(args.max_tile_size),
        "min_tile_gap_bp": int(args.min_tile_gap_bp),
    }
    (outdir / "tile_design_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
