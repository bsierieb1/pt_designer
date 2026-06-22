#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate and select a complete hotspot tile plan.")
    p.add_argument("--candidate-plans-tsv", required=True)
    p.add_argument("--tile-plan-candidates-tsv", required=True)
    p.add_argument("--targets-tsv", required=True)
    p.add_argument("--locus-json", required=True)
    p.add_argument("--scripts-dir", required=True)
    p.add_argument("--base-output-dir", required=True)
    p.add_argument("--common-snps-bed")
    p.add_argument("--ucsc-cache-dir")
    p.add_argument("--max-tile-size-bp", type=int, required=True)
    p.add_argument("--min-tile-gap-bp", type=int, default=500)
    p.add_argument("--min-guide-target-distance-bp", type=int, default=50)
    p.add_argument("--max-candidate-plans", type=int, default=60)
    p.add_argument("--disable-local-repair", action="store_true")
    p.add_argument("--local-repair-score-threshold", type=float, default=-200.0)
    p.add_argument("--local-repair-max-block-radius", type=int, default=1)
    p.add_argument("--local-repair-max-candidates", type=int, default=24)
    p.add_argument("--local-repair-shift-step-bp", type=int, default=100)
    p.add_argument("--local-repair-max-placements-per-segment", type=int, default=25)
    p.add_argument("--local-repair-min-score-improvement", type=float, default=0.0)
    p.add_argument("--show-substep-output", action="store_true")
    return p.parse_args()


def run(cmd: List[str], quiet: bool = True) -> None:
    if quiet:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            if proc.stdout:
                print(proc.stdout, end="")
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr)
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    else:
        print("[RUN]", " ".join(str(x) for x in cmd), flush=True)
        subprocess.run(cmd, check=True)


def standardize_tile_plan(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "tile_id" not in out.columns:
        out["tile_id"] = range(1, len(out) + 1)
    chrom_col = next((c for c in ["chrom", "tile_chrom", "chr"] if c in out.columns), None)
    start_col = next((c for c in ["start_1based", "tile_start_1based"] if c in out.columns), None)
    end_col = next((c for c in ["end_1based", "tile_end_1based"] if c in out.columns), None)
    if not all([chrom_col, start_col, end_col]):
        raise ValueError("Tile plan candidate missing chrom/start/end columns.")
    norm = pd.DataFrame(
        {
            "tile_id": pd.to_numeric(out["tile_id"], errors="coerce").astype(int),
            "tile_chrom": out[chrom_col].astype(str),
            "start_1based": pd.to_numeric(out[start_col], errors="coerce").astype(int),
            "end_1based": pd.to_numeric(out[end_col], errors="coerce").astype(int),
        }
    )
    norm = norm.sort_values(["start_1based", "end_1based", "tile_id"]).reset_index(drop=True)
    norm["tile_id"] = range(1, len(norm) + 1)
    norm["chrom"] = norm["tile_chrom"]
    norm["tile_start_1based"] = norm["start_1based"]
    norm["tile_end_1based"] = norm["end_1based"]
    norm["start_0based"] = norm["start_1based"] - 1
    norm["end_0based"] = norm["end_1based"]
    norm["length_bp"] = norm["end_1based"] - norm["start_1based"] + 1
    return norm


def rerun_pipeline(tiles_tsv: Path, outdir: Path, args: argparse.Namespace) -> Path:
    scripts = Path(args.scripts_dir)
    py = sys.executable
    outdir.mkdir(parents=True, exist_ok=True)
    quiet = not args.show_substep_output

    step03 = outdir / "03_boundary_windows"
    step04 = outdir / "04_fetch_guides"
    step05 = outdir / "05_oriented"
    step06a = outdir / "06a_annotated"
    step06d = outdir / "06d_ranked"
    step07 = outdir / "07_paired"
    step08 = outdir / "08_global"

    run([py, str(scripts / "03_make_boundary_windows.py"), "--locus-json", args.locus_json, "--tiles-tsv", str(tiles_tsv), "--outdir", str(step03)], quiet)
    cache_dir = Path(args.ucsc_cache_dir) if args.ucsc_cache_dir else Path(args.base_output_dir) / "ucsc_cache"
    run([
        py,
        str(scripts / "04_fetch_guides.py"),
        "--boundary-windows-tsv",
        str(step03 / "boundary_windows.tsv"),
        "--outdir",
        str(step04),
        "--cache-dir",
        str(cache_dir),
    ], quiet)
    run([py, str(scripts / "05_filter_orientation.py"), "--candidates-raw-tsv", str(step04 / "candidates_raw.tsv"), "--outdir", str(step05)], quiet)
    cmd06a = [py, str(scripts / "06a_annotate_guides.py"), "--candidates-oriented-tsv", str(step05 / "candidates_oriented.tsv"), "--outdir", str(step06a)]
    if args.common_snps_bed:
        cmd06a += ["--common-snps-bed", args.common_snps_bed]
    cmd06a += [
        "--avoid-targets-tsv",
        args.targets_tsv,
        "--min-distance-to-target-bp",
        str(args.min_guide_target_distance_bp),
    ]
    run(cmd06a, quiet)
    run([py, str(scripts / "06d_rank_guides.py"), "--candidates-annotated-tsv", str(step06a / "candidates_annotated.tsv"), "--outdir", str(step06d)], quiet)
    run([py, str(scripts / "07_pair_guides.py"), "--require-nonunacceptable", "--drop-unacceptable-pairs", "--candidates-scored-tsv", str(step06d / "candidates_scored.tsv"), "--outdir", str(step07)], quiet)
    run([py, str(scripts / "08_optimize_tiles.py"), "--guide-pairs-tsv", str(step07 / "guide_pairs_top.tsv"), "--outdir", str(step08)], quiet)
    selected = step08 / "guide_pairs_global_selected.tsv"
    if not selected.exists():
        raise FileNotFoundError(f"Missing selected pairs output: {selected}")
    return selected


def target_coverage_metrics(tiles: pd.DataFrame, targets: pd.DataFrame) -> Dict:
    rows = []
    n_uncovered = 0
    n_multi = 0
    for _, target in targets.iterrows():
        hits = tiles[
            (tiles["tile_chrom"] == str(target["chrom"]))
            & (tiles["start_1based"] <= int(target["start_1based"]))
            & (tiles["end_1based"] >= int(target["end_1based"]))
        ]
        n_hits = int(len(hits))
        if n_hits == 0:
            n_uncovered += 1
        if n_hits > 1:
            n_multi += 1
        rows.append(
            {
                "target_id": int(target["target_id"]),
                "n_covering_tiles": n_hits,
                "covering_tile_ids": ",".join(hits["tile_id"].astype(str).tolist()),
            }
        )
    return {
        "n_uncovered_targets": int(n_uncovered),
        "n_targets_covered_by_multiple_tiles": int(n_multi),
        "target_coverage": rows,
    }


def gap_metrics(tiles: pd.DataFrame, min_gap: int) -> Dict:
    rows = []
    violations = 0
    min_observed = None
    for i in range(len(tiles) - 1):
        left = tiles.iloc[i]
        right = tiles.iloc[i + 1]
        gap = int(right["start_1based"]) - int(left["end_1based"]) - 1
        if min_observed is None or gap < min_observed:
            min_observed = gap
        if gap < min_gap:
            violations += 1
            rows.append(
                {
                    "left_tile_id": int(left["tile_id"]),
                    "right_tile_id": int(right["tile_id"]),
                    "gap_bp": int(gap),
                }
            )
    return {
        "min_intertile_gap_bp": int(min_observed) if min_observed is not None else None,
        "n_gap_violations": int(violations),
        "gap_violations": rows,
    }


def tile_signature(tiles: pd.DataFrame) -> str:
    work = standardize_tile_plan(tiles)
    return "|".join(
        f"{row.tile_chrom}:{int(row.start_1based)}-{int(row.end_1based)}"
        for row in work.itertuples(index=False)
    )


def selected_weak_tile_ids(selected: pd.DataFrame, score_threshold: float) -> List[int]:
    if selected.empty or "tile_id" not in selected.columns or "pair_score" not in selected.columns:
        return []
    work = selected.copy()
    work["tile_id"] = pd.to_numeric(work["tile_id"], errors="coerce")
    work["pair_score"] = pd.to_numeric(work["pair_score"], errors="coerce")
    weak = work[work["pair_score"] < score_threshold].dropna(subset=["tile_id", "pair_score"])
    weak = weak.sort_values(["pair_score", "tile_id"], kind="stable")
    return [int(x) for x in weak["tile_id"].tolist()]


def tile_target_assignments(tiles: pd.DataFrame, targets: pd.DataFrame) -> Dict[int, List[int]]:
    assignments: Dict[int, List[int]] = {}
    for _, tile in tiles.iterrows():
        hits = targets[
            (targets["chrom"].astype(str) == str(tile["tile_chrom"]))
            & (pd.to_numeric(targets["start_1based"], errors="coerce") >= int(tile["start_1based"]))
            & (pd.to_numeric(targets["end_1based"], errors="coerce") <= int(tile["end_1based"]))
        ]
        assignments[int(tile["tile_id"])] = [int(x) for x in hits["target_id"].tolist()]
    return assignments


def valid_complete_plan(tiles: pd.DataFrame, targets: pd.DataFrame, args: argparse.Namespace) -> bool:
    if tiles.empty:
        return False
    lengths = pd.to_numeric(tiles["length_bp"], errors="coerce")
    if lengths.isna().any() or (lengths > args.max_tile_size_bp).any():
        return False
    coverage = target_coverage_metrics(tiles, targets)
    gaps = gap_metrics(tiles, args.min_tile_gap_bp)
    return (
        coverage["n_uncovered_targets"] == 0
        and coverage["n_targets_covered_by_multiple_tiles"] == 0
        and gaps["n_gap_violations"] == 0
    )


def local_segmentations(
    target_block: pd.DataFrame,
    n_segments: int,
    *,
    max_tile_size: int,
    min_gap: int,
) -> List[List[Dict]]:
    rows = target_block.sort_values(["start_1based", "end_1based", "target_id"]).reset_index(drop=True)
    if rows.empty or n_segments <= 0:
        return []

    out: List[List[Dict]] = []

    def rec(start_idx: int, remaining: int, prefix: List[Dict]) -> None:
        if remaining == 0:
            if start_idx == len(rows):
                out.append(prefix.copy())
            return
        min_remaining_targets = remaining - 1
        max_end_idx = len(rows) - min_remaining_targets - 1
        for end_idx in range(start_idx, max_end_idx + 1):
            sub = rows.iloc[start_idx : end_idx + 1]
            tight_start = int(sub["start_1based"].min())
            tight_end = int(sub["end_1based"].max())
            if tight_end - tight_start + 1 > max_tile_size:
                break
            seg = {
                "chrom": str(sub["chrom"].iloc[0]),
                "target_start_idx": int(start_idx),
                "target_end_idx": int(end_idx),
                "target_start_1based": tight_start,
                "target_end_1based": tight_end,
                "target_ids": [int(x) for x in sub["target_id"].tolist()],
            }
            if prefix:
                prev = prefix[-1]
                if tight_start - int(prev["target_end_1based"]) - 1 < min_gap:
                    continue
            rec(end_idx + 1, remaining - 1, prefix + [seg])

    rec(0, n_segments, [])
    return out


def sampled_starts(start_min: int, start_max: int, step: int, max_count: int, anchors: List[int]) -> List[int]:
    if start_min > start_max:
        return []
    vals = {int(start_min), int(start_max)}
    step = max(1, int(step))
    vals.update(range(int(start_min), int(start_max) + 1, step))
    for anchor in anchors:
        if start_min <= int(anchor) <= start_max:
            vals.add(int(anchor))
    ordered = sorted(vals)
    if len(ordered) <= max_count:
        return ordered
    keep = {ordered[0], ordered[-1]}
    keep.update(int(anchor) for anchor in anchors if start_min <= int(anchor) <= start_max)
    slots = max(0, max_count - len(keep))
    if slots > 0:
        if slots == 1:
            keep.add(ordered[len(ordered) // 2])
        else:
            for i in range(slots):
                idx = round(i * (len(ordered) - 1) / (slots - 1))
                keep.add(ordered[idx])
    return sorted(keep)


def placements_for_segment(
    seg: Dict,
    *,
    accepted_tile: Optional[pd.Series],
    max_tile_size: int,
    shift_step: int,
    max_placements: int,
) -> List[Dict]:
    tight_start = int(seg["target_start_1based"])
    tight_end = int(seg["target_end_1based"])
    tight_len = tight_end - tight_start + 1
    if tight_len > max_tile_size:
        return []

    accepted_len = None
    accepted_start = None
    if accepted_tile is not None:
        accepted_len = int(accepted_tile["length_bp"])
        accepted_start = int(accepted_tile["start_1based"])

    length_options = [max_tile_size, accepted_len, tight_len]
    length_options = sorted({int(x) for x in length_options if x is not None and int(x) >= tight_len and int(x) <= max_tile_size}, reverse=True)

    placements = []
    seen = set()
    for length in length_options:
        start_min = tight_end - length + 1
        start_max = tight_start
        center = (start_min + start_max) // 2
        anchors = [center]
        if accepted_start is not None:
            anchors.append(accepted_start)
        for start in sampled_starts(start_min, start_max, shift_step, max_placements, anchors):
            end = start + length - 1
            sig = (start, end)
            if sig in seen:
                continue
            seen.add(sig)
            placements.append(
                {
                    "chrom": str(seg["chrom"]),
                    "start_1based": int(start),
                    "end_1based": int(end),
                    "length_bp": int(length),
                    "target_ids": ",".join(str(x) for x in seg["target_ids"]),
                }
            )
    return placements


def combine_block_placements(
    segment_placements: List[List[Dict]],
    *,
    left_neighbor: Optional[pd.Series],
    right_neighbor: Optional[pd.Series],
    min_gap: int,
    max_combinations: int,
) -> List[List[Dict]]:
    out: List[List[Dict]] = []

    def rec(idx: int, prefix: List[Dict]) -> None:
        if len(out) >= max_combinations:
            return
        if idx == len(segment_placements):
            if right_neighbor is not None and prefix:
                gap = int(right_neighbor["start_1based"]) - int(prefix[-1]["end_1based"]) - 1
                if gap < min_gap:
                    return
            out.append(prefix.copy())
            return
        for placement in segment_placements[idx]:
            if idx == 0 and left_neighbor is not None:
                gap = int(placement["start_1based"]) - int(left_neighbor["end_1based"]) - 1
                if gap < min_gap:
                    continue
            if prefix:
                gap = int(placement["start_1based"]) - int(prefix[-1]["end_1based"]) - 1
                if gap < min_gap:
                    continue
            rec(idx + 1, prefix + [placement])

    rec(0, [])
    return out


def build_repair_tile_plan(
    accepted_tiles: pd.DataFrame,
    block_start: int,
    block_end: int,
    placements: List[Dict],
) -> pd.DataFrame:
    rows = []
    placement_by_tile = {
        tile_id: placement
        for tile_id, placement in zip(range(block_start, block_end + 1), placements)
    }
    for _, tile in accepted_tiles.sort_values("tile_id").iterrows():
        tile_id = int(tile["tile_id"])
        if block_start <= tile_id <= block_end:
            placement = placement_by_tile[tile_id]
            rows.append(
                {
                    "tile_id": tile_id,
                    "tile_chrom": placement["chrom"],
                    "start_1based": int(placement["start_1based"]),
                    "end_1based": int(placement["end_1based"]),
                }
            )
        else:
            rows.append(
                {
                    "tile_id": tile_id,
                    "tile_chrom": str(tile["tile_chrom"]),
                    "start_1based": int(tile["start_1based"]),
                    "end_1based": int(tile["end_1based"]),
                }
            )
    return standardize_tile_plan(pd.DataFrame(rows))


def generate_local_repair_plans(
    accepted_tiles: pd.DataFrame,
    accepted_selected: pd.DataFrame,
    targets: pd.DataFrame,
    existing_signatures: set,
    args: argparse.Namespace,
) -> List[Dict]:
    weak_tile_ids = selected_weak_tile_ids(accepted_selected, args.local_repair_score_threshold)
    if not weak_tile_ids:
        return []

    assignments = tile_target_assignments(accepted_tiles, targets)
    repairs: List[Dict] = []
    seen = set(existing_signatures)
    n_tiles = int(len(accepted_tiles))
    n_radius_levels = max(0, int(args.local_repair_max_block_radius)) + 1
    per_block_limit = max(1, int(args.local_repair_max_candidates) // max(1, len(weak_tile_ids) * n_radius_levels))

    for weak_tile_id in weak_tile_ids:
        for radius in range(0, max(0, int(args.local_repair_max_block_radius)) + 1):
            block_added = 0
            block_start = max(1, weak_tile_id - radius)
            block_end = min(n_tiles, weak_tile_id + radius)
            block_tile_ids = list(range(block_start, block_end + 1))
            target_ids = []
            for tile_id in block_tile_ids:
                target_ids.extend(assignments.get(tile_id, []))
            target_ids = sorted(set(target_ids))
            if not target_ids:
                continue
            target_block = targets[targets["target_id"].isin(target_ids)].copy()
            if target_block.empty:
                continue

            segmentations = local_segmentations(
                target_block,
                len(block_tile_ids),
                max_tile_size=args.max_tile_size_bp,
                min_gap=args.min_tile_gap_bp,
            )
            left_neighbor = accepted_tiles[accepted_tiles["tile_id"] == block_start - 1]
            right_neighbor = accepted_tiles[accepted_tiles["tile_id"] == block_end + 1]
            left_neighbor_row = left_neighbor.iloc[0] if not left_neighbor.empty else None
            right_neighbor_row = right_neighbor.iloc[0] if not right_neighbor.empty else None

            for segmentation in segmentations:
                segment_placements = []
                for offset, seg in enumerate(segmentation):
                    tile_id = block_start + offset
                    accepted_tile = accepted_tiles[accepted_tiles["tile_id"] == tile_id]
                    placements = placements_for_segment(
                        seg,
                        accepted_tile=accepted_tile.iloc[0] if not accepted_tile.empty else None,
                        max_tile_size=args.max_tile_size_bp,
                        shift_step=args.local_repair_shift_step_bp,
                        max_placements=args.local_repair_max_placements_per_segment,
                    )
                    if not placements:
                        segment_placements = []
                        break
                    segment_placements.append(placements)
                if not segment_placements:
                    continue

                combinations = combine_block_placements(
                    segment_placements,
                    left_neighbor=left_neighbor_row,
                    right_neighbor=right_neighbor_row,
                    min_gap=args.min_tile_gap_bp,
                    max_combinations=max(1, min(per_block_limit - block_added, args.local_repair_max_candidates - len(repairs))),
                )
                for combo in combinations:
                    plan = build_repair_tile_plan(accepted_tiles, block_start, block_end, combo)
                    if not valid_complete_plan(plan, targets, args):
                        continue
                    sig = tile_signature(plan)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    repairs.append(
                        {
                            "tiles": plan,
                            "weak_tile_id": int(weak_tile_id),
                            "block_start": int(block_start),
                            "block_end": int(block_end),
                            "target_ids": ",".join(str(x) for x in target_ids),
                        }
                    )
                    block_added += 1
                    if len(repairs) >= args.local_repair_max_candidates:
                        return repairs
                    if block_added >= per_block_limit:
                        break
                if block_added >= per_block_limit:
                    break
            if len(repairs) >= args.local_repair_max_candidates:
                return repairs
    return repairs


def merge_tile_context(selected: pd.DataFrame, tiles: pd.DataFrame) -> pd.DataFrame:
    keep = tiles[["tile_id", "tile_chrom", "tile_start_1based", "tile_end_1based", "length_bp"]].copy()
    out = selected.merge(keep, on="tile_id", how="left")
    return out


def evaluate_candidate(
    selected_path: Path,
    tiles: pd.DataFrame,
    targets: pd.DataFrame,
    plan_meta: Dict,
    args: argparse.Namespace,
) -> Dict:
    selected = pd.read_csv(selected_path, sep="\t") if selected_path.exists() else pd.DataFrame()
    if "tile_id" in selected.columns:
        selected["tile_id"] = pd.to_numeric(selected["tile_id"], errors="coerce").astype("Int64")

    tile_ids = set(tiles["tile_id"].astype(int).tolist())
    selected_tile_ids = set(selected["tile_id"].dropna().astype(int).tolist()) if "tile_id" in selected.columns else set()
    n_missing = len(tile_ids - selected_tile_ids)

    unacceptable = pd.Series(False, index=selected.index)
    if "pair_design_class" in selected.columns:
        unacceptable = unacceptable | selected["pair_design_class"].astype(str).str.lower().eq("unacceptable")
    for col in [
        "left_has_exact_in_locus_offtarget",
        "right_has_exact_in_locus_offtarget",
        "left_guide_too_close_to_target",
        "right_guide_too_close_to_target",
        "pair_has_guide_too_close_to_target",
    ]:
        if col in selected.columns:
            unacceptable = unacceptable | selected[col].fillna(False).astype(bool)

    pair_score = pd.to_numeric(selected.get("pair_score", pd.Series(dtype=float)), errors="coerce").fillna(0)
    total_pair_score = float(pair_score.sum()) if not pair_score.empty else 0.0
    min_pair_score = float(pair_score.min()) if not pair_score.empty else 0.0

    coverage = target_coverage_metrics(tiles, targets)
    gaps = gap_metrics(tiles, args.min_tile_gap_bp)
    size_violations = int((tiles["length_bp"] > args.max_tile_size_bp).sum())
    total_tile_bp = int(tiles["length_bp"].sum())

    n_tiles = int(len(tiles))
    score = 0.0
    score -= 50000.0 * n_tiles
    score += total_pair_score
    score += 0.5 * min_pair_score
    score -= 0.01 * total_tile_bp
    score -= 1000000.0 * int(n_missing)
    score -= 1000000.0 * int(coverage["n_uncovered_targets"])
    score -= 100000.0 * int(coverage["n_targets_covered_by_multiple_tiles"])
    score -= 100000.0 * int(gaps["n_gap_violations"])
    score -= 100000.0 * int(size_violations)
    score -= 20000.0 * int(unacceptable.sum())

    return {
        "candidate_plan_id": str(plan_meta.get("candidate_plan_id", "")),
        "strategy": str(plan_meta.get("strategy", "")),
        "strategy_details": str(plan_meta.get("strategy_details", "")),
        "n_tiles": n_tiles,
        "n_selected": int(len(selected)),
        "n_missing": int(n_missing),
        "n_unacceptable": int(unacceptable.sum()),
        "total_pair_score": total_pair_score,
        "min_pair_score": min_pair_score,
        "total_tile_bp": total_tile_bp,
        "max_tile_bp": int(tiles["length_bp"].max()) if not tiles.empty else 0,
        "size_violations": size_violations,
        "score": float(score),
        "is_valid": int(
            n_missing == 0
            and coverage["n_uncovered_targets"] == 0
            and coverage["n_targets_covered_by_multiple_tiles"] == 0
            and gaps["n_gap_violations"] == 0
            and size_violations == 0
        ),
        "n_uncovered_targets": int(coverage["n_uncovered_targets"]),
        "n_targets_covered_by_multiple_tiles": int(coverage["n_targets_covered_by_multiple_tiles"]),
        "target_coverage": json.dumps(coverage["target_coverage"]),
        "min_intertile_gap_bp": gaps["min_intertile_gap_bp"],
        "n_gap_violations": int(gaps["n_gap_violations"]),
        "gap_violations": json.dumps(gaps["gap_violations"]),
    }


def write_bed(df: pd.DataFrame, outpath: Path, name_cols: List[str]) -> None:
    if df.empty:
        outpath.write_text("")
        return
    work = df.copy()
    cols = [c for c in name_cols if c in work.columns]
    work["_name"] = work[cols].astype(str).agg("|".join, axis=1) if cols else work["tile_id"].astype(str)
    bed = pd.DataFrame(
        {
            "chrom": work["tile_chrom"].astype(str),
            "start": pd.to_numeric(work["tile_start_1based"], errors="coerce").astype(int) - 1,
            "end": pd.to_numeric(work["tile_end_1based"], errors="coerce").astype(int),
            "name": work["_name"],
        }
    )
    bed.to_csv(outpath, sep="\t", index=False, header=False)


def main() -> int:
    args = parse_args()
    base = Path(args.base_output_dir)
    outdir = base / "09h_hotspot_optimization"
    outdir.mkdir(parents=True, exist_ok=True)
    search_dir = outdir / "search_history"
    search_dir.mkdir(parents=True, exist_ok=True)

    plans = pd.read_csv(args.candidate_plans_tsv, sep="\t")
    candidates = pd.read_csv(args.tile_plan_candidates_tsv, sep="\t")
    targets = pd.read_csv(args.targets_tsv, sep="\t")
    if len(plans) > args.max_candidate_plans:
        plans = plans.head(args.max_candidate_plans).copy()

    eval_rows: List[Dict] = []
    tile_history: List[pd.DataFrame] = []
    candidate_records: Dict[str, Dict] = {}
    existing_signatures = set()
    best: Optional[Dict] = None

    for idx, (_, plan) in enumerate(plans.iterrows(), start=1):
        pid = str(plan["candidate_plan_id"])
        print(f"[HOTSPOT-CANDIDATE] {idx}/{len(plans)} {pid}", flush=True)
        plan_df = candidates[candidates["candidate_plan_id"] == pid].copy()
        if plan_df.empty:
            continue
        tiles = standardize_tile_plan(plan_df)
        existing_signatures.add(tile_signature(tiles))
        candidate_dir = outdir / "candidates" / pid
        candidate_dir.mkdir(parents=True, exist_ok=True)
        tile_path = candidate_dir / "tile_plan.tsv"
        tiles.to_csv(tile_path, sep="\t", index=False)
        try:
            selected_path = rerun_pipeline(tile_path, candidate_dir, args)
            metrics = evaluate_candidate(selected_path, tiles, targets, plan.to_dict(), args)
            metrics["round"] = 1
            eval_rows.append(metrics)
            tile_history.append(tiles.assign(round=1, candidate_plan_id=pid, strategy=metrics["strategy"], strategy_details=metrics["strategy_details"], score=metrics["score"]))
            candidate_records[pid] = {
                "candidate_dir": candidate_dir,
                "selected_path": selected_path,
                "tiles": tiles,
                "metrics": metrics,
            }
            print(f"[HOTSPOT-SCORE] {json.dumps(metrics, sort_keys=True)}", flush=True)
            if int(metrics["is_valid"]) == 1 and (best is None or float(metrics["score"]) > float(best["score"])):
                best = metrics
        except Exception as exc:
            metrics = {
                "round": 1,
                "candidate_plan_id": pid,
                "strategy": str(plan.get("strategy", "")),
                "strategy_details": str(plan.get("strategy_details", "")),
                "is_valid": 0,
                "score": -1e12,
                "failure_reason": str(exc),
            }
            eval_rows.append(metrics)
            print(f"[HOTSPOT-SKIP] {pid} {exc}", flush=True)

    if best is None:
        raise RuntimeError("No valid hotspot candidate plan produced complete guide pairs.")

    round_summary_rows = [
        {
            "round": 1,
            "problematic_tiles": "",
            "problem_blocks": "",
            "candidate_plans": int(len(plans)),
            "accepted_candidate_plan_id": str(best["candidate_plan_id"]),
            "accepted_score": float(best["score"]),
            "stop_reason": "accepted_best_valid_hotspot_plan",
        }
    ]
    accepted_path_rows = [
        {
            "round": 1,
            "accepted_candidate_plan_id": str(best["candidate_plan_id"]),
            "strategy": best.get("strategy", ""),
            "strategy_details": best.get("strategy_details", ""),
            "accepted_score": float(best["score"]),
        }
    ]

    if not args.disable_local_repair:
        initial_best = best.copy()
        initial_best_pid = str(initial_best["candidate_plan_id"])
        initial_record = candidate_records[initial_best_pid]
        initial_selected = pd.read_csv(initial_record["selected_path"], sep="\t")
        weak_tile_ids = selected_weak_tile_ids(initial_selected, args.local_repair_score_threshold)
        repair_plans = generate_local_repair_plans(
            initial_record["tiles"],
            initial_selected,
            targets,
            existing_signatures,
            args,
        )
        repair_best: Optional[Dict] = None

        for idx, repair in enumerate(repair_plans, start=1):
            pid = f"repair_{idx:03d}"
            block = f"{repair['block_start']}-{repair['block_end']}"
            print(f"[HOTSPOT-REPAIR] {idx}/{len(repair_plans)} {pid} weak_tile={repair['weak_tile_id']} block={block}", flush=True)
            candidate_dir = outdir / "candidates" / pid
            candidate_dir.mkdir(parents=True, exist_ok=True)
            tiles = repair["tiles"]
            tile_path = candidate_dir / "tile_plan.tsv"
            tiles.to_csv(tile_path, sep="\t", index=False)
            plan_meta = {
                "candidate_plan_id": pid,
                "strategy": "local_repair",
                "strategy_details": (
                    f"from={initial_best_pid};weak_tile={repair['weak_tile_id']};"
                    f"block={block};target_ids={repair['target_ids']}"
                ),
            }
            try:
                selected_path = rerun_pipeline(tile_path, candidate_dir, args)
                metrics = evaluate_candidate(selected_path, tiles, targets, plan_meta, args)
                metrics["round"] = 2
                eval_rows.append(metrics)
                tile_history.append(
                    tiles.assign(
                        round=2,
                        candidate_plan_id=pid,
                        strategy=metrics["strategy"],
                        strategy_details=metrics["strategy_details"],
                        score=metrics["score"],
                    )
                )
                candidate_records[pid] = {
                    "candidate_dir": candidate_dir,
                    "selected_path": selected_path,
                    "tiles": tiles,
                    "metrics": metrics,
                }
                print(f"[HOTSPOT-REPAIR-SCORE] {json.dumps(metrics, sort_keys=True)}", flush=True)
                if int(metrics["is_valid"]) == 1 and float(metrics["score"]) > float(best["score"]) + float(args.local_repair_min_score_improvement):
                    best = metrics
                    repair_best = metrics
            except Exception as exc:
                metrics = {
                    "round": 2,
                    "candidate_plan_id": pid,
                    "strategy": "local_repair",
                    "strategy_details": plan_meta["strategy_details"],
                    "is_valid": 0,
                    "score": -1e12,
                    "failure_reason": str(exc),
                }
                eval_rows.append(metrics)
                print(f"[HOTSPOT-REPAIR-SKIP] {pid} {exc}", flush=True)

        stop_reason = "no_weak_tiles_for_local_repair"
        accepted_repair_id = ""
        accepted_repair_score = ""
        if repair_plans:
            stop_reason = "accepted_local_repair" if repair_best is not None else "no_local_repair_improved_score"
        elif weak_tile_ids:
            stop_reason = "no_valid_local_repair_candidates_generated"

        if repair_best is not None:
            accepted_repair_id = str(repair_best["candidate_plan_id"])
            accepted_repair_score = float(repair_best["score"])
            accepted_path_rows.append(
                {
                    "round": 2,
                    "accepted_candidate_plan_id": accepted_repair_id,
                    "strategy": repair_best.get("strategy", ""),
                    "strategy_details": repair_best.get("strategy_details", ""),
                    "accepted_score": float(repair_best["score"]),
                }
            )

        round_summary_rows.append(
            {
                "round": 2,
                "problematic_tiles": ",".join(str(x) for x in weak_tile_ids),
                "problem_blocks": "",
                "candidate_plans": int(len(repair_plans)),
                "accepted_candidate_plan_id": accepted_repair_id,
                "accepted_score": accepted_repair_score,
                "stop_reason": stop_reason,
            }
        )

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(search_dir / "candidate_evaluation_all_rounds.tsv", sep="\t", index=False)
    if tile_history:
        tile_history_df = pd.concat(tile_history, ignore_index=True, sort=False)
    else:
        tile_history_df = pd.DataFrame()
    tile_history_df.to_csv(search_dir / "tile_plan_candidates_all_rounds.tsv", sep="\t", index=False)
    write_bed(tile_history_df, search_dir / "tile_plan_candidates_all_rounds.bed", ["round", "candidate_plan_id", "strategy", "tile_id"])

    accepted_pid = str(best["candidate_plan_id"])
    accepted_record = candidate_records[accepted_pid]
    accepted_tiles = accepted_record["tiles"]
    accepted_selected = pd.read_csv(accepted_record["selected_path"], sep="\t")
    accepted_selected = merge_tile_context(accepted_selected, accepted_tiles)

    accepted_selected.to_csv(outdir / "selected_pairs_final.tsv", sep="\t", index=False)
    accepted_tiles.to_csv(outdir / "final_tile_plan.tsv", sep="\t", index=False)
    write_bed(accepted_tiles, outdir / "final_tile_plan.bed", ["tile_id"])
    shutil.copy2(outdir / "final_tile_plan.bed", search_dir / "final_tile_plan.bed")

    round_summary = pd.DataFrame(round_summary_rows)
    round_summary.to_csv(search_dir / "round_summary.tsv", sep="\t", index=False)
    pd.DataFrame(accepted_path_rows).to_csv(search_dir / "accepted_design_path.tsv", sep="\t", index=False)

    print(f"[DONE] Hotspot final selected pairs: {outdir / 'selected_pairs_final.tsv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
