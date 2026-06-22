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
    best: Optional[Dict] = None

    for idx, (_, plan) in enumerate(plans.iterrows(), start=1):
        pid = str(plan["candidate_plan_id"])
        print(f"[HOTSPOT-CANDIDATE] {idx}/{len(plans)} {pid}", flush=True)
        plan_df = candidates[candidates["candidate_plan_id"] == pid].copy()
        if plan_df.empty:
            continue
        tiles = standardize_tile_plan(plan_df)
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

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(search_dir / "candidate_evaluation_all_rounds.tsv", sep="\t", index=False)
    if tile_history:
        tile_history_df = pd.concat(tile_history, ignore_index=True, sort=False)
    else:
        tile_history_df = pd.DataFrame()
    tile_history_df.to_csv(search_dir / "tile_plan_candidates_all_rounds.tsv", sep="\t", index=False)
    write_bed(tile_history_df, search_dir / "tile_plan_candidates_all_rounds.bed", ["round", "candidate_plan_id", "strategy", "tile_id"])

    if best is None:
        raise RuntimeError("No valid hotspot candidate plan produced complete guide pairs.")

    accepted_pid = str(best["candidate_plan_id"])
    accepted_dir = outdir / "candidates" / accepted_pid
    accepted_tiles = standardize_tile_plan(candidates[candidates["candidate_plan_id"] == accepted_pid].copy())
    accepted_selected = pd.read_csv(accepted_dir / "08_global" / "guide_pairs_global_selected.tsv", sep="\t")
    accepted_selected = merge_tile_context(accepted_selected, accepted_tiles)

    accepted_selected.to_csv(outdir / "selected_pairs_final.tsv", sep="\t", index=False)
    accepted_tiles.to_csv(outdir / "final_tile_plan.tsv", sep="\t", index=False)
    write_bed(accepted_tiles, outdir / "final_tile_plan.bed", ["tile_id"])
    shutil.copy2(outdir / "final_tile_plan.bed", search_dir / "final_tile_plan.bed")

    round_summary = pd.DataFrame(
        [
            {
                "round": 1,
                "problematic_tiles": "",
                "problem_blocks": "",
                "candidate_plans": int(len(plans)),
                "accepted_candidate_plan_id": accepted_pid,
                "accepted_score": float(best["score"]),
                "stop_reason": "accepted_best_valid_hotspot_plan",
            }
        ]
    )
    round_summary.to_csv(search_dir / "round_summary.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "round": 1,
                "accepted_candidate_plan_id": accepted_pid,
                "strategy": best.get("strategy", ""),
                "strategy_details": best.get("strategy_details", ""),
                "accepted_score": float(best["score"]),
            }
        ]
    ).to_csv(search_dir / "accepted_design_path.tsv", sep="\t", index=False)

    print(f"[DONE] Hotspot final selected pairs: {outdir / 'selected_pairs_final.tsv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
