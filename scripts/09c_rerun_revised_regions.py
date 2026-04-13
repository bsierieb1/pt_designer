#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Iterative redesign controller for revised regions or candidate tile plans.")
    p.add_argument("--tiles-revised-tsv", help="Legacy single-pass revised tiles TSV")
    p.add_argument("--selected-pairs-tsv", help="Step 8 selected pairs TSV for iterative mode")
    p.add_argument("--tile-plan-tsv", help="Current full tile plan TSV for iterative mode")
    p.add_argument("--locus-json", required=True)
    p.add_argument("--fasta")
    p.add_argument("--flashfry-jar")
    p.add_argument("--flashfry-database")
    p.add_argument("--scripts-dir", required=True)
    p.add_argument("--base-output-dir", required=True)
    p.add_argument("--common-snps-bed")
    p.add_argument("--repeats-bed")
    p.add_argument("--segdups-bed")
    p.add_argument("--max-mismatches", type=int, default=3)
    p.add_argument("--tracr-rna", default="Hsu2013")
    p.add_argument("--max-redesign-rounds", type=int, default=5)
    p.add_argument("--boundary-shift-jump-bp", type=int, default=1000)
    p.add_argument("--max-boundary-jumps", type=int, default=2)
    p.add_argument("--target-tile-size-bp", type=int, default=10000)
    p.add_argument("--max-tile-size-bp", type=int, default=12000)
    p.add_argument("--target-overlap-size-bp", type=int, default=1200)
    p.add_argument("--min-overlap-bp", type=int, default=800)
    p.add_argument("--max-candidate-plans", type=int, default=24)
    p.add_argument("--early-stop-min-pair-score", type=float, default=120.0)
    p.add_argument("--show-substep-output", action="store_true")
    p.add_argument("--tile-overlaps-tsv", help="Optional overlap TSV passed to 09a")
    return p.parse_args()


def run(cmd: List[str], quiet: bool = False) -> None:
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


def choose_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = False):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of the required columns found: {list(candidates)}")
    return None


def standardize_tile_plan(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def coalesce_numeric(cols: List[str]) -> pd.Series:
        present = [c for c in cols if c in out.columns]
        if not present:
            return pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
        s = pd.to_numeric(out[present[0]], errors="coerce")
        for c in present[1:]:
            s = s.fillna(pd.to_numeric(out[c], errors="coerce"))
        return s

    def coalesce_text(cols: List[str], default: str = "") -> pd.Series:
        present = [c for c in cols if c in out.columns]
        if not present:
            return pd.Series([default] * len(out), index=out.index, dtype="object")
        s = out[present[0]].astype("object")
        for c in present[1:]:
            s = s.where(~s.isna() & (s.astype(str) != "nan"), out[c].astype("object"))
        return s.fillna(default).astype(str)

    if "tile_id" not in out.columns:
        out["tile_id"] = pd.NA

    norm = pd.DataFrame({
        "tile_id": pd.to_numeric(out["tile_id"], errors="coerce"),
        "chrom": coalesce_text(["chrom", "chr", "chromosome"], default=""),
        "start_1based": coalesce_numeric(["tile_start_1based", "start_1based", "revised_start_1based", "new_start_1based"]),
        "end_1based": coalesce_numeric(["tile_end_1based", "end_1based", "revised_end_1based", "new_end_1based"]),
    })

    bad_coords = norm["start_1based"].isna() | norm["end_1based"].isna()
    if bad_coords.any():
        bad_rows = norm.loc[bad_coords]
        raise ValueError("tile plan contains missing coordinates after normalization:\n" + bad_rows.to_string(index=False))

    norm["start_1based"] = norm["start_1based"].astype(int)
    norm["end_1based"] = norm["end_1based"].astype(int)
    norm = norm.sort_values(["start_1based", "end_1based"]).reset_index(drop=True)

    need_reassign = norm["tile_id"].isna().any() or norm["tile_id"].duplicated().any()
    if need_reassign:
        norm["tile_id"] = range(1, len(norm) + 1)
    else:
        norm["tile_id"] = norm["tile_id"].astype(int)
        norm = norm.sort_values(["tile_id", "start_1based", "end_1based"]).drop_duplicates(subset=["tile_id"], keep="last")
        norm = norm.sort_values(["tile_id", "start_1based", "end_1based"]).reset_index(drop=True)

    return norm.reset_index(drop=True)


def normalize_revised_tiles(tiles_revised_tsv: Path, outdir: Path) -> Path:
    df = pd.read_csv(tiles_revised_tsv, sep="\t")
    df = standardize_tile_plan(df)
    out = outdir / "tiles_revised.normalized_for_step03.tsv"
    df.to_csv(out, sep="\t", index=False)
    return out


def rerun_pipeline(tiles_tsv: Path, outdir: Path, args: argparse.Namespace, quiet: bool = True) -> Path:
    scripts = Path(args.scripts_dir)
    py = sys.executable
    outdir.mkdir(parents=True, exist_ok=True)

    step03 = outdir / "03_boundary_windows"
    step04 = outdir / "04_fetch_guides"
    step05 = outdir / "05_oriented"
    step06a = outdir / "06a_annotated"
    step06d = outdir / "06d_ranked"
    step07 = outdir / "07_paired"
    step08 = outdir / "08_global"

    run([py, str(scripts / "03_make_boundary_windows.py"), "--locus-json", args.locus_json, "--tiles-tsv", str(tiles_tsv), "--outdir", str(step03)], quiet)
    run([py, str(scripts / "04_fetch_guides.py"), "--boundary-windows-tsv", str(step03 / "boundary_windows.tsv"), "--outdir", str(step04)], quiet)
    run([py, str(scripts / "05_filter_orientation.py"), "--candidates-raw-tsv", str(step04 / "candidates_raw.tsv"), "--outdir", str(step05)], quiet)

    cmd06a = [py, str(scripts / "06a_annotate_guides.py"), "--candidates-oriented-tsv", str(step05 / "candidates_oriented.tsv"), "--outdir", str(step06a)]
    if args.common_snps_bed:
        cmd06a += ["--common-snps-bed", args.common_snps_bed]
    run(cmd06a, quiet)

    run([py, str(scripts / "06d_rank_guides.py"), "--candidates-annotated-tsv", str(step06a / "candidates_annotated.tsv"), "--outdir", str(step06d)], quiet)
    run([py, str(scripts / "07_pair_guides.py"), "--drop-unacceptable-pairs", "--candidates-scored-tsv", str(step06d / "candidates_scored.tsv"), "--outdir", str(step07)], quiet)
    run([py, str(scripts / "08_optimize_tiles.py"), "--guide-pairs-tsv", str(step07 / "guide_pairs_top.tsv"), "--outdir", str(step08)], quiet)

    selected = step08 / "guide_pairs_global_selected.tsv"
    if not selected.exists():
        raise FileNotFoundError(f"Missing selected pairs output: {selected}")
    return selected


def merge_selected_pairs(current_selected_path: Path, new_selected_path: Path, redesigned_tile_ids: List[int], outpath: Path) -> Path:
    cur = pd.read_csv(current_selected_path, sep="\t") if current_selected_path.exists() else pd.DataFrame()
    new = pd.read_csv(new_selected_path, sep="\t") if new_selected_path.exists() else pd.DataFrame()
    redesigned = {int(x) for x in redesigned_tile_ids}
    if not cur.empty and "tile_id" in cur.columns:
        cur = cur[~pd.to_numeric(cur["tile_id"], errors="coerce").isin(redesigned)]
    merged = pd.concat([cur, new], ignore_index=True, sort=False)
    if "tile_id" in merged.columns:
        merged["tile_id"] = pd.to_numeric(merged["tile_id"], errors="coerce").astype("Int64")
        merged = merged.sort_values(["tile_id"]).drop_duplicates(subset=["tile_id"], keep="last")
    merged.to_csv(outpath, sep="\t", index=False)
    return outpath


def merge_tile_plans(current_tiles: pd.DataFrame, candidate_tiles: pd.DataFrame) -> pd.DataFrame:
    cur = standardize_tile_plan(current_tiles)
    cand = standardize_tile_plan(candidate_tiles)
    redesigned = set(cand["tile_id"].tolist())
    merged = pd.concat([cur[~cur["tile_id"].isin(redesigned)], cand], ignore_index=True, sort=False)
    merged = merged.sort_values(["tile_id"]).drop_duplicates(subset=["tile_id"], keep="last").reset_index(drop=True)
    return merged


def adjacent_overlap_metrics(tiles: pd.DataFrame):
    tiles = standardize_tile_plan(tiles)
    bad = []
    min_overlap = None
    for i in range(len(tiles) - 1):
        left = tiles.iloc[i]
        right = tiles.iloc[i + 1]
        overlap = int(left["end_1based"] - right["start_1based"] + 1)
        if min_overlap is None or overlap < min_overlap:
            min_overlap = overlap
        bad.append({
            "left_tile_id": int(left["tile_id"]),
            "right_tile_id": int(right["tile_id"]),
            "overlap_bp": overlap,
        })
    return (min_overlap if min_overlap is not None else 0), bad


def same_pool_overlap_metrics(tiles: pd.DataFrame):
    tiles = standardize_tile_plan(tiles)
    hits = []
    max_overlap = 0
    rows = tiles.to_dict('records')
    for i in range(len(rows)):
        for j in range(i + 2, len(rows), 2):
            left = rows[i]
            right = rows[j]
            overlap = int(min(left['end_1based'], right['end_1based']) - max(left['start_1based'], right['start_1based']) + 1)
            if overlap > 0:
                hits.append({
                    'left_tile_id': int(left['tile_id']),
                    'right_tile_id': int(right['tile_id']),
                    'overlap_bp': overlap,
                })
                if overlap > max_overlap:
                    max_overlap = overlap
    return int(max_overlap), hits


def evaluate_candidate(selected_pairs_path: Path, merged_tiles: pd.DataFrame, args: argparse.Namespace) -> dict:
    tiles = standardize_tile_plan(merged_tiles)
    selected = pd.read_csv(selected_pairs_path, sep="\t") if Path(selected_pairs_path).exists() else pd.DataFrame()
    if "tile_id" in selected.columns:
        selected["tile_id"] = pd.to_numeric(selected["tile_id"], errors="coerce").astype("Int64")

    tile_ids = set(tiles["tile_id"].tolist())
    selected_tile_ids = set(selected["tile_id"].dropna().astype(int).tolist()) if "tile_id" in selected.columns else set()
    n_missing = len(tile_ids - selected_tile_ids)

    pair_score_col = choose_col(selected, ["pair_score", "optimization_score", "score"])
    min_off_col = choose_col(selected, ["pair_min_offtarget_score", "pair_mean_offtarget_score", "left_offtarget_score", "offtarget_score"])
    unacceptable_mask = pd.Series(False, index=selected.index)
    for col in ["pair_design_class", "design_class"]:
        if col in selected.columns:
            unacceptable_mask = unacceptable_mask | selected[col].astype(str).str.lower().eq("unacceptable")
    for col in ["left_has_exact_in_locus_offtarget", "right_has_exact_in_locus_offtarget", "has_exact_in_locus_offtarget"]:
        if col in selected.columns:
            unacceptable_mask = unacceptable_mask | selected[col].fillna(False).astype(bool)
    n_unacceptable = int(unacceptable_mask.sum())

    total_pair_score = float(pd.to_numeric(selected[pair_score_col], errors="coerce").fillna(0).sum()) if pair_score_col else 0.0
    min_pair_score = float(pd.to_numeric(selected[pair_score_col], errors="coerce").min()) if pair_score_col and not selected.empty else 0.0
    min_offtarget_score = float(pd.to_numeric(selected[min_off_col], errors="coerce").min()) if min_off_col and not selected.empty else 0.0

    min_overlap, overlap_rows = adjacent_overlap_metrics(tiles)
    bad_adjacent = [r for r in overlap_rows if r["overlap_bp"] < args.min_overlap_bp]
    max_same_pool_overlap, same_pool_rows = same_pool_overlap_metrics(tiles)
    size_violations = int(((tiles["end_1based"] - tiles["start_1based"] + 1) > args.max_tile_size_bp).sum())
    n_same_pool_overlaps = int(len(same_pool_rows))
    is_valid = int(n_same_pool_overlaps == 0)

    score = total_pair_score
    score -= 1500.0 * n_missing
    score -= 1200.0 * n_unacceptable
    score -= 800.0 * len(bad_adjacent)
    score -= 500.0 * size_violations
    score -= 5000.0 * n_same_pool_overlaps
    score += 0.5 * min_pair_score
    score += 0.25 * min_overlap

    return {
        "n_tiles": int(len(tiles)),
        "n_selected": int(len(selected)),
        "n_missing": int(n_missing),
        "n_unacceptable": int(n_unacceptable),
        "total_pair_score": float(total_pair_score),
        "min_pair_score": float(min_pair_score),
        "min_offtarget_score": float(min_offtarget_score),
        "min_adjacent_overlap_bp": int(min_overlap),
        "n_bad_adjacent_overlaps": int(len(bad_adjacent)),
        "bad_adjacent_overlaps": json.dumps(bad_adjacent),
        "n_same_pool_overlaps": int(n_same_pool_overlaps),
        "max_same_pool_overlap_bp": int(max_same_pool_overlap),
        "same_pool_overlaps": json.dumps(same_pool_rows),
        "is_valid": int(is_valid),
        "size_violations": int(size_violations),
        "score": float(score),
    }


def grouped_problem_blocks(problems: pd.DataFrame) -> pd.DataFrame:
    if problems.empty:
        return pd.DataFrame(columns=["redesign_block_id", "tile_ids", "n_tiles"])
    probs = problems.copy()
    probs["tile_id"] = pd.to_numeric(probs["tile_id"], errors="coerce")
    probs = probs[probs["is_problematic"].fillna(False)].sort_values("tile_id")
    if probs.empty:
        return pd.DataFrame(columns=["redesign_block_id", "tile_ids", "n_tiles"])
    rows = []
    current = []
    prev = None
    block_idx = 0
    for tid in probs["tile_id"].dropna().astype(int).tolist():
        if prev is None or tid == prev + 1:
            current.append(tid)
        else:
            block_idx += 1
            rows.append({"redesign_block_id": f"block_{block_idx}", "tile_ids": ",".join(map(str, current)), "n_tiles": len(current)})
            current = [tid]
        prev = tid
    if current:
        block_idx += 1
        rows.append({"redesign_block_id": f"block_{block_idx}", "tile_ids": ",".join(map(str, current)), "n_tiles": len(current)})
    return pd.DataFrame(rows)


def write_bed(df: pd.DataFrame, outpath: Path, name_cols: list[str], extra_cols: list[str] | None = None) -> None:
    if df is None or df.empty:
        return
    work = df.copy()
    start_col = choose_col(work, ["start_1based", "tile_start_1based"], required=True)
    end_col = choose_col(work, ["end_1based", "tile_end_1based"], required=True)
    chrom_col = choose_col(work, ["chrom", "chr", "chromosome"], required=True)
    cols = [c for c in name_cols if c in work.columns]
    if not cols:
        work["_name"] = "tile"
    else:
        work["_name"] = work[cols].astype(str).agg("|".join, axis=1)
    work["_start0"] = pd.to_numeric(work[start_col], errors="coerce") - 1
    work["_end"] = pd.to_numeric(work[end_col], errors="coerce")
    work = work.dropna(subset=[chrom_col, "_start0", "_end"]).copy()
    if work.empty:
        return
    work["_start0"] = work["_start0"].astype(int).clip(lower=0)
    work["_end"] = work["_end"].astype(int)
    bed = pd.DataFrame({0: work[chrom_col].astype(str), 1: work["_start0"], 2: work["_end"], 3: work["_name"]})
    extras = extra_cols or []
    for c in extras:
        if c in work.columns:
            bed[len(bed.columns)] = work[c].astype(str)
    bed.to_csv(outpath, sep="	", index=False, header=False)


def write_search_html(search_dir: Path, locus_label: str = "") -> Path:
    import html

    eval_path = search_dir / "candidate_evaluation_all_rounds.tsv"
    rounds_path = search_dir / "round_summary.tsv"
    accepted_path = search_dir / "accepted_design_path.tsv"
    tiles_path = search_dir / "tile_plan_candidates_all_rounds.tsv"

    eval_df = pd.read_csv(eval_path, sep="\t") if eval_path.exists() else pd.DataFrame()
    rounds_df = pd.read_csv(rounds_path, sep="\t") if rounds_path.exists() else pd.DataFrame()
    accepted_df = pd.read_csv(accepted_path, sep="\t") if accepted_path.exists() else pd.DataFrame()
    tiles_df = pd.read_csv(tiles_path, sep="\t") if tiles_path.exists() else pd.DataFrame()

    def _norm_tile_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        x = df.copy()

        if "start_1based" not in x.columns and "tile_start_1based" in x.columns:
            x["start_1based"] = x["tile_start_1based"]
        if "end_1based" not in x.columns and "tile_end_1based" in x.columns:
            x["end_1based"] = x["tile_end_1based"]

        required = ["round", "candidate_plan_id", "start_1based", "end_1based"]
        missing = [c for c in required if c not in x.columns]
        if missing:
            return pd.DataFrame()

        x = x.dropna(subset=["round", "candidate_plan_id", "start_1based", "end_1based"]).copy()
        if x.empty:
            return pd.DataFrame()

        x["round"] = pd.to_numeric(x["round"], errors="coerce")
        x["start_1based"] = pd.to_numeric(x["start_1based"], errors="coerce")
        x["end_1based"] = pd.to_numeric(x["end_1based"], errors="coerce")
        x = x.dropna(subset=["round", "start_1based", "end_1based"]).copy()
        if x.empty:
            return pd.DataFrame()

        x["round"] = x["round"].astype(int)
        x["start_1based"] = x["start_1based"].astype(int)
        x["end_1based"] = x["end_1based"].astype(int)
        return x

    def _candidate_svg(sub: pd.DataFrame, width: int = 900) -> str:
        sub = sub.sort_values(["start_1based", "end_1based"]).reset_index(drop=True).copy()
        if sub.empty:
            return "<p><em>No tile geometry available.</em></p>"

        locus_start = int(sub["start_1based"].min())
        locus_end = int(sub["end_1based"].max())
        span = max(1, locus_end - locus_start + 1)

        left_pad = 70
        right_pad = 20
        track_w = width - left_pad - right_pad
        row_h = 28
        rect_h = 16
        top_y = 18
        bot_y = top_y + row_h
        height = bot_y + rect_h + 18

        def xpos(pos: int) -> float:
            return left_pad + ((pos - locus_start) / span) * track_w

        parts = []
        parts.append(
            f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
            f"xmlns='http://www.w3.org/2000/svg'>"
        )

        parts.append(f"<text x='6' y='{top_y + 12}' font-size='12' fill='#444'>odd</text>")
        parts.append(f"<text x='6' y='{bot_y + 12}' font-size='12' fill='#444'>even</text>")

        x0 = xpos(locus_start)
        x1 = xpos(locus_end)
        parts.append(f"<line x1='{x0:.1f}' y1='{top_y + rect_h/2:.1f}' x2='{x1:.1f}' y2='{top_y + rect_h/2:.1f}' stroke='#bbb' stroke-width='1'/>")
        parts.append(f"<line x1='{x0:.1f}' y1='{bot_y + rect_h/2:.1f}' x2='{x1:.1f}' y2='{bot_y + rect_h/2:.1f}' stroke='#bbb' stroke-width='1'/>")

        parts.append(f"<text x='{x0:.1f}' y='12' font-size='11' fill='#666'>{locus_start}</text>")
        parts.append(f"<text x='{max(x0, x1 - 70):.1f}' y='12' font-size='11' fill='#666'>{locus_end}</text>")

        for i, row in sub.iterrows():
            label = f"T{i + 1}"
            y = top_y if ((i + 1) % 2 == 1) else bot_y
            fill = "#4C78A8" if ((i + 1) % 2 == 1) else "#F58518"

            rx = xpos(int(row["start_1based"]))
            rw = max(2.0, xpos(int(row["end_1based"])) - rx)

            parts.append(
                f"<rect x='{rx:.1f}' y='{y:.1f}' width='{rw:.1f}' height='{rect_h}' "
                f"rx='3' ry='3' fill='{fill}' fill-opacity='0.85' stroke='#333' stroke-width='0.8'/>"
            )
            parts.append(
                f"<text x='{rx + 4:.1f}' y='{y + 12:.1f}' font-size='11' fill='white'>{html.escape(label)}</text>"
            )

        parts.append("</svg>")
        return "".join(parts)

    tiles_df = _norm_tile_df(tiles_df)

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Design search history</title>",
        (
            "<style>"
            "body{font-family:Arial,sans-serif;margin:24px;color:#222}"
            "table{border-collapse:collapse;width:100%;margin:12px 0;font-size:12px}"
            "th,td{border:1px solid #ccc;padding:6px 8px;text-align:left}"
            "th{background:#f3f3f3}"
            ".muted{color:#666;font-size:12px}"
            ".candidate-block{border:1px solid #ddd;padding:10px 12px;margin:14px 0;border-radius:8px}"
            ".candidate-title{font-weight:bold;margin-bottom:6px}"
            ".svg-wrap{overflow-x:auto;background:#fafafa;border:1px solid #eee;padding:8px;border-radius:6px}"
            "</style>"
        ),
        "</head><body>",
        f"<h1>Design search history</h1><p><strong>Locus:</strong> {html.escape(locus_label or 'NA')}</p>",
    ]

    html_parts.append("<h2>Accepted path by round</h2>")
    html_parts.append(accepted_df.to_html(index=False) if not accepted_df.empty else "<p><em>No accepted-path table available.</em></p>")

    html_parts.append("<h2>Round summary</h2>")
    html_parts.append(rounds_df.to_html(index=False) if not rounds_df.empty else "<p><em>No round summary available.</em></p>")

    html_parts.append("<h2>Candidate tile geometry</h2>")
    if tiles_df.empty:
        html_parts.append("<p><em>No tile geometry table available.</em></p>")
    else:
        meta_cols = ["round", "candidate_plan_id"]
        meta = pd.DataFrame()
        if not eval_df.empty and all(c in eval_df.columns for c in meta_cols):
            keep_cols = [c for c in ["round", "candidate_plan_id", "strategy", "strategy_details", "score"] if c in eval_df.columns]
            meta = eval_df[keep_cols].drop_duplicates()

        for (rnd, cand), sub in tiles_df.groupby(["round", "candidate_plan_id"], sort=True):
            title = f"r{rnd} {cand}"
            subtitle = ""
            if not meta.empty:
                hit = meta[(meta["round"] == rnd) & (meta["candidate_plan_id"] == cand)]
                if not hit.empty:
                    row = hit.iloc[0]
                    strategy = row["strategy"] if "strategy" in hit.columns else ""
                    details = row["strategy_details"] if "strategy_details" in hit.columns else ""
                    score = row["score"] if "score" in hit.columns else ""
                    subtitle = f"{strategy} | {details} | score={score}"

            html_parts.append("<div class='candidate-block'>")
            html_parts.append(f"<div class='candidate-title'>{html.escape(title)}</div>")
            if subtitle:
                html_parts.append(f"<div class='muted'>{html.escape(str(subtitle))}</div>")
            html_parts.append(f"<div class='svg-wrap'>{_candidate_svg(sub)}</div>")
            html_parts.append("</div>")

    html_parts.append("<h2>All candidate evaluations</h2>")
    html_parts.append(eval_df.to_html(index=False) if not eval_df.empty else "<p><em>No candidate evaluations available.</em></p>")

    html_parts.append("<p class='muted'>This report is intended to help inspect what the redesign controller tried, which options were accepted, and which geometries were rejected.</p>")
    html_parts.append("</body></html>")

    out = search_dir / "design_search_history.html"
    out.write_text("".join(html_parts), encoding="utf-8")
    return out


def iterative_redesign(args: argparse.Namespace) -> int:
    scripts = Path(args.scripts_dir)
    base = Path(args.base_output_dir)
    outdir = base / "09c_redesign_iterative"
    outdir.mkdir(parents=True, exist_ok=True)
    search_dir = outdir / "search_history"
    search_dir.mkdir(parents=True, exist_ok=True)

    current_tiles = standardize_tile_plan(pd.read_csv(args.tile_plan_tsv, sep="\t"))
    current_selected_path = Path(args.selected_pairs_tsv)

    all_eval_rows = []
    all_candidate_tile_rows = []
    round_rows = []
    accepted_rows = []

    locus_label = ""
    try:
        locus = json.loads(Path(args.locus_json).read_text())
        locus_label = locus.get("locus", locus.get("region", ""))
    except Exception:
        pass

    py = sys.executable

    for rnd in range(1, args.max_redesign_rounds + 1):
        round_dir = outdir / f"round_{rnd:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        current_tiles_path = round_dir / "current_tile_plan.tsv"
        current_tiles.to_csv(current_tiles_path, sep="\t", index=False)
        print(f"[ROUND] {rnd}/{args.max_redesign_rounds}", flush=True)

        detect_dir = round_dir / "09a_detect"
        cmd09a = [py, str(scripts / "09a_detect_problematic_tiles.py"), "--selected-pairs-tsv", str(current_selected_path), "--tile-plan-tsv", str(current_tiles_path), "--outdir", str(detect_dir), "--max-fragment-length-bp", str(args.max_tile_size_bp)]
        if args.tile_overlaps_tsv:
            cmd09a += ["--tile-overlaps-tsv", args.tile_overlaps_tsv]
        run(cmd09a, quiet=not args.show_substep_output)

        problems_path = detect_dir / "tiles_problematic.tsv"
        problems = pd.read_csv(problems_path, sep="\t") if problems_path.exists() else pd.DataFrame()
        blocks = grouped_problem_blocks(problems)
        n_problematic = int(problems["is_problematic"].fillna(False).sum()) if not problems.empty else 0
        print(f"[ROUND-INFO] problematic_tiles={n_problematic} problem_blocks={len(blocks)}", flush=True)
        if n_problematic == 0 or blocks.empty:
            round_rows.append({"round": rnd, "problematic_tiles": n_problematic, "problem_blocks": int(len(blocks)), "candidate_plans": 0, "accepted_candidate_plan_id": "", "accepted_score": "", "stop_reason": "no_problematic_tiles"})
            break

        gen_dir = round_dir / "09b_generate"
        cmd09b = [py, str(scripts / "09b_retile_difficult_intervals.py"), "--tiles-problematic-tsv", str(problems_path), "--tile-plan-tsv", str(current_tiles_path), "--outdir", str(gen_dir)]
        run(cmd09b, quiet=not args.show_substep_output)

        plans_path = gen_dir / "candidate_plans_summary.tsv"
        cand_tiles_path = gen_dir / "tile_plan_candidates.tsv"
        if not plans_path.exists() or not cand_tiles_path.exists():
            raise FileNotFoundError("09b did not produce candidate_plans_summary.tsv and tile_plan_candidates.tsv")
        plans = pd.read_csv(plans_path, sep="\t")
        cand_tiles = pd.read_csv(cand_tiles_path, sep="\t")
        if len(plans) > args.max_candidate_plans:
            plans = plans.head(args.max_candidate_plans).copy()
        print(f"[ROUND-INFO] evaluating {len(plans)} candidate plans", flush=True)

        baseline_score = None
        round_results = []
        best = None
        for idx, (_, plan) in enumerate(plans.iterrows(), start=1):
            pid = str(plan["candidate_plan_id"])
            strategy = str(plan.get("strategy", "unknown"))
            details = str(plan.get("strategy_details", ""))
            print(f"[CANDIDATE] round={rnd} index={idx}/{len(plans)} id={pid} strategy={strategy} details={details}", flush=True)
            plan_df = cand_tiles[cand_tiles["candidate_plan_id"] == pid].copy() if "candidate_plan_id" in cand_tiles.columns else cand_tiles.copy()
            if plan_df.empty:
                print(f"[CANDIDATE-SKIP] id={pid} strategy={strategy} reason=empty candidate tile set", flush=True)
                continue
            plan_save = plan_df.copy()
            all_candidate_tile_rows.append(plan_save.assign(round=rnd, candidate_plan_id=pid, strategy=strategy, strategy_details=details))
            plan_path = round_dir / f"{pid}.tiles.tsv"
            standardize_tile_plan(plan_df).to_csv(plan_path, sep="\t", index=False)
            rerun_dir = round_dir / "candidates" / pid
            try:
                selected_path = rerun_pipeline(plan_path, rerun_dir, args, quiet=not args.show_substep_output)
                plan_std = standardize_tile_plan(plan_df)
                plan_std.to_csv(rerun_dir / "candidate_tile_plan.standardized.tsv", sep="\t", index=False)
                metrics = evaluate_candidate(selected_path, plan_std, args)
                metrics.update({"round": rnd, "candidate_plan_id": pid, "strategy": strategy, "strategy_details": details})
                round_results.append(metrics)
                all_eval_rows.append(metrics)
                print(f"[CANDIDATE-SCORE] {json.dumps(metrics, sort_keys=True)}", flush=True)
                if baseline_score is None and strategy == "baseline_keep_current":
                    baseline_score = float(metrics["score"])
                if int(metrics.get("is_valid", 1)) != 1:
                    print(f"[CANDIDATE-INVALID] round={rnd} candidate={pid} same_pool_overlaps={metrics.get('same_pool_overlaps', '[]')}", flush=True)
                elif best is None or float(metrics["score"]) > float(best["score"]):
                    best = metrics
                    print(f"[CANDIDATE-BEST-SO-FAR] {json.dumps(best, sort_keys=True)}", flush=True)
                if int(metrics.get("is_valid", 1)) == 1 and int(metrics["n_missing"]) == 0 and int(metrics["n_unacceptable"]) == 0 and int(metrics["n_bad_adjacent_overlaps"]) == 0 and float(metrics["min_pair_score"]) >= float(args.early_stop_min_pair_score):
                    print(f"[CANDIDATE-EARLY-STOP] round={rnd} candidate={pid} min_pair_score={metrics['min_pair_score']}", flush=True)
                    break
            except Exception as e:
                print(f"[CANDIDATE-SKIP] id={pid} strategy={strategy} reason={e}", flush=True)
                continue

        pd.DataFrame(round_results).to_csv(round_dir / "candidate_evaluation.tsv", sep="\t", index=False)
        if all_eval_rows:
            pd.DataFrame(all_eval_rows).to_csv(search_dir / "candidate_evaluation_all_rounds.tsv", sep="\t", index=False)
        if all_candidate_tile_rows:
            pd.concat(all_candidate_tile_rows, ignore_index=True, sort=False).to_csv(search_dir / "tile_plan_candidates_all_rounds.tsv", sep="\t", index=False)
        all_candidate_tiles_df = pd.concat(all_candidate_tile_rows, ignore_index=True, sort=False)
        all_candidate_tiles_df.to_csv(search_dir / "tile_plan_candidates_all_rounds.tsv", sep="\t", index=False)

        write_bed(
            all_candidate_tiles_df,
            search_dir / "tile_plan_candidates_all_rounds.bed",
            name_cols=["round", "candidate_plan_id", "strategy", "tile_id"],
            extra_cols=["strategy_details"],
        )

        if best is None:
            round_rows.append({"round": rnd, "problematic_tiles": n_problematic, "problem_blocks": int(len(blocks)), "candidate_plans": len(plans), "accepted_candidate_plan_id": "", "accepted_score": "", "stop_reason": "no_valid_candidates"})
            break

        print(f"[ROUND-BEST] {json.dumps(best, sort_keys=True)}", flush=True)
        best_score = float(best["score"])
        if baseline_score is None:
            baseline_score = best_score
        improve = best_score > baseline_score + 1e-9 and str(best["strategy"]) != "baseline_keep_current"
        accepted_pid = ""
        if improve:
            accepted_pid = str(best["candidate_plan_id"])
            accepted_mask = (cand_tiles["candidate_plan_id"] == accepted_pid)
            if "round" in cand_tiles.columns:
                accepted_mask = accepted_mask & (cand_tiles["round"] == rnd)
            accepted_plan_df = cand_tiles.loc[accepted_mask].copy()
            accepted_plan_std = standardize_tile_plan(accepted_plan_df)
            current_tiles = accepted_plan_std.copy()
            accepted_selected_path = round_dir / "candidates" / accepted_pid / "08_global" / "guide_pairs_global_selected.tsv"
            current_selected_path = accepted_selected_path
            current_tiles.to_csv(round_dir / "accepted_current_tile_plan.tsv", sep="\t", index=False)
            accepted_plan_std.to_csv(round_dir / "accepted_candidate_tiles.tsv", sep="\t", index=False)

            write_bed(
                accepted_plan_std,
                round_dir / "accepted_candidate_tiles.bed",
                name_cols=["tile_id"],
            )

            write_bed(
                current_tiles,
                round_dir / "accepted_current_tile_plan.bed",
                name_cols=["tile_id"],
            )
            accepted_rows.append({
                "round": rnd,
                "accepted_candidate_plan_id": accepted_pid,
                "strategy": best["strategy"],
                "strategy_details": best.get("strategy_details", ""),
                "baseline_score": baseline_score,
                "accepted_score": best_score,
                "score_delta": best_score - baseline_score,
            })
            print(f"[ROUND-ACCEPTED] round={rnd} candidate={accepted_pid} strategy={best['strategy']} score={best_score}", flush=True)
            round_rows.append({"round": rnd, "problematic_tiles": n_problematic, "problem_blocks": int(len(blocks)), "candidate_plans": len(plans), "accepted_candidate_plan_id": accepted_pid, "accepted_score": best_score, "stop_reason": "accepted_improvement"})
        else:
            round_rows.append({"round": rnd, "problematic_tiles": n_problematic, "problem_blocks": int(len(blocks)), "candidate_plans": len(plans), "accepted_candidate_plan_id": "", "accepted_score": baseline_score, "stop_reason": "no_improvement_over_baseline"})
            print(f"[DONE] stopping after round {rnd}: no candidate improved over baseline", flush=True)
            break

        pd.DataFrame(round_rows).to_csv(search_dir / "round_summary.tsv", sep="\t", index=False)
        pd.DataFrame(accepted_rows).to_csv(search_dir / "accepted_design_path.tsv", sep="\t", index=False)

    current_tiles.to_csv(outdir / "final_tile_plan.tsv", sep="\t", index=False)
    write_bed(
        current_tiles,
        outdir / "final_tile_plan.bed",
        name_cols=["tile_id"],
    )

    write_bed(
        current_tiles,
        search_dir / "final_tile_plan.bed",
        name_cols=["tile_id"],
    )
    if current_selected_path.exists():
        shutil.copy2(current_selected_path, outdir / "selected_pairs_final.tsv")
    if round_rows:
        pd.DataFrame(round_rows).to_csv(search_dir / "round_summary.tsv", sep="\t", index=False)
    if accepted_rows:
        pd.DataFrame(accepted_rows).to_csv(search_dir / "accepted_design_path.tsv", sep="\t", index=False)
    write_search_html(search_dir, locus_label=locus_label)
    print(f"[DONE] final tile plan: {outdir / 'final_tile_plan.tsv'}", flush=True)
    print(f"[DONE] design search history: {search_dir}", flush=True)
    return 0


def legacy_mode(args: argparse.Namespace) -> int:
    base_output_dir = Path(args.base_output_dir)
    outdir = base_output_dir / "09_redesign_rerun"
    outdir.mkdir(parents=True, exist_ok=True)
    normalized_tiles_tsv = normalize_revised_tiles(Path(args.tiles_revised_tsv), outdir)
    selected = rerun_pipeline(normalized_tiles_tsv, outdir, args, quiet=not args.show_substep_output)
    shutil.copy2(selected, outdir / "selected_pairs_final.tsv")
    print(f"[DONE] Final selected pairs: {outdir / 'selected_pairs_final.tsv'}")
    return 0


def main() -> int:
    args = parse_args()
    if args.tiles_revised_tsv:
        return legacy_mode(args)
    if not args.selected_pairs_tsv or not args.tile_plan_tsv:
        raise SystemExit("Provide either --tiles-revised-tsv for legacy mode, or both --selected-pairs-tsv and --tile-plan-tsv for iterative mode.")
    return iterative_redesign(args)


if __name__ == "__main__":
    raise SystemExit(main())
