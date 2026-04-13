from __future__ import annotations

import io
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


@dataclass
class RunResult:
    run_dir: Path
    guides_even_path: Path
    guides_odd_path: Path
    guide_order_table_path: Path
    round_summary_path: Optional[Path]
    candidate_tiles_path: Optional[Path]
    log_lines: List[str]


APP_TITLE = "PureTarget designer"
DEFAULT_OUTROOT = "streamlit_runs"


def quote_cmd(parts: List[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


@st.cache_data(show_spinner=False)
def read_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


@st.cache_data(show_spinner=False)
def read_table(path: str, sep: str = "\t") -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p, sep=sep)


@st.cache_data(show_spinner=False)
def make_candidate_geometry_figure(path: str):
    df = read_table(path)
    if df.empty:
        return None

    start_col = "tile_start_1based" if "tile_start_1based" in df.columns else "start_1based"
    end_col = "tile_end_1based" if "tile_end_1based" in df.columns else "end_1based"
    required = {"round", "candidate_plan_id", start_col, end_col}
    if not required.issubset(df.columns):
        return None

    work = df.copy()
    work[start_col] = pd.to_numeric(work[start_col], errors="coerce")
    work[end_col] = pd.to_numeric(work[end_col], errors="coerce")
    work = work.dropna(subset=["round", "candidate_plan_id", start_col, end_col]).copy()
    if work.empty:
        return None

    work["round"] = pd.to_numeric(work["round"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["round"]).copy()
    work["round"] = work["round"].astype(int)

    row_df = (
        work.groupby(["round", "candidate_plan_id"], dropna=False)
        .agg(xmin=(start_col, "min"), xmax=(end_col, "max"), n_tiles=("tile_id", "count"))
        .reset_index()
        .sort_values(["round", "candidate_plan_id"], kind="stable")
    )
    if row_df.empty:
        return None

    max_rows = 40
    if len(row_df) > max_rows:
        row_df = row_df.head(max_rows).copy()
        keep_keys = set(tuple(x) for x in row_df[["round", "candidate_plan_id"]].itertuples(index=False, name=None))
        work = work[work[["round", "candidate_plan_id"]].apply(tuple, axis=1).isin(keep_keys)].copy()

    locus_min = int(work[start_col].min())
    locus_max = int(work[end_col].max())
    span = max(1, locus_max - locus_min + 1)

    fig_h = max(4, 0.45 * len(row_df) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    y_positions = list(range(len(row_df)))
    ax.set_xlim(locus_min, locus_max)
    ax.set_ylim(-1, len(row_df))
    ax.set_xlabel("Genomic coordinate")
    ax.set_ylabel("Round / candidate")

    labels: List[str] = []
    for y, (_, crow) in zip(y_positions, row_df.iterrows()):
        labels.append(f"r{crow['round']} {crow['candidate_plan_id']}")
        mask = (work["round"] == crow["round"]) & (work["candidate_plan_id"] == crow["candidate_plan_id"])
        sub = work.loc[mask].sort_values([start_col, end_col], kind="stable")
        for _, row in sub.iterrows():
            start = int(row[start_col])
            width = max(1, int(row[end_col]) - start + 1)
            ax.broken_barh([(start, width)], (y - 0.3, 0.6))

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def ensure_path(path_text: str, kind: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    return path


def build_commands(
    scripts_dir: Path,
    run_dir: Path,
    locus: str,
    target_tile_size: int,
    min_tile_size: int,
    max_tile_size: int,
    force_single_tile: bool,
) -> List[List[str]]:
    py = sys.executable

    step01 = [
        py,
        str(scripts_dir / "01_parse_locus.py"),
        "--locus",
        locus,
        "--outdir",
        str(run_dir),
    ]

    step02 = [
        py,
        str(scripts_dir / "02_make_tiles.py"),
        "--locus-json",
        str(run_dir / "01_locus_annotation" / "locus.json"),
        "--common-snps-bed",
        str(run_dir / "01_locus_annotation" / "common_snps.bed"),
        "--outdir",
        str(run_dir / "02_tiles"),
        "--target-tile-size",
        str(target_tile_size),
        "--target-overlap-size",
        "2000",
        "--min-tile-size",
        str(min_tile_size),
        "--max-tile-size",
        str(max_tile_size),
        "--boundary-shift-max",
        "1000",
        "--boundary-shift-step",
        "100",
    ]
    if force_single_tile:
        step02.append("--force-single-tile")

    step03 = [
        py,
        str(scripts_dir / "03_make_boundary_windows.py"),
        "--locus-json",
        str(run_dir / "01_locus_annotation" / "locus.json"),
        "--tiles-tsv",
        str(run_dir / "02_tiles" / "tiles_phasing_optimized.tsv"),
        "--outdir",
        str(run_dir / "03_boundaries"),
    ]

    step04 = [
        py,
        str(scripts_dir / "04_fetch_guides.py"),
        "--boundary-windows-tsv",
        str(run_dir / "03_boundaries" / "boundary_windows.tsv"),
        "--outdir",
        str(run_dir / "04_candidates"),
    ]

    step05 = [
        py,
        str(scripts_dir / "05_filter_orientation.py"),
        "--candidates-raw-tsv",
        str(run_dir / "04_candidates" / "candidates_raw.tsv"),
        "--outdir",
        str(run_dir / "05_oriented"),
    ]

    step06a = [
        py,
        str(scripts_dir / "06a_annotate_guides.py"),
        "--candidates-oriented-tsv",
        str(run_dir / "05_oriented" / "candidates_oriented.tsv"),
        "--common-snps-bed",
        str(run_dir / "01_locus_annotation" / "common_snps.bed"),
        "--outdir",
        str(run_dir / "06a_annotated"),
    ]

    step06d = [
        py,
        str(scripts_dir / "06d_rank_guides.py"),
        "--candidates-annotated-tsv",
        str(run_dir / "06a_annotated" / "candidates_annotated.tsv"),
        "--outdir",
        str(run_dir / "06d_ranked"),
    ]

    step07 = [
        py,
        str(scripts_dir / "07_pair_guides.py"),
        "--candidates-scored-tsv",
        str(run_dir / "06d_ranked" / "candidates_scored.tsv"),
        "--drop-unacceptable-pairs",
        "--outdir",
        str(run_dir / "07_paired"),
    ]

    step08 = [
        py,
        str(scripts_dir / "08_optimize_tiles.py"),
        "--guide-pairs-tsv",
        str(run_dir / "07_paired" / "guide_pairs_top.tsv"),
        "--outdir",
        str(run_dir / "08_global"),
    ]

    step09a = [
        py,
        str(scripts_dir / "09a_detect_problematic_tiles.py"),
        "--selected-pairs-tsv",
        str(run_dir / "08_global" / "guide_pairs_global_selected.tsv"),
        "--tile-plan-tsv",
        str(run_dir / "02_tiles" / "tiles_phasing_optimized.tsv"),
        "--tile-overlaps-tsv",
        str(run_dir / "02_tiles" / "tile_overlaps_phasing_optimized.tsv"),
        "--outdir",
        str(run_dir / "09a_problem_tiles"),
    ]

    step09b = [
        py,
        str(scripts_dir / "09b_retile_difficult_intervals.py"),
        "--tiles-problematic-tsv",
        str(run_dir / "09a_problem_tiles" / "tiles_problematic.tsv"),
        "--tile-plan-tsv",
        str(run_dir / "02_tiles" / "tiles_phasing_optimized.tsv"),
        "--outdir",
        str(run_dir / "09b_revised_tiles"),
    ]

    step09c = [
        py,
        str(scripts_dir / "09c_rerun_revised_regions.py"),
        "--selected-pairs-tsv",
        str(run_dir / "08_global" / "guide_pairs_global_selected.tsv"),
        "--tile-plan-tsv",
        str(run_dir / "02_tiles" / "tiles_phasing_optimized.tsv"),
        "--locus-json",
        str(run_dir / "01_locus_annotation" / "locus.json"),
        "--common-snps-bed",
        str(run_dir / "01_locus_annotation" / "common_snps.bed"),
        "--scripts-dir",
        str(scripts_dir),
        "--base-output-dir",
        str(run_dir),
    ]

    step10a = [
        py,
        str(scripts_dir / "10a_assign_odd_even_pools.py"),
        "--selected-pairs-final-tsv",
        str(run_dir / "09c_redesign_iterative" / "selected_pairs_final.tsv"),
        "--outdir",
        str(run_dir / "10_pools"),
    ]

    step10b = [
        py,
        str(scripts_dir / "10b_create_pool_specific_exports.py"),
        "--selected-pairs-final-tsv",
        str(run_dir / "09c_redesign_iterative" / "selected_pairs_final.tsv"),
        "--tile-pool-assignment-tsv",
        str(run_dir / "10_pools" / "tile_pool_assignment.tsv"),
        "--outdir",
        str(run_dir / "10_pools"),
    ]

    step11a = [
        py,
        str(scripts_dir / "11a_generate_final_guide_order_table.py"),
        "--selected-pairs-final-tsv",
        str(run_dir / "09c_redesign_iterative" / "selected_pairs_final.tsv"),
        "--tile-pool-assignment-tsv",
        str(run_dir / "10_pools" / "tile_pool_assignment.tsv"),
        "--outdir",
        str(run_dir / "11_deliverables"),
    ]

    step11c = [
        py,
        str(scripts_dir / "11c_generate_human_readable_report.py"),
        "--selected-pairs-final-tsv",
        str(run_dir / "09c_redesign_iterative" / "selected_pairs_final.tsv"),
        "--tile-pool-assignment-tsv",
        str(run_dir / "10_pools" / "tile_pool_assignment.tsv"),
        "--guides-for-ordering-csv",
        str(run_dir / "11_deliverables" / "guides_for_ordering.csv"),
        "--tiles-problematic-tsv",
        str(run_dir / "09a_problem_tiles" / "tiles_problematic.tsv"),
        "--locus-json",
        str(run_dir / "01_locus_annotation" / "locus.json"),
        "--design-search-dir",
        str(run_dir / "09c_redesign_iterative" / "search_history"),
        "--outdir",
        str(run_dir / "11_deliverables"),
    ]

    return [
        step01,
        step02,
        step03,
        step04,
        step05,
        step06a,
        step06d,
        step07,
        step08,
        step09a,
        step09b,
        step09c,
        step10a,
        step10b,
        step11a,
        step11c,
    ]


def run_pipeline(
    repo_root: Path,
    scripts_dir: Path,
    out_root: Path,
    locus: str,
    target_tile_size: int,
    min_tile_size: int,
    max_tile_size: int,
    force_single_tile: bool,
    progress_placeholder,
    log_placeholder,
) -> RunResult:
    run_name = locus.replace(":", "_").replace("-", "_").replace(",", "")
    tmp = tempfile.mkdtemp(prefix=f"ptd_{run_name}_", dir=str(out_root))
    run_dir = Path(tmp)
    commands = build_commands(
        scripts_dir=scripts_dir,
        run_dir=run_dir,
        locus=locus,
        target_tile_size=target_tile_size,
        min_tile_size=min_tile_size,
        max_tile_size=max_tile_size,
        force_single_tile=force_single_tile,
    )

    logs: List[str] = []
    total = len(commands)

    for idx, cmd in enumerate(commands, start=1):
        progress_placeholder.progress((idx - 1) / total, text=f"Running step {idx} of {total}")
        logs.append(f"$ {quote_cmd(cmd)}")
        code, stdout, stderr = run_command(cmd, cwd=repo_root)
        if stdout.strip():
            logs.append(stdout.strip())
        if stderr.strip():
            logs.append(stderr.strip())
        log_placeholder.code("\n\n".join(logs), language="bash")
        if code != 0:
            progress_placeholder.empty()
            raise RuntimeError(f"Command failed with exit code {code}: {quote_cmd(cmd)}")

    progress_placeholder.progress(1.0, text="Done")

    result = RunResult(
        run_dir=run_dir,
        guides_even_path=run_dir / "11_deliverables" / "guides_even.fasta",
        guides_odd_path=run_dir / "11_deliverables" / "guides_odd.fasta",
        guide_order_table_path=run_dir / "11_deliverables" / "guides_for_ordering.csv",
        round_summary_path=run_dir / "09c_redesign_iterative" / "search_history" / "round_summary.tsv",
        candidate_tiles_path=run_dir / "09c_redesign_iterative" / "search_history" / "tile_plan_candidates_all_rounds.tsv",
        log_lines=logs,
    )
    return result


def show_result(result: RunResult):
    st.success(f"Run complete: {result.run_dir}")

    col1, col2 = st.columns(2)
    with col1:
        if result.guides_even_path.exists():
            st.download_button(
                "Download guides_even.fasta",
                data=read_bytes(str(result.guides_even_path)),
                file_name="guides_even.fasta",
                mime="text/plain",
            )
    with col2:
        if result.guides_odd_path.exists():
            st.download_button(
                "Download guides_odd.fasta",
                data=read_bytes(str(result.guides_odd_path)),
                file_name="guides_odd.fasta",
                mime="text/plain",
            )

    st.subheader("Designed guides")
    guide_df = read_table(str(result.guide_order_table_path), sep=",")
    guide_df.columns = ["Tile", "Pool", "Pair score", "Warning", "Tile flank", "Sequence", "Strand", "Cut site", "On-target score", "Off-target score"]
    if guide_df.empty:
        st.info("No guide order table found.")
    else:
        st.dataframe(guide_df, use_container_width=True, hide_index=True)

    st.subheader("Iterative design summary")
    if result.round_summary_path and result.round_summary_path.exists():
        round_df = read_table(str(result.round_summary_path))
        round_df = round_df.loc[["round", "problematic_tiles", "accepted_candidate_plan_id", "accepted_score", "stop_reason"]]
        round_df.columns = ["Round", "Number of problematic tiles", "Accepted design ID", "Accepted design score", "Optimization result"]
        if round_df.empty:
            st.info("Round summary file exists but is empty.")
        else:
            st.dataframe(round_df, use_container_width=True, hide_index=True)
    else:
        st.info("No round summary produced.")

    st.subheader("Designs tested")
    if result.candidate_tiles_path and result.candidate_tiles_path.exists():
        fig = make_candidate_geometry_figure(str(result.candidate_tiles_path))
        if fig is None:
            st.info("Could not render candidate tile geometry.")
        else:
            st.pyplot(fig, clear_figure=True)
    else:
        st.info("No candidate tile geometry file produced.")

    with st.expander("Pipeline log"):
        st.code("\n\n".join(result.log_lines), language="bash")


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Phase 1: thin Streamlit wrapper around the existing step scripts")

with st.form("run_form"):
    locus = st.text_input("Locus coordinates, e.g. chr12:6532449-6540303", value="chr22:42122692-42144483")
    c1, c2, c3 = st.columns(3)
    with c1:
        target_tile_size = st.number_input("Target tile size", min_value=1000, value=10000, step=500)
    with c2:
        min_tile_size = st.number_input("Min tile size", min_value=500, value=5000, step=500)
    with c3:
        max_tile_size = st.number_input("Max tile size", min_value=1000, value=12000, step=500)

    force_single_tile = st.toggle("Disable multi-tile design", value=True)
    submitted = st.form_submit_button("Design guides", use_container_width=True)

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

progress_placeholder = st.empty()
log_placeholder = st.empty()

if submitted:
    try:
        repo_root = ensure_path(repo_root_text, "dir")
        scripts_dir = ensure_path(scripts_dir_text, "dir")
        out_root = Path(out_root_text).expanduser().resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        result = run_pipeline(
            repo_root=repo_root,
            scripts_dir=scripts_dir,
            out_root=out_root,
            locus=locus.strip(),
            target_tile_size=int(target_tile_size),
            min_tile_size=int(min_tile_size),
            max_tile_size=int(max_tile_size),
            force_single_tile=bool(force_single_tile),
            progress_placeholder=progress_placeholder,
            log_placeholder=log_placeholder,
        )
        st.session_state["last_result"] = result
    except Exception as exc:
        st.error(str(exc))

last_result = st.session_state.get("last_result")
if last_result is not None:
    show_result(last_result)
