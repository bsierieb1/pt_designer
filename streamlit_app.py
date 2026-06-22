from __future__ import annotations

import shlex
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import streamlit as st


@dataclass
class RunResult:
    mode: str
    run_dir: Path
    guides_path: Optional[Path]
    guides_even_path: Path
    guides_odd_path: Path
    guide_order_table_path: Path
    round_summary_path: Optional[Path]
    candidate_tiles_path: Optional[Path]
    log_lines: List[str]


APP_TITLE = "PureTarget designer"
DEFAULT_OUTROOT = "streamlit_runs"
LOCUS_RE = re.compile(r"^(chr[\w]+):(\d[\d,]*)-(\d[\d,]*)$")
REGION_TOKEN_RE = re.compile(r"^(chr[\w]+):(\d[\d,]*)-(\d[\d,]*)$")
HOTSPOT_EXAMPLE = """chr9\t108874893\t108874894\tELP1:c.3931+1G>T
chr9\t108878680\t108878681\tELP1:c.3643dupG
chr9\t108878730\t108878731\tELP1:c.3592C>T"""


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
        .agg(
            xmin=(start_col, "min"),
            xmax=(end_col, "max"),
            n_tiles=("tile_id", "count"),
        )
        .reset_index()
        .sort_values(["round", "candidate_plan_id"], kind="stable")
    )
    if row_df.empty:
        return None

    max_rows = 40
    if len(row_df) > max_rows:
        row_df = row_df.head(max_rows).copy()
        keep_keys = set(
            tuple(x)
            for x in row_df[["round", "candidate_plan_id"]].itertuples(index=False, name=None)
        )
        work = work[
            work[["round", "candidate_plan_id"]].apply(tuple, axis=1).isin(keep_keys)
        ].copy()

    locus_min = int(work[start_col].min())
    locus_max = int(work[end_col].max())

    fig_h = max(4, 0.55 * len(row_df) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))

    colors = ["#4C78A8", "#F58518"]
    y_positions = list(range(len(row_df)))
    ax.set_xlim(locus_min, locus_max)
    ax.set_ylim(-0.8, len(row_df) - 0.2)
    ax.set_xlabel("Genomic coordinate")
    ax.set_ylabel("Round / candidate")

    labels: List[str] = []
    for y, (_, crow) in zip(y_positions, row_df.iterrows()):
        labels.append(f"r{crow['round']} {crow['candidate_plan_id']}")
        mask = (
            (work["round"] == crow["round"])
            & (work["candidate_plan_id"] == crow["candidate_plan_id"])
        )
        sub = work.loc[mask].sort_values([start_col, end_col], kind="stable").reset_index(drop=True)

        for i, (_, row) in enumerate(sub.iterrows(), start=1):
            start = int(row[start_col])
            end = int(row[end_col])
            width = max(1, end - start + 1)

            # Stagger odd/even tiles vertically to make overlaps easier to see.
            is_odd_tile = (i % 2 == 1)
            y_center = y - 0.12 if is_odd_tile else y + 0.12
            rect_y = y_center - 0.18
            rect_h = 0.28

            rect = Rectangle(
                (start, rect_y),
                width,
                rect_h,
                facecolor=colors[(i - 1) % len(colors)],
                edgecolor="black",
                linewidth=1.0,
            )
            ax.add_patch(rect)

            ax.vlines([start, end], rect_y - 0.02, rect_y + rect_h + 0.02, linewidth=0.8)

            if width > max(250, (locus_max - locus_min) * 0.02):
                ax.text(
                    start + width / 2,
                    y_center,
                    f"T{i}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.98, bottom=0.12)
    return fig



def validate_locus_syntax(locus: str) -> str:
    locus = locus.strip()
    m = LOCUS_RE.match(locus)
    if not m:
        raise ValueError(
            "Invalid locus format. Use chr:start-end, for example chr9:27545914-27573770"
        )

    start = int(m.group(2).replace(",", ""))
    end = int(m.group(3).replace(",", ""))

    if start < 1:
        raise ValueError("Locus start must be >= 1.")
    if end < start:
        raise ValueError("Locus end must be greater than or equal to start.")

    return f"{m.group(1)}:{start}-{end}"


def _parse_coord_value(text: str) -> int:
    return int(str(text).replace(",", ""))


def parse_hotspot_targets_text(targets_text: str, coordinate_system: str) -> pd.DataFrame:
    rows = []
    for raw_line in targets_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        chrom = None
        start = None
        end = None
        label = ""

        m = REGION_TOKEN_RE.match(parts[0]) if parts else None
        if m:
            chrom = m.group(1)
            start = _parse_coord_value(m.group(2))
            end = _parse_coord_value(m.group(3))
            label = " ".join(parts[1:])
        elif len(parts) >= 3:
            chrom = parts[0]
            start = _parse_coord_value(parts[1])
            end = _parse_coord_value(parts[2])
            label = " ".join(parts[3:])

        if chrom is None or start is None or end is None:
            raise ValueError(f"Could not parse hotspot target line: {raw_line}")
        if not str(chrom).startswith("chr"):
            raise ValueError(f"Hotspot target chromosome must start with chr: {raw_line}")

        if coordinate_system == "0based":
            if end <= start:
                raise ValueError(f"0-based half-open target must have end > start: {raw_line}")
            start_1based = start + 1
            end_1based = end
        else:
            if end < start:
                raise ValueError(f"1-based inclusive target must have end >= start: {raw_line}")
            start_1based = start
            end_1based = end

        rows.append(
            {
                "target_id": len(rows) + 1,
                "chrom": chrom,
                "input_start": start,
                "input_end": end,
                "coordinate_system": coordinate_system,
                "start_1based": start_1based,
                "end_1based": end_1based,
                "label": label or f"target_{len(rows) + 1}",
                "input_line": raw_line,
            }
        )

    if not rows:
        raise ValueError("Hotspot mode requires at least one target region.")

    df = pd.DataFrame(rows)
    if df["chrom"].nunique() != 1:
        raise ValueError("Hotspot mode currently requires all target regions to be on one chromosome.")
    df = df.sort_values(["chrom", "start_1based", "end_1based", "target_id"]).reset_index(drop=True)
    df["target_id"] = range(1, len(df) + 1)
    return df


def hotspot_locus_from_targets(targets_df: pd.DataFrame) -> str:
    chrom = str(targets_df["chrom"].iloc[0])
    start = int(targets_df["start_1based"].min())
    end = int(targets_df["end_1based"].max())
    return f"{chrom}:{start}-{end}"

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
    design_mode: str,
    locus: str,
    target_tile_size: int,
    min_tile_size: int,
    max_tile_size: int,
    force_single_tile: bool,
    hotspot_targets_tsv: Optional[Path] = None,
    min_tile_gap_bp: int = 500,
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

    if design_mode == "hotspot":
        if hotspot_targets_tsv is None:
            raise ValueError("Hotspot mode requires a normalized target TSV.")

        step02b = [
            py,
            str(scripts_dir / "02b_make_hotspot_tiles.py"),
            "--targets-tsv",
            str(hotspot_targets_tsv),
            "--locus-json",
            str(run_dir / "01_locus_annotation" / "locus.json"),
            "--outdir",
            str(run_dir / "02_tiles"),
            "--target-tile-size",
            str(target_tile_size),
            "--max-tile-size",
            str(max_tile_size),
            "--min-tile-gap-bp",
            str(min_tile_gap_bp),
        ]

        step09h = [
            py,
            str(scripts_dir / "09h_optimize_hotspot_plans.py"),
            "--candidate-plans-tsv",
            str(run_dir / "02_tiles" / "candidate_plans_summary.tsv"),
            "--tile-plan-candidates-tsv",
            str(run_dir / "02_tiles" / "tile_plan_candidates.tsv"),
            "--targets-tsv",
            str(hotspot_targets_tsv),
            "--locus-json",
            str(run_dir / "01_locus_annotation" / "locus.json"),
            "--scripts-dir",
            str(scripts_dir),
            "--base-output-dir",
            str(run_dir),
            "--common-snps-bed",
            str(run_dir / "01_locus_annotation" / "common_snps.bed"),
            "--ucsc-cache-dir",
            str(run_dir / "ucsc_cache"),
            "--max-tile-size-bp",
            str(max_tile_size),
            "--min-tile-gap-bp",
            str(min_tile_gap_bp),
            "--min-guide-target-distance-bp",
            "50",
        ]

        selected_final = run_dir / "09h_hotspot_optimization" / "selected_pairs_final.tsv"
        design_search_dir = run_dir / "09h_hotspot_optimization" / "search_history"

        step11a = [
            py,
            str(scripts_dir / "11a_generate_final_guide_order_table.py"),
            "--selected-pairs-final-tsv",
            str(selected_final),
            "--unpooled",
            "--outdir",
            str(run_dir / "11_deliverables"),
        ]

        step11c = [
            py,
            str(scripts_dir / "11c_generate_human_readable_report.py"),
            "--selected-pairs-final-tsv",
            str(selected_final),
            "--guides-for-ordering-csv",
            str(run_dir / "11_deliverables" / "guides_for_ordering.csv"),
            "--locus-json",
            str(run_dir / "01_locus_annotation" / "locus.json"),
            "--design-search-dir",
            str(design_search_dir),
            "--unpooled",
            "--outdir",
            str(run_dir / "11_deliverables"),
        ]

        return [
            step01,
            step02b,
            step09h,
            step11a,
            step11c,
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
        "--cache-dir",
        str(run_dir / "ucsc_cache"),
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
        "--ucsc-cache-dir",
        str(run_dir / "ucsc_cache"),
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
    design_mode: str,
    locus: Optional[str],
    target_tile_size: int,
    min_tile_size: int,
    max_tile_size: int,
    force_single_tile: bool,
    hotspot_targets_text: str,
    hotspot_coordinate_system: str,
    min_tile_gap_bp: int,
    progress_placeholder
) -> RunResult:
    hotspot_targets_tsv: Optional[Path] = None
    if design_mode == "hotspot":
        targets_df = parse_hotspot_targets_text(hotspot_targets_text, hotspot_coordinate_system)
        locus = hotspot_locus_from_targets(targets_df)
    elif locus is None:
        raise ValueError("Multi-tile mode requires locus coordinates.")

    run_name = locus.replace(":", "_").replace("-", "_").replace(",", "")
    tmp = tempfile.mkdtemp(prefix=f"ptd_{run_name}_", dir=str(out_root))
    run_dir = Path(tmp)

    if design_mode == "hotspot":
        hotspot_dir = run_dir / "00_hotspot_targets"
        hotspot_dir.mkdir(parents=True, exist_ok=True)
        targets_df.to_csv(hotspot_dir / "targets_normalized.tsv", sep="\t", index=False)
        (hotspot_dir / "targets_input.txt").write_text(hotspot_targets_text, encoding="utf-8")
        hotspot_targets_tsv = hotspot_dir / "targets_normalized.tsv"

    commands = build_commands(
        scripts_dir=scripts_dir,
        run_dir=run_dir,
        design_mode=design_mode,
        locus=locus,
        target_tile_size=target_tile_size,
        min_tile_size=min_tile_size,
        max_tile_size=max_tile_size,
        force_single_tile=force_single_tile,
        hotspot_targets_tsv=hotspot_targets_tsv,
        min_tile_gap_bp=min_tile_gap_bp,
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
        if code != 0:
            progress_placeholder.empty()
            raise RuntimeError(f"Command failed with exit code {code}: {quote_cmd(cmd)}")

    progress_placeholder.progress(1.0, text="Done")

    if design_mode == "hotspot":
        round_summary_path = run_dir / "09h_hotspot_optimization" / "search_history" / "round_summary.tsv"
        candidate_tiles_path = run_dir / "09h_hotspot_optimization" / "search_history" / "tile_plan_candidates_all_rounds.tsv"
    else:
        round_summary_path = run_dir / "09c_redesign_iterative" / "search_history" / "round_summary.tsv"
        candidate_tiles_path = run_dir / "09c_redesign_iterative" / "search_history" / "tile_plan_candidates_all_rounds.tsv"

    result = RunResult(
        mode=design_mode,
        run_dir=run_dir,
        guides_path=run_dir / "11_deliverables" / "guides.fasta" if design_mode == "hotspot" else None,
        guides_even_path=run_dir / "11_deliverables" / "guides_even.fasta",
        guides_odd_path=run_dir / "11_deliverables" / "guides_odd.fasta",
        guide_order_table_path=run_dir / "11_deliverables" / "guides_for_ordering.csv",
        round_summary_path=round_summary_path,
        candidate_tiles_path=candidate_tiles_path,
        log_lines=logs,
    )
    return result


def show_result(result: RunResult):
    st.success(f"Design completed. Note: the tool produces an optimal set of guides within given search parameters, but there's **NO GUARANTEE THIS IS A GOOD SET**. *Please examine the outputs critically*.")

    def _has_nonempty_file(path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0

    if result.mode == "hotspot":
        guides_ready = result.guides_path is not None and _has_nonempty_file(result.guides_path)
        if guides_ready and result.guides_path is not None:
            st.download_button(
                "Download guides.fasta",
                data=read_bytes(str(result.guides_path)),
                file_name="guides.fasta",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.info("No FASTA guide file was produced.")
    else:
        even_ready = _has_nonempty_file(result.guides_even_path)
        odd_ready = _has_nonempty_file(result.guides_odd_path)

        if even_ready and odd_ready:
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download guides_even.fasta",
                    data=read_bytes(str(result.guides_even_path)),
                    file_name="guides_even.fasta",
                    mime="text/plain",
                )
            with col2:
                st.download_button(
                    "Download guides_odd.fasta",
                    data=read_bytes(str(result.guides_odd_path)),
                    file_name="guides_odd.fasta",
                    mime="text/plain",
                )
        elif odd_ready:
            st.download_button(
                "Download guides.fasta",
                data=read_bytes(str(result.guides_odd_path)),
                file_name="guides.fasta",
                mime="text/plain",
                use_container_width=True,
            )
        elif even_ready:
            st.download_button(
                "Download guides.fasta",
                data=read_bytes(str(result.guides_even_path)),
                file_name="guides.fasta",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.info("No FASTA guide file was produced.")

    st.subheader("Designed guides")
    guide_df = read_table(str(result.guide_order_table_path), sep=",")
    if guide_df.empty:
        st.info("No guide order table found.")
    else:
        if result.mode == "hotspot":
            expected_cols = [
                "Tile",
                "Pair score",
                "Warning",
                "Tile flank",
                "Sequence",
                "Strand",
                "Cut site",
                "On-target score",
                "Off-target score",
            ]
        else:
            expected_cols = [
                "Tile",
                "Pool",
                "Pair score",
                "Warning",
                "Tile flank",
                "Sequence",
                "Strand",
                "Cut site",
                "On-target score",
                "Off-target score",
            ]
        if len(guide_df.columns) == len(expected_cols):
            guide_df.columns = expected_cols
        st.dataframe(guide_df, use_container_width=True, hide_index=True)

    summary_title = "Hotspot plan optimization summary" if result.mode == "hotspot" else "Iterative redesign summary (if initial design attempt had issues)"
    visualization_title = "Candidate tile plan visualization" if result.mode == "hotspot" else "Iterative redesign visualization"

    st.subheader(summary_title)
    if result.round_summary_path and result.round_summary_path.exists():
        round_df = read_table(str(result.round_summary_path))
        needed_cols = ["round", "problematic_tiles", "accepted_candidate_plan_id", "accepted_score", "stop_reason"]
        present_cols = [c for c in needed_cols if c in round_df.columns]
        round_df = round_df[present_cols]
        rename_map = {
            "round": "Round",
            "problematic_tiles": "Number of problematic tiles",
            "accepted_candidate_plan_id": "Accepted design ID",
            "accepted_score": "Accepted design score",
            "stop_reason": "Optimization result",
        }
        round_df = round_df.rename(columns=rename_map)
        if round_df.empty:
            st.info("Round summary file exists but is empty.")
        else:
            st.dataframe(round_df, use_container_width=True, hide_index=True)
    else:
        st.info("No round summary produced.")

    st.subheader(visualization_title)
    if result.candidate_tiles_path and result.candidate_tiles_path.exists():
        fig = make_candidate_geometry_figure(str(result.candidate_tiles_path))
        if fig is None:
            st.info("Could not render candidate tile geometry.")
        else:
            st.pyplot(fig, clear_figure=True, use_container_width=True)
    else:
        st.info("No candidate tile geometry file produced.")

    with st.expander("Log"):
        st.code("\n\n".join(result.log_lines), language="bash")


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("This is NOT an official PacBio tool.")

repo_root = Path.cwd().resolve()
scripts_dir = (repo_root / "scripts").resolve()
out_root = (repo_root / DEFAULT_OUTROOT).resolve()

mode_label = st.radio(
    "Design mode",
    ["Multi-tile locus", "Hotspot / variant list"],
    horizontal=True,
    index=0,
)
design_mode = "hotspot" if mode_label.startswith("Hotspot") else "multi_tile"

with st.form("run_form"):
    locus: Optional[str] = None
    hotspot_targets_text = ""
    hotspot_coordinate_system = "1based"
    min_tile_gap_bp = 500

    if design_mode == "multi_tile":
        locus = st.text_input("Locus coordinates, e.g. chr9:27545914-27573770", value="chr9:27545914-27573770")
        c1, c2, c3 = st.columns(3)
        with c1:
            target_tile_size = st.number_input("Target tile size", min_value=1000, value=10000, step=500)
        with c2:
            min_tile_size = st.number_input("Min tile size", min_value=500, value=5000, step=500)
        with c3:
            max_tile_size = st.number_input("Max tile size", min_value=1000, value=12000, step=500)
    else:
        hotspot_targets_text = st.text_area(
            "Target regions (either BED-like, e.g. chr9<TAB>108874893<TAB>108874894<TAB>other_fields, or genome browser-like, e.g. chr9:108874893-108874894)",
            value=HOTSPOT_EXAMPLE,
            height=180,
        )
        coord_label = st.radio(
            "Coordinate system",
            ["0-based half-open", "1-based inclusive"],
            horizontal=True,
            index=0,
        )
        hotspot_coordinate_system = "0based" if coord_label.startswith("0-based") else "1based"
        c1, c2, c3 = st.columns(3)
        with c1:
            target_tile_size = st.number_input("Target tile size", min_value=1000, value=10000, step=500)
        with c2:
            max_tile_size = st.number_input("Max tile size", min_value=1000, value=12000, step=500)
        with c3:
            min_tile_gap_bp = st.number_input("Minimum tile gap", min_value=0, value=500, step=100)
        min_tile_size = 500

    submitted = st.form_submit_button("Design guides", use_container_width=True)

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

progress_placeholder = st.empty()

if submitted:
    try:
        ensure_path(str(repo_root), "dir")
        ensure_path(str(scripts_dir), "dir")
        out_root.mkdir(parents=True, exist_ok=True)

        normalized_locus = validate_locus_syntax(locus) if design_mode == "multi_tile" and locus else None

        result = run_pipeline(
            repo_root=repo_root,
            scripts_dir=scripts_dir,
            out_root=out_root,
            design_mode=design_mode,
            locus=normalized_locus,
            target_tile_size=int(target_tile_size),
            min_tile_size=int(min_tile_size),
            max_tile_size=int(max_tile_size),
            force_single_tile=False,
            hotspot_targets_text=hotspot_targets_text,
            hotspot_coordinate_system=hotspot_coordinate_system,
            min_tile_gap_bp=int(min_tile_gap_bp),
            progress_placeholder=progress_placeholder
        )
        st.session_state["last_result"] = result
    except Exception as exc:
        st.error(str(exc))

last_result = st.session_state.get("last_result")
if last_result is not None:
    show_result(last_result)
