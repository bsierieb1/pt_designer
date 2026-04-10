#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None):
    print("[RUN]", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run the full PureTarget tiled-design pipeline with project defaults. "
            "Only --locus and --outdir are required."
        )
    )
    ap.add_argument("--locus", required=True, help="Genomic locus like chr22:42122692-42144483")
    ap.add_argument("--outdir", required=True, help="Base output directory for the full run")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to use for child scripts")
    ap.add_argument("--scripts-dir", default="scripts", help="Directory containing pipeline scripts")
    ap.add_argument("--refs-dir", default="refs", help="Directory containing reference files")
    ap.add_argument("--tools-dir", default="tools", help="Directory containing helper tools")
    ap.add_argument("--target-tile-size", required=True, type=int, help="Desired tile size in bp for Step 2")
    ap.add_argument("--min-tile-size", required=True, type=int, help="Minimum allowed tile size in bp for Step 2")
    ap.add_argument("--max-tile-size", required=True, type=int, help="Maximum allowed tile size in bp for Step 2")
    ap.add_argument("--target-overlap-size", default=2000, type=int, help="Desired overlap size in bp for Step 2")
    ap.add_argument("--boundary-shift-max", default=1000, type=int, help="Maximum internal boundary shift in bp for Step 2")
    ap.add_argument("--boundary-shift-step", default=100, type=int, help="Boundary shift step size in bp for Step 2")
    args = ap.parse_args()

    py = args.python
    outdir = Path(args.outdir)
    scripts = Path(args.scripts_dir)
    refs = Path(args.refs_dir)
    tools = Path(args.tools_dir)

    outdir.mkdir(parents=True, exist_ok=True)

    script = lambda name: str(scripts / name)
    ref = lambda name: str(refs / name)
    tool = lambda name: str(tools / name)
    out = lambda *parts: str(outdir.joinpath(*parts))

    # 01
    run([
        py, script("01_parse_locus.py"),
        "--locus", args.locus,
        "--fasta", ref("hg38.fa"),
        "--common-snps", ref("common_snps.vcf.gz"),
        "--repeats", ref("repeats.bed.gz"),
        "--segdups", ref("segdups.bed.gz"),
        "--genes", ref("genes.bed.gz"),
        "--outdir", outdir.as_posix(),
    ])

    # 02
    run([
        py, script("02_make_tiles.py"),
        "--locus-json", out("01_locus_annotation", "locus.json"),
        "--common-snps-bed", out("01_locus_annotation", "common_snps.bed"),
        "--outdir", out("02_tiles"),
        "--target-tile-size", str(args.target_tile_size),
        "--target-overlap-size", str(args.target_overlap_size),
        "--min-tile-size", str(args.min_tile_size),
        "--max-tile-size", str(args.max_tile_size),
        "--boundary-shift-max", str(args.boundary_shift_max),
        "--boundary-shift-step", str(args.boundary_shift_step),
    ])

    # 03
    run([
        py, script("03_make_boundary_windows.py"),
        "--locus-json", out("01_locus_annotation", "locus.json"),
        "--tiles-tsv", out("02_tiles", "tiles_phasing_optimized.tsv"),
        "--outdir", out("03_boundaries"),
    ])

    # 04
    run([
        py, script("04_scan_pams.py"),
        "--boundary-windows-tsv", out("03_boundaries", "boundary_windows.tsv"),
        "--fasta", ref("hg38.fa"),
        "--outdir", out("04_candidates"),
    ])

    # 05
    run([
        py, script("05_filter_orientation.py"),
        "--candidates-raw-tsv", out("04_candidates", "candidates_raw.tsv"),
        "--outdir", out("05_oriented"),
    ])

    # 06a
    run([
        py, script("06a_annotate_guides.py"),
        "--candidates-oriented-tsv", out("05_oriented", "candidates_oriented.tsv"),
        "--common-snps-bed", out("01_locus_annotation", "common_snps.bed"),
        "--repeats-bed", out("01_locus_annotation", "repeats.bed"),
        "--segdups-bed", out("01_locus_annotation", "segdups.bed"),
        "--outdir", out("06a_annotated"),
    ])

    # 06b
    run([
        py, script("06b_score_ontarget.py"),
        "--candidates-annotated-tsv", out("06a_annotated", "candidates_annotated.tsv"),
        "--fasta", ref("hg38.fa"),
        "--outdir", out("06b_ontarget"),
    ])

    # 06c
    run([
        py, script("06c_score_offtarget.py"),
        "--candidates-ontarget-tsv", out("06b_ontarget", "candidates_ontarget.tsv"),
        "--locus-json", out("01_locus_annotation", "locus.json"),
        "--fasta", ref("hg38.fa"),
        "--flashfry-jar", tool("FlashFry-assembly-1.15.jar"),
        "--database", ref("flashfry_index"),
        "--outdir", out("06c_offtargets"),
        "--max-mismatches", "3",
    ])

    # 06d
    run([
        py, script("06d_rank_guides.py"),
        "--candidates-annotated-tsv", out("06a_annotated", "candidates_annotated.tsv"),
        "--candidates-ontarget-tsv", out("06b_ontarget", "candidates_ontarget.tsv"),
        "--offtarget-scores-tsv", out("06c_offtargets", "offtarget_scores.tsv"),
        "--outdir", out("06d_ranked"),
    ])

    # 07
    run([
        py, script("07_pair_guides.py"),
        "--candidates-scored-tsv", out("06d_ranked", "candidates_scored.tsv"),
        "--drop-unacceptable-pairs",
        "--outdir", out("07_paired"),
    ])

    # 08
    run([
        py, script("08_optimize_tiles.py"),
        "--guide-pairs-tsv", out("07_paired", "guide_pairs_top.tsv"),
        "--outdir", out("08_global"),
    ])

    # 09a
    run([
        py, script("09a_detect_problematic_tiles.py"),
        "--selected-pairs-tsv", out("08_global", "guide_pairs_global_selected.tsv"),
        "--tile-plan-tsv", out("02_tiles", "tiles_phasing_optimized.tsv"),
        "--tile-overlaps-tsv", out("02_tiles", "tile_overlaps_phasing_optimized.tsv"),
        "--outdir", out("09a_problem_tiles"),
    ])

    # 09b
    run([
        py, script("09b_retile_difficult_intervals.py"),
        "--tiles-problematic-tsv", out("09a_problem_tiles", "tiles_problematic.tsv"),
        "--tile-plan-tsv", out("02_tiles", "tiles_phasing_optimized.tsv"),
        "--outdir", out("09b_revised_tiles"),
    ])

    # 09c
    run([
        py, script("09c_rerun_revised_regions.py"),
        "--selected-pairs-tsv", out("08_global", "guide_pairs_global_selected.tsv"),
        "--tile-plan-tsv", out("02_tiles", "tiles_phasing_optimized.tsv"),
        "--locus-json", out("01_locus_annotation", "locus.json"),
        "--fasta", ref("hg38.fa"),
        "--flashfry-jar", tool("FlashFry-assembly-1.15.jar"),
        "--flashfry-database", ref("flashfry_index"),
        "--scripts-dir", str(scripts),
        "--base-output-dir", outdir.as_posix(),
    ])

    # 10a
    run([
        py, script("10a_assign_odd_even_pools.py"),
        "--selected-pairs-final-tsv", out("09_redesign_iterative", "selected_pairs_final.tsv"),
        "--outdir", out("10_pools"),
    ])

    # 10b
    run([
        py, script("10b_create_pool_specific_exports.py"),
        "--selected-pairs-final-tsv", out("09_redesign_iterative", "selected_pairs_final.tsv"),
        "--tile-pool-assignment-tsv", out("10_pools", "tile_pool_assignment.tsv"),
        "--outdir", out("10_pools"),
    ])

    # 11a
    run([
        py, script("11a_generate_final_guide_order_table.py"),
        "--selected-pairs-final-tsv", out("09_redesign_iterative", "selected_pairs_final.tsv"),
        "--tile-pool-assignment-tsv", out("10_pools", "tile_pool_assignment.tsv"),
        "--outdir", out("11_deliverables"),
    ])

    # 11b
    run([
        py, script("11b_generate_final_bed_and_summary.py"),
        "--selected-pairs-final-tsv", out("09_redesign_iterative", "selected_pairs_final.tsv"),
        "--tile-pool-assignment-tsv", out("10_pools", "tile_pool_assignment.tsv"),
        "--outdir", out("11_deliverables"),
    ])

    # 11c
    run([
        py, script("11c_generate_human_readable_report.py"),
        "--selected-pairs-final-tsv", out("09_redesign_iterative", "selected_pairs_final.tsv"),
        "--tile-pool-assignment-tsv", out("10_pools", "tile_pool_assignment.tsv"),
        "--guides-for-ordering-csv", out("11_deliverables", "guides_for_ordering.csv"),
        "--tiles-problematic-tsv", out("09a_problem_tiles", "tiles_problematic.tsv"),
        "--locus-json", out("01_locus_annotation", "locus.json"),
        "--design-search-dir", out("09_redesign_iterative", "search_history"),
        "--outdir", out("11_deliverables"),
    ])

    print(f"[DONE] Full pipeline completed. Deliverables are in {out('11_deliverables')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
