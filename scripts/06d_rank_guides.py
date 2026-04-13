#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank guides using UCSC-provided CRISPR scores plus local SNP/sequence filters."
    )
    parser.add_argument('--candidates-annotated-tsv', required=True, help='Path to candidates_annotated.tsv from 06a.')
    parser.add_argument('--outdir', required=True, help='Output directory.')
    return parser.parse_args()


def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t')


def classify_design(row: pd.Series) -> str:
    if bool(row['overlaps_common_snp']):
        return 'unacceptable'
    if bool(row.get('has_exact_in_locus_offtarget', False)):
        return 'unacceptable'
    if row['ontarget_score'] >= 55 and row['offtarget_score'] >= 70 and not bool(row['basic_sequence_warning']):
        return 'ideal'
    if row['ontarget_score'] >= 30 and row['offtarget_score'] >= 50:
        return 'good'
    return 'suboptimal'


def rank_score(row: pd.Series) -> float:
    score = 0.0
    score += 1.00 * float(row['ontarget_score'])
    score += 0.50 * float(row['offtarget_score'])
    score -= 80.0 * float(row['n_common_snp_overlaps'])
    score -= 150.0 * float(bool(row.get('has_exact_in_locus_offtarget', False)))
    if bool(row['basic_sequence_warning']):
        score -= 8.0
    if bool(row['has_homopolymer']):
        score -= 6.0
    if bool(row['is_low_complexity']):
        score -= 6.0
    score -= 0.05 * float(row['distance_abs_to_boundary_bp'])
    return round(score, 4)


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_df(Path(args.candidates_annotated_tsv)).copy()

    # UCSC CRISPR track exposes MIT specificity (0-100) directly and Doench 2016 as a percentile.
    df['offtarget_score'] = pd.to_numeric(df.get('ucsc_mit_specificity_score', 0.0), errors='coerce').fillna(0.0)
    df['offtarget_method'] = 'ucsc_crispr_mit_specificity'
    df['offtarget_class'] = pd.cut(
        df['offtarget_score'],
        bins=[-1, 50, 80, 101],
        labels=['low', 'medium', 'high'],
    ).astype(str)

    df['ontarget_score'] = pd.to_numeric(df.get('ucsc_doench2016_percentile', 0.0), errors='coerce').fillna(0.0)
    fallback_moreno = pd.to_numeric(df.get('ucsc_moreno_mateos_percentile', 0.0), errors='coerce').fillna(0.0)
    missing_on = df['ontarget_score'] <= 0
    df.loc[missing_on, 'ontarget_score'] = fallback_moreno.loc[missing_on]
    df['ontarget_method'] = 'ucsc_crispr_doench2016_percentile'
    df.loc[missing_on, 'ontarget_method'] = 'ucsc_crispr_moreno_mateos_percentile'
    df['ontarget_class'] = pd.cut(
        df['ontarget_score'],
        bins=[-1, 30, 55, 101],
        labels=['low', 'medium', 'high'],
    ).astype(str)

    if 'has_exact_in_locus_offtarget' not in df.columns:
        df['has_exact_in_locus_offtarget'] = False
    else:
        df['has_exact_in_locus_offtarget'] = df['has_exact_in_locus_offtarget'].fillna(False).astype(bool)
    if 'n_exact_in_locus_offtargets' not in df.columns:
        df['n_exact_in_locus_offtargets'] = 0
    else:
        df['n_exact_in_locus_offtargets'] = pd.to_numeric(df['n_exact_in_locus_offtargets'], errors='coerce').fillna(0).astype(int)
    if 'exact_in_locus_offtarget_sites' not in df.columns:
        df['exact_in_locus_offtarget_sites'] = ''
    else:
        df['exact_in_locus_offtarget_sites'] = df['exact_in_locus_offtarget_sites'].fillna('')

    df['final_rank_score'] = df.apply(rank_score, axis=1)
    df['design_class'] = df.apply(classify_design, axis=1)

    df = df.sort_values(
        ['tile_id', 'boundary_type', 'final_rank_score', 'ontarget_score', 'offtarget_score', 'distance_abs_to_boundary_bp', 'guide_seq'],
        ascending=[True, True, False, False, False, True, True],
    ).reset_index(drop=True)
    df['guide_rank_within_boundary'] = df.groupby(['tile_id', 'boundary_type']).cumcount() + 1

    df.to_csv(outdir / 'candidates_scored.tsv', sep='\t', index=False)

    summary = pd.DataFrame([
        {'metric': 'n_guides', 'value': int(df.shape[0])},
        {'metric': 'mean_final_rank_score', 'value': round(float(df['final_rank_score'].mean()), 4) if not df.empty else 0.0},
        {'metric': 'mean_offtarget_score', 'value': round(float(df['offtarget_score'].mean()), 4) if not df.empty else 0.0},
        {'metric': 'mean_ontarget_score', 'value': round(float(df['ontarget_score'].mean()), 4) if not df.empty else 0.0},
        {'metric': 'n_ideal', 'value': int((df['design_class'] == 'ideal').sum())},
        {'metric': 'n_good', 'value': int((df['design_class'] == 'good').sum())},
        {'metric': 'n_suboptimal', 'value': int((df['design_class'] == 'suboptimal').sum())},
        {'metric': 'n_unacceptable', 'value': int((df['design_class'] == 'unacceptable').sum())},
    ])
    summary.to_csv(outdir / 'guide_ranking_summary.tsv', sep='\t', index=False)

    by_boundary = (
        df.groupby(['tile_id', 'boundary_type'], sort=True)
        .agg(
            n_guides=('guide_seq', 'count'),
            best_final_rank_score=('final_rank_score', 'max'),
            best_ontarget=('ontarget_score', 'max'),
            best_offtarget=('offtarget_score', 'max'),
        )
        .reset_index()
    )
    by_boundary.to_csv(outdir / 'guide_ranking_summary_by_boundary.tsv', sep='\t', index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
