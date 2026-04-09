#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Merge annotation, on-target, and FlashFry off-target scores into final guide ranks.'
    )
    parser.add_argument('--candidates-annotated-tsv', required=True, help='Path to candidates_annotated.tsv from 06a.')
    parser.add_argument('--candidates-ontarget-tsv', required=True, help='Path to candidates_ontarget.tsv from 06b.')
    parser.add_argument('--offtarget-scores-tsv', required=True, help='Path to offtarget_scores.tsv from new 06c.')
    parser.add_argument('--outdir', required=True, help='Output directory.')
    return parser.parse_args()


def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t')


def classify_design(row: pd.Series) -> str:
    if bool(row.get('has_exact_in_locus_offtarget', False)):
        return 'unacceptable'
    if bool(row['overlaps_common_snp']):
        return 'unacceptable'
    if row['ontarget_score'] >= 70 and row['offtarget_score'] >= 80 and not bool(row['basic_sequence_warning']):
        return 'ideal'
    if row['ontarget_score'] >= 60 and row['offtarget_score'] >= 50:
        return 'good'
    return 'suboptimal'


def rank_score(row: pd.Series) -> float:
    score = 0.0
    score += 1.0 * float(row['ontarget_score'])
    score += 0.35 * float(row['offtarget_score'])
    score -= 80.0 * float(row['n_common_snp_overlaps'])
    score -= 150.0 * float(bool(row.get('has_exact_in_locus_offtarget', False)))
    score -= 12.0 * float(row['n_repeat_overlaps'])
    score -= 20.0 * float(row['n_segdup_overlaps'])

    if float(row['offtarget_score']) <= 0.0:
        score -= 30.0

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

    annotated = load_df(Path(args.candidates_annotated_tsv))
    ontarget = load_df(Path(args.candidates_ontarget_tsv))
    offtarget = load_df(Path(args.offtarget_scores_tsv))

    ontarget_cols = [c for c in ['guide_seq', 'ontarget_score', 'ontarget_method', 'ontarget_class'] if c in ontarget.columns]
    offtarget_cols = [
        c for c in [
            'guide_seq', 'offtarget_score', 'offtarget_method', 'offtarget_class',
            'n_exact_total', 'n_extra_exact', 'n_exact_in_locus_offtargets', 'has_exact_in_locus_offtarget',
            'exact_in_locus_offtarget_sites', 'n_mm1', 'n_mm2', 'n_mm3', 'n_mm4',
            'dangerous_in_genome', 'hsu2013', 'doench2016cfd_specificityscore',
            'doench2016cfd_maxot', 'minot', 'basesdifftoclosesthit', 'closesthitcount',
            'flashfry_overflow', 'flashfry_otcount'
        ] if c in offtarget.columns
    ]

    df = annotated.merge(ontarget[ontarget_cols], on=['guide_seq'], how='left')
    df = df.merge(offtarget[offtarget_cols], on=['guide_seq'], how='left')

    for col in ['n_exact_total', 'n_extra_exact', 'n_exact_in_locus_offtargets', 'n_mm1', 'n_mm2', 'n_mm3', 'n_mm4', 'dangerous_in_genome', 'basesdifftoclosesthit', 'closesthitcount', 'flashfry_otcount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    df['offtarget_score'] = pd.to_numeric(df.get('offtarget_score', 0.0), errors='coerce').fillna(0.0)
    if 'offtarget_class' not in df.columns:
        df['offtarget_class'] = 'unknown'
    else:
        df['offtarget_class'] = df['offtarget_class'].fillna('unknown')
    if 'offtarget_method' not in df.columns:
        df['offtarget_method'] = 'flashfry_missing_default'
    else:
        df['offtarget_method'] = df['offtarget_method'].fillna('flashfry_missing_default')
    if 'has_exact_in_locus_offtarget' not in df.columns:
        df['has_exact_in_locus_offtarget'] = False
    else:
        df['has_exact_in_locus_offtarget'] = df['has_exact_in_locus_offtarget'].fillna(False).astype(bool)
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
        {'metric': 'mean_final_rank_score', 'value': round(float(df['final_rank_score'].mean()), 4)},
        {'metric': 'mean_offtarget_score', 'value': round(float(df['offtarget_score'].mean()), 4)},
        {'metric': 'n_ideal', 'value': int((df['design_class'] == 'ideal').sum())},
        {'metric': 'n_good', 'value': int((df['design_class'] == 'good').sum())},
        {'metric': 'n_suboptimal', 'value': int((df['design_class'] == 'suboptimal').sum())},
        {'metric': 'n_unacceptable', 'value': int((df['design_class'] == 'unacceptable').sum())},
        {'metric': 'n_with_exact_in_locus_offtarget', 'value': int(df['has_exact_in_locus_offtarget'].sum())},
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
