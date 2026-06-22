#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


LEFT_PREFIX = 'left_'
RIGHT_PREFIX = 'right_'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Pair ranked left/right boundary guides within each tile for downstream optimization.'
    )
    parser.add_argument(
        '--candidates-scored-tsv',
        required=True,
        help='Path to candidates_scored.tsv from current Step 06d.',
    )
    parser.add_argument(
        '--outdir',
        required=True,
        help='Output directory.',
    )
    parser.add_argument(
        '--max-guides-per-boundary',
        type=int,
        default=25,
        help='Max guides to retain per tile/boundary before pair enumeration. Default: 25',
    )
    parser.add_argument(
        '--max-pairs-per-tile',
        type=int,
        default=100,
        help='Max paired-guide candidates to retain per tile after scoring. Default: 100',
    )
    parser.add_argument(
        '--require-nonunacceptable',
        '--require-nonrecalcitrant',
        dest='require_nonunacceptable',
        action='store_true',
        help='If set, exclude guides with design_class == unacceptable before pairing.',
    )
    parser.add_argument(
        '--drop-unacceptable-pairs',
        action='store_true',
        help='If set, drop pairs with pair_design_class == unacceptable after pair scoring/classification.',
    )
    return parser.parse_args()


REQUIRED_COLUMNS = [
    'tile_id',
    'boundary_type',
    'guide_seq',
    'final_rank_score',
    'guide_rank_within_boundary',
    'ontarget_score',
    'offtarget_score',
    'distance_abs_to_boundary_bp',
    'design_class',
]


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'candidates_scored.tsv missing required columns: {missing}')

    df['tile_id'] = df['tile_id'].astype(int)
    df['guide_rank_within_boundary'] = df['guide_rank_within_boundary'].astype(int)
    df['final_rank_score'] = pd.to_numeric(df['final_rank_score'], errors='coerce')
    df['ontarget_score'] = pd.to_numeric(df['ontarget_score'], errors='coerce')
    df['offtarget_score'] = pd.to_numeric(df['offtarget_score'], errors='coerce')
    df['distance_abs_to_boundary_bp'] = pd.to_numeric(df['distance_abs_to_boundary_bp'], errors='coerce')
    df['boundary_type'] = df['boundary_type'].astype(str)
    df['guide_seq'] = df['guide_seq'].astype(str).str.upper()

    return df.sort_values(
        ['tile_id', 'boundary_type', 'guide_rank_within_boundary', 'final_rank_score', 'guide_seq'],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)


def limit_candidates(df: pd.DataFrame, max_guides_per_boundary: int, require_nonunacceptable: bool) -> pd.DataFrame:
    out = df.copy()
    if require_nonunacceptable and 'design_class' in out.columns:
        out = out[out['design_class'] != 'unacceptable'].copy()

    out = (
        out.groupby(['tile_id', 'boundary_type'], group_keys=False)
        .head(max_guides_per_boundary)
        .reset_index(drop=True)
    )
    return out


SAFE_BONUS_CLASSES = {'ideal', 'good'}


def classify_pair(row: pd.Series) -> str:
    if row.get('pair_has_exact_in_locus_offtarget', False):
        return 'unacceptable'
    if row.get('pair_has_guide_too_close_to_target', False):
        return 'unacceptable'
    if row['pair_has_common_snp_overlap']:
        return 'unacceptable'
    if row['pair_min_ontarget_score'] >= 55 and row['pair_min_offtarget_score'] >= 70 and row['pair_total_warnings'] == 0:
        return 'ideal'
    if row['pair_min_ontarget_score'] >= 55 and row['pair_min_offtarget_score'] >= 45:
        return 'good'
    return 'suboptimal'


def compute_pair_score(row: pd.Series) -> float:
    score = 0.0
    score += float(row['left_final_rank_score'])
    score += float(row['right_final_rank_score'])
    score += 0.30 * float(row['pair_min_ontarget_score'])
    score += 0.20 * float(row['pair_min_offtarget_score'])
    score += 0.10 * float(row['pair_mean_ontarget_score'])
    score += 0.10 * float(row['pair_mean_offtarget_score'])
    score -= 4.0 * float(row['pair_total_distance_abs_to_boundary_bp'])
    score -= 6.0 * float(row['pair_total_warnings'])
    score -= 120.0 * float(row.get('pair_has_exact_in_locus_offtarget', False))
    score -= 120.0 * float(row.get('pair_has_guide_too_close_to_target', False))
    score -= 40.0 * float(row['pair_has_common_snp_overlap'])

    if row['left_design_class'] in SAFE_BONUS_CLASSES and row['right_design_class'] in SAFE_BONUS_CLASSES:
        score += 8.0
    if row['left_design_class'] == 'ideal' and row['right_design_class'] == 'ideal':
        score += 6.0

    return round(score, 4)


GUIDE_EXPORT_COLUMNS = [
    'guide_seq',
    'strand',
    'chrom',
    'boundary_type',
    'desired_boundary_1based',
    'cut_site_1based',
    'distance_cut_to_boundary_bp',
    'distance_abs_to_boundary_bp',
    'ontarget_score',
    'ontarget_method',
    'ontarget_class',
    'offtarget_score',
    'offtarget_method',
    'offtarget_class',
    'final_rank_score',
    'guide_rank_within_boundary',
    'design_class',
    'basic_sequence_warning',
    'has_homopolymer',
    'is_low_complexity',
    'overlaps_common_snp',
    'overlaps_repeat',
    'overlaps_segdup',
    'n_common_snp_overlaps',
    'n_repeat_overlaps',
    'n_segdup_overlaps',
    'has_exact_in_locus_offtarget',
    'n_exact_in_locus_offtargets',
    'exact_in_locus_offtarget_sites',
    'guide_too_close_to_target',
    'nearest_target_distance_bp',
    'nearest_target_id',
    'nearest_target_label',
]


def available_prefixed_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = []
    for c in GUIDE_EXPORT_COLUMNS:
        if c in df.columns:
            cols.append(c)
    rename_map = {c: prefix + c for c in cols}
    return cols, rename_map


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    left = df[df['boundary_type'] == 'left'].copy()
    right = df[df['boundary_type'] == 'right'].copy()

    left_cols, left_rename = available_prefixed_columns(left, LEFT_PREFIX)
    right_cols, right_rename = available_prefixed_columns(right, RIGHT_PREFIX)

    left_small = left[['tile_id'] + left_cols].rename(columns=left_rename)
    right_small = right[['tile_id'] + right_cols].rename(columns=right_rename)

    pairs = left_small.merge(right_small, on='tile_id', how='inner')
    if pairs.empty:
        return pairs

    pairs['pair_id'] = 'tile' + pairs['tile_id'].astype(str) + '_pair' + (pairs.groupby('tile_id').cumcount() + 1).astype(str)

    pairs['pair_mean_ontarget_score'] = ((pairs['left_ontarget_score'] + pairs['right_ontarget_score']) / 2.0).round(4)
    pairs['pair_min_ontarget_score'] = pairs[['left_ontarget_score', 'right_ontarget_score']].min(axis=1).round(4)
    pairs['pair_mean_offtarget_score'] = ((pairs['left_offtarget_score'] + pairs['right_offtarget_score']) / 2.0).round(4)
    pairs['pair_min_offtarget_score'] = pairs[['left_offtarget_score', 'right_offtarget_score']].min(axis=1).round(4)

    pairs['pair_total_distance_abs_to_boundary_bp'] = (
        pairs['left_distance_abs_to_boundary_bp'] + pairs['right_distance_abs_to_boundary_bp']
    ).astype(float)
    pairs['pair_max_distance_abs_to_boundary_bp'] = pairs[[
        'left_distance_abs_to_boundary_bp', 'right_distance_abs_to_boundary_bp'
    ]].max(axis=1).astype(float)

    for base_col, pair_col in [
        ('basic_sequence_warning', 'pair_n_basic_sequence_warnings'),
        ('has_homopolymer', 'pair_n_homopolymer_guides'),
        ('is_low_complexity', 'pair_n_low_complexity_guides'),
        ('overlaps_common_snp', 'pair_n_guides_overlapping_common_snp'),
        ('has_exact_in_locus_offtarget', 'pair_n_guides_with_exact_in_locus_offtarget'),
        ('guide_too_close_to_target', 'pair_n_guides_too_close_to_target'),
        ('overlaps_repeat', 'pair_n_guides_overlapping_repeat'),
        ('overlaps_segdup', 'pair_n_guides_overlapping_segdup'),
    ]:
        lcol = LEFT_PREFIX + base_col
        rcol = RIGHT_PREFIX + base_col
        if lcol in pairs.columns and rcol in pairs.columns:
            pairs[pair_col] = pairs[[lcol, rcol]].fillna(False).astype(bool).sum(axis=1).astype(int)
        else:
            pairs[pair_col] = 0

    pairs['pair_total_warnings'] = (
        pairs['pair_n_basic_sequence_warnings']
        + pairs['pair_n_homopolymer_guides']
        + pairs['pair_n_low_complexity_guides']
    ).astype(int)
    pairs['pair_total_basic_sequence_warnings'] = pairs['pair_n_basic_sequence_warnings']
    pairs['pair_total_homopolymer_flags'] = pairs['pair_n_homopolymer_guides']
    pairs['pair_total_low_complexity_flags'] = pairs['pair_n_low_complexity_guides']

    for count_col, pair_col in [
        ('n_common_snp_overlaps', 'pair_total_common_snp_overlaps'),
        ('n_exact_in_locus_offtargets', 'pair_total_exact_in_locus_offtargets'),
        ('n_repeat_overlaps', 'pair_total_repeat_overlaps'),
        ('n_segdup_overlaps', 'pair_total_segdup_overlaps'),
    ]:
        lcol = LEFT_PREFIX + count_col
        rcol = RIGHT_PREFIX + count_col
        if lcol in pairs.columns and rcol in pairs.columns:
            pairs[pair_col] = pairs[[lcol, rcol]].fillna(0).sum(axis=1).astype(int)
        else:
            pairs[pair_col] = 0

    pairs['pair_has_common_snp_overlap'] = pairs['pair_total_common_snp_overlaps'] > 0
    pairs['pair_has_exact_in_locus_offtarget'] = pairs['pair_total_exact_in_locus_offtargets'] > 0
    pairs['pair_has_guide_too_close_to_target'] = pairs['pair_n_guides_too_close_to_target'] > 0
    pairs['pair_has_repeat_overlap'] = pairs['pair_total_repeat_overlaps'] > 0
    pairs['pair_has_segdup_overlap'] = pairs['pair_total_segdup_overlaps'] > 0

    pairs['pair_score'] = pairs.apply(compute_pair_score, axis=1)
    pairs['pair_design_class'] = pairs.apply(classify_pair, axis=1)

    sort_cols = [
        'tile_id',
        'pair_score',
        'pair_min_ontarget_score',
        'pair_min_offtarget_score',
        'pair_total_distance_abs_to_boundary_bp',
        'left_guide_rank_within_boundary',
        'right_guide_rank_within_boundary',
        'left_guide_seq',
        'right_guide_seq',
    ]
    asc = [True, False, False, False, True, True, True, True, True]
    pairs = pairs.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
    pairs['pair_rank_within_tile'] = pairs.groupby('tile_id').cumcount() + 1
    return pairs


def drop_unacceptable_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty or 'pair_design_class' not in pairs.columns:
        return pairs.copy()
    return pairs[pairs['pair_design_class'] != 'unacceptable'].copy().reset_index(drop=True)


def summarize_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame([
            {'metric': 'n_tiles_with_pairs', 'value': 0},
            {'metric': 'n_total_pairs', 'value': 0},
        ])

    rows = [
        {'metric': 'n_tiles_with_pairs', 'value': int(pairs['tile_id'].nunique())},
        {'metric': 'n_total_pairs', 'value': int(pairs.shape[0])},
        {'metric': 'mean_pair_score', 'value': round(float(pairs['pair_score'].mean()), 4)},
        {'metric': 'mean_pair_min_ontarget_score', 'value': round(float(pairs['pair_min_ontarget_score'].mean()), 4)},
        {'metric': 'mean_pair_min_offtarget_score', 'value': round(float(pairs['pair_min_offtarget_score'].mean()), 4)},
        {'metric': 'n_ideal_pairs', 'value': int((pairs['pair_design_class'] == 'ideal').sum())},
        {'metric': 'n_good_pairs', 'value': int((pairs['pair_design_class'] == 'good').sum())},
        {'metric': 'n_suboptimal_pairs', 'value': int((pairs['pair_design_class'] == 'suboptimal').sum())},
        {'metric': 'n_unacceptable_pairs', 'value': int((pairs['pair_design_class'] == 'unacceptable').sum())},
    ]
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scored = load_candidates(Path(args.candidates_scored_tsv))
    reduced = limit_candidates(scored, args.max_guides_per_boundary, args.require_nonunacceptable)

    reduced.to_csv(outdir / 'candidates_scored_for_pairing.tsv', sep='\t', index=False)

    pairs = build_pairs(reduced)
    if args.drop_unacceptable_pairs:
        pairs = drop_unacceptable_pairs(pairs)

    if pairs.empty:
        empty = pd.DataFrame(columns=['tile_id', 'pair_id'])
        empty.to_csv(outdir / 'guide_pairs_all.tsv', sep='\t', index=False)
        empty.to_csv(outdir / 'guide_pairs_top.tsv', sep='\t', index=False)
        summarize_pairs(pairs).to_csv(outdir / 'guide_pairing_summary.tsv', sep='\t', index=False)
        pd.DataFrame(columns=['tile_id']).to_csv(outdir / 'guide_pairing_summary_by_tile.tsv', sep='\t', index=False)
        print('No valid left/right guide pairs could be constructed.')
        return 0

    pairs.to_csv(outdir / 'guide_pairs_all.tsv', sep='\t', index=False)

    top_pairs = (
        pairs.groupby('tile_id', group_keys=False)
        .head(args.max_pairs_per_tile)
        .reset_index(drop=True)
    )
    top_pairs.to_csv(outdir / 'guide_pairs_top.tsv', sep='\t', index=False)

    summary = summarize_pairs(top_pairs)
    summary.to_csv(outdir / 'guide_pairing_summary.tsv', sep='\t', index=False)

    by_tile = (
        top_pairs.groupby('tile_id', sort=True)
        .agg(
            n_pairs=('pair_id', 'count'),
            best_pair_score=('pair_score', 'max'),
            best_pair_min_ontarget=('pair_min_ontarget_score', 'max'),
            best_pair_min_offtarget=('pair_min_offtarget_score', 'max'),
            best_left_guide=('left_guide_seq', 'first'),
            best_right_guide=('right_guide_seq', 'first'),
        )
        .reset_index()
    )
    by_tile.to_csv(outdir / 'guide_pairing_summary_by_tile.tsv', sep='\t', index=False)

    print(summary.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
