#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pandas.errors import EmptyDataError
from pathlib import Path
import pandas as pd

LOCAL = {
    'no_selected_pair','poor_offtarget_options','poor_ontarget_options','guides_far_from_boundaries',
    'guide_overlaps_common_snp'
}
LONG = {'no_selected_pair','poor_offtarget_options','guide_has_exact_in_locus_offtarget'}

def parse_args():
    p = argparse.ArgumentParser(description='Detect problematic tiles and group them into redesign blocks.')
    p.add_argument('--selected-pairs-tsv', required=True)
    p.add_argument('--tile-plan-tsv', required=True)
    p.add_argument('--tile-overlaps-tsv')
    p.add_argument('--outdir', required=True)
    p.add_argument('--min-pair-score', type=float, default=35.0)
    p.add_argument('--min-offtarget-score', type=float, default=20.0)
    p.add_argument('--min-ontarget-score', type=float, default=25.0)
    p.add_argument('--max-fragment-length-bp', type=int, default=12000)
    p.add_argument('--min-fragment-length-bp', type=int, default=500)
    p.add_argument('--max-boundary-distance-bp', type=int, default=150)
    p.add_argument('--min-overlap-length-bp', type=int, default=800)
    p.add_argument('--warn-overlap-length-bp', type=int, default=1200)
    p.add_argument('--min-overlap-snp-count', type=int, default=1)
    p.add_argument('--max-local-block-size', type=int, default=2)
    return p.parse_args()

def choose_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def safe_bool(v):
    if pd.isna(v):
        return False
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {'1','true','t','yes','y'}

def trigger(reasons, severity, block_size, pair_found):
    rs = set(reasons)
    if severity == 'ok':
        return 'none'
    if (not pair_found or rs & LONG) and block_size >= 2:
        return 'long_tile_rescue'
    if block_size <= 2 and rs <= (LOCAL | {'overlap_lacks_common_snps','overlap_short_warning'}):
        return 'local_boundary_shift'
    return 'local_retile'


def find_same_pool_overlaps(tiles_df, start_col, end_col):
    work = tiles_df.copy()
    work['tile_id'] = work['tile_id'].astype(str)
    work = work.sort_values([start_col, end_col]).reset_index(drop=True)
    overlap_map = {tid: [] for tid in work['tile_id'].tolist()}
    rows = work[['tile_id', start_col, end_col]].to_dict('records')

    for i in range(len(rows)):
        for j in range(i + 2, len(rows), 2):
            left = rows[i]
            right = rows[j]
            ov = min(int(left[end_col]), int(right[end_col])) - max(int(left[start_col]), int(right[start_col])) + 1
            if ov > 0:
                overlap_map[str(left['tile_id'])].append({'other_tile_id': str(right['tile_id']), 'overlap_bp': int(ov)})
                overlap_map[str(right['tile_id'])].append({'other_tile_id': str(left['tile_id']), 'overlap_bp': int(ov)})
    return overlap_map


def main():
    a = parse_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    tiles = pd.read_csv(a.tile_plan_tsv, sep='\t').copy()
    pairs = pd.read_csv(a.selected_pairs_tsv, sep='\t').copy()
    overlaps = None
    if a.tile_overlaps_tsv:
        overlap_path = Path(a.tile_overlaps_tsv)
        if overlap_path.exists() and overlap_path.stat().st_size > 0:
            try:
                overlaps = pd.read_csv(overlap_path, sep='\t')
            except EmptyDataError:
                overlaps = None
    tid = choose_col(tiles, ['tile_id','tile_index'])
    scol = choose_col(tiles, ['tile_start_1based','start_1based','tile_start','start'])
    ecol = choose_col(tiles, ['tile_end_1based','end_1based','tile_end','end'])
    ccol = choose_col(tiles, ['chrom','chr'])
    if not all([tid, scol, ecol]):
        raise ValueError('tile plan must contain tile_id and start/end columns')
    tiles['tile_id'] = tiles[tid].astype(str)
    tiles = tiles.sort_values([scol, ecol]).reset_index(drop=True)
    ptid = choose_col(pairs, ['tile_id','tile_index']) if not pairs.empty else None
    if ptid:
        pairs['tile_id'] = pairs[ptid].astype(str)
    same_pool_overlap_map = find_same_pool_overlaps(tiles, scol, ecol)
    overlap_map = {}
    if overlaps is not None and not overlaps.empty:
        lcol = choose_col(overlaps, ['left_tile_id','tile_left','tile1_id','tile_1_id'])
        rcol = choose_col(overlaps, ['right_tile_id','tile_right','tile2_id','tile_2_id'])
        olen = choose_col(overlaps, ['overlap_length_bp','overlap_bp','length_bp'])
        osnp = choose_col(overlaps, ['n_common_snps_in_overlap','n_common_snps','common_snp_count'])
        if lcol and rcol and olen:
            for _, r in overlaps.iterrows():
                overlap_map[str(r[lcol])] = {
                    'overlap_with_next_bp': int(r[olen]),
                    'overlap_common_snp_count': int(r[osnp]) if osnp and not pd.isna(r[osnp]) else None,
                    'next_tile_id': str(r[rcol]),
                }
    recs = []
    for _, t in tiles.iterrows():
        tile_id = str(t['tile_id'])
        sub = pairs[pairs['tile_id'] == tile_id] if ptid else pd.DataFrame()
        chosen = sub.iloc[0] if len(sub) else None
        reasons, severity = [], 'ok'
        start, end = int(t[scol]), int(t[ecol])
        frag = end - start + 1
        overlap_len = overlap_map.get(tile_id, {}).get('overlap_with_next_bp')
        overlap_snp = overlap_map.get(tile_id, {}).get('overlap_common_snp_count')
        if chosen is None:
            reasons.append('no_selected_pair'); severity = 'hard'
        else:
            pair_score = float(chosen.get('pair_score', 0.0))
            min_off = float(chosen.get('pair_min_offtarget_score', chosen.get('offtarget_score', 0.0)))
            min_on = float(chosen.get('pair_min_ontarget_score', chosen.get('ontarget_score', 0.0)))
            bad_dist = int(chosen.get('pair_total_distance_abs_to_boundary_bp', 0))
            if pair_score < a.min_pair_score: reasons.append('low_pair_score')
            if min_off < a.min_offtarget_score: reasons.append('poor_offtarget_options')
            if min_on < a.min_ontarget_score: reasons.append('poor_ontarget_options')
            if int(chosen.get('pair_total_common_snp_overlaps', 0)) > 0: reasons.append('guide_overlaps_common_snp')
            if int(chosen.get('pair_total_low_complexity_flags', 0)) > 0: reasons.append('guide_low_complexity_warning')
            if int(chosen.get('pair_total_homopolymer_flags', 0)) > 0: reasons.append('guide_homopolymer_warning')
            if bad_dist > 2 * a.max_boundary_distance_bp: reasons.append('guides_far_from_boundaries')
            if safe_bool(chosen.get('pair_has_exact_in_locus_offtarget', False)): reasons.append('guide_has_exact_in_locus_offtarget')
            if reasons:
                severity = 'warning'
                if any(r in reasons for r in ['poor_offtarget_options','guide_overlaps_common_snp','guides_far_from_boundaries','guide_has_exact_in_locus_offtarget']):
                    severity = 'hard'
        if frag > a.max_fragment_length_bp: reasons.append('fragment_too_long'); severity = 'hard'
        if frag < a.min_fragment_length_bp: reasons.append('fragment_too_short'); severity = 'hard'
        if overlap_len is not None:
            if overlap_len < a.min_overlap_length_bp: reasons.append('overlap_too_weak_for_phasing'); severity = 'hard'
            elif overlap_len < a.warn_overlap_length_bp and severity == 'ok': reasons.append('overlap_short_warning'); severity = 'warning'
        if overlap_snp is not None and overlap_snp < a.min_overlap_snp_count:
            reasons.append('overlap_lacks_common_snps')
            if severity == 'ok': severity = 'warning'
        same_pool_hits = same_pool_overlap_map.get(tile_id, [])
        if same_pool_hits:
            reasons.append('same_pool_overlap')
            severity = 'hard'
        recs.append({
            'tile_id': tile_id,
            'tile_order': len(recs) + 1,
            'chrom': t[ccol] if ccol else None,
            'tile_start_1based': start,
            'tile_end_1based': end,
            'fragment_length_bp': frag,
            'selected_pair_found': chosen is not None,
            'selected_pair_id': None if chosen is None else chosen.get('pair_id'),
            'pair_score': None if chosen is None else chosen.get('pair_score'),
            'pair_min_ontarget_score': None if chosen is None else chosen.get('pair_min_ontarget_score', chosen.get('ontarget_score')),
            'pair_min_offtarget_score': None if chosen is None else chosen.get('pair_min_offtarget_score', chosen.get('offtarget_score')),
            'pair_total_distance_abs_to_boundary_bp': None if chosen is None else chosen.get('pair_total_distance_abs_to_boundary_bp'),
            'pair_total_common_snp_overlaps': None if chosen is None else chosen.get('pair_total_common_snp_overlaps'),
            'pair_total_repeat_overlaps': None if chosen is None else chosen.get('pair_total_repeat_overlaps'),
            'pair_total_segdup_overlaps': None if chosen is None else chosen.get('pair_total_segdup_overlaps'),
            'pair_has_exact_in_locus_offtarget': False if chosen is None else safe_bool(chosen.get('pair_has_exact_in_locus_offtarget', False)),
            'overlap_with_next_bp': overlap_len,
            'overlap_common_snp_count': overlap_snp,
            'has_same_pool_overlap': bool(same_pool_hits),
            'same_pool_overlap_bp_max': max((int(x['overlap_bp']) for x in same_pool_hits), default=0),
            'same_pool_overlap_partners': json.dumps(same_pool_hits, sort_keys=True),
            'problem_severity': severity,
            'is_problematic': severity != 'ok',
            'problem_reason_count': len(reasons),
            'problem_reasons': ';'.join(reasons),
        })
    out = pd.DataFrame(recs)
    block_ids, block_size_map = [], {}
    current = 0; start_idx = None
    for i, row in out.iterrows():
        if row['is_problematic']:
            if start_idx is None:
                current += 1; start_idx = i
            block_ids.append(f'block_{current}')
        else:
            if start_idx is not None:
                block_size_map[f'block_{current}'] = i - start_idx
                start_idx = None
            block_ids.append(None)
    if start_idx is not None:
        block_size_map[f'block_{current}'] = len(out) - start_idx
    out['redesign_block_id'] = block_ids
    out['redesign_block_size'] = out['redesign_block_id'].map(block_size_map)
    out['redesign_trigger'] = [trigger((r.problem_reasons or '').split(';') if pd.notna(r.problem_reasons) else [], r.problem_severity, int(r.redesign_block_size) if pd.notna(r.redesign_block_size) else 0, bool(r.selected_pair_found)) for _, r in out.iterrows()]
    out.to_csv(outdir / 'tiles_problematic.tsv', sep='\t', index=False)
    blocks = []
    for bid, sub in out.dropna(subset=['redesign_block_id']).groupby('redesign_block_id', sort=False):
        reasons = sub.assign(problem_reasons=sub['problem_reasons'].fillna('')).problem_reasons.str.split(';').explode().replace('', pd.NA).dropna().value_counts().to_dict()
        blocks.append({
            'redesign_block_id': bid,
            'tile_ids': ','.join(sub['tile_id'].astype(str)),
            'tile_count': int(len(sub)),
            'block_start_1based': int(sub['tile_start_1based'].min()),
            'block_end_1based': int(sub['tile_end_1based'].max()),
            'n_hard_tiles': int((sub['problem_severity'] == 'hard').sum()),
            'n_warning_tiles': int((sub['problem_severity'] == 'warning').sum()),
            'redesign_trigger': sub['redesign_trigger'].mode().iloc[0],
            'reason_counts': json.dumps(reasons, sort_keys=True),
        })
    blocks_df = pd.DataFrame(blocks)
    blocks_df.to_csv(outdir / 'problem_blocks.tsv', sep='\t', index=False)
    summary = {
        'n_tiles': int(len(out)),
        'n_problematic_tiles': int(out['is_problematic'].sum()),
        'n_hard_tiles': int((out['problem_severity'] == 'hard').sum()),
        'n_warning_tiles': int((out['problem_severity'] == 'warning').sum()),
        'n_problem_blocks': int(len(blocks_df)),
    }
    (outdir / 'problematic_tiles_summary.json').write_text(json.dumps(summary, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
