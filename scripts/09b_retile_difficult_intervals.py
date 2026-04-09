#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description='Generate candidate redesign plans for one problematic block.')
    p.add_argument('--tiles-problematic-tsv', required=True)
    p.add_argument('--tile-plan-tsv', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--block-id')
    p.add_argument('--boundary-shift-jump-bp', type=int, default=1000)
    p.add_argument('--max-boundary-jumps', type=int, default=2)
    p.add_argument('--target-tile-size-bp', type=int, default=10000)
    p.add_argument('--max-tile-size-bp', type=int, default=12000)
    p.add_argument('--target-overlap-size-bp', type=int, default=1200)
    p.add_argument('--min-overlap-bp', type=int, default=800)
    p.add_argument('--min-tile-length-bp', type=int, default=1500)
    p.add_argument('--max-candidate-plans', type=int, default=30)
    return p.parse_args()

def choose_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def normalize_tiles(df):
    tid = choose_col(df, ['tile_id','tile_index'])
    scol = choose_col(df, ['tile_start_1based','start_1based','tile_start','start'])
    ecol = choose_col(df, ['tile_end_1based','end_1based','tile_end','end'])
    ccol = choose_col(df, ['chrom','chr'])
    if not all([tid, scol, ecol]):
        raise ValueError('tile plan must contain tile_id and start/end columns')
    out = df.copy()
    out['tile_id'] = out[tid].astype(str)
    out['chrom'] = out[ccol].astype(str) if ccol else None
    out['tile_start_1based'] = pd.to_numeric(out[scol]).astype(int)
    out['tile_end_1based'] = pd.to_numeric(out[ecol]).astype(int)
    return out.sort_values(['tile_start_1based','tile_end_1based']).reset_index(drop=True)

def select_block(problems, block_id=None):
    probs = problems[problems['is_problematic'].astype(bool)].copy()
    if probs.empty:
        return None
    if block_id:
        sub = probs[probs['redesign_block_id'] == block_id]
        if sub.empty:
            raise ValueError(f'block-id {block_id!r} not found')
        return block_id, sub
    rank = probs.groupby('redesign_block_id', dropna=True).agg(
        n_hard=('problem_severity', lambda s: int((s == 'hard').sum())),
        n_problem=('is_problematic', 'sum'),
        start=('tile_start_1based', 'min'),
    ).reset_index()
    rank = rank.sort_values(['n_hard','n_problem','start'], ascending=[False,False,True])
    bid = rank.iloc[0]['redesign_block_id']
    return bid, probs[probs['redesign_block_id'] == bid].copy()

def retile_interval(tile_ids, chrom, start, end, n_tiles, overlap_bp):
    if n_tiles < 1:
        return []
    total = end - start + 1
    if n_tiles == 1:
        return [{'tile_id': tile_ids[0], 'chrom': chrom, 'tile_start_1based': start, 'tile_end_1based': end}]
    base = math.ceil((total + (n_tiles - 1) * overlap_bp) / n_tiles)
    rows = []
    s = start
    for i in range(n_tiles):
        e = end if i == n_tiles - 1 else s + base - 1
        rows.append({'tile_id': tile_ids[i], 'chrom': chrom, 'tile_start_1based': int(s), 'tile_end_1based': int(e)})
        s = e - overlap_bp + 1
    rows[-1]['tile_end_1based'] = int(end)
    return rows

def shift_seam(tiles, seam_idx, delta_bp, min_tile_bp=1500):
    out = tiles.copy()
    left_i = seam_idx
    right_i = seam_idx + 1
    if left_i < 0 or right_i >= len(out):
        return None
    left_start = int(out.iloc[left_i, out.columns.get_loc('tile_start_1based')])
    left_end = int(out.iloc[left_i, out.columns.get_loc('tile_end_1based')])
    right_start = int(out.iloc[right_i, out.columns.get_loc('tile_start_1based')])
    right_end = int(out.iloc[right_i, out.columns.get_loc('tile_end_1based')])
    new_left_end = left_end + int(delta_bp)
    new_right_start = right_start + int(delta_bp)
    if (new_left_end - left_start + 1) < min_tile_bp:
        return None
    if (right_end - new_right_start + 1) < min_tile_bp:
        return None
    out.iloc[left_i, out.columns.get_loc('tile_end_1based')] = new_left_end
    out.iloc[right_i, out.columns.get_loc('tile_start_1based')] = new_right_start
    return out


def expand_terminal_edge(tiles, tile_idx, delta_bp, direction, min_tile_bp=1500, max_tile_bp=None):
    out = tiles.copy()
    start_col = out.columns.get_loc('tile_start_1based')
    end_col = out.columns.get_loc('tile_end_1based')
    start = int(out.iloc[tile_idx, start_col])
    end = int(out.iloc[tile_idx, end_col])

    if direction == 'left_outward':
        new_start = start - int(delta_bp)
        new_end = end
    elif direction == 'right_outward':
        new_start = start
        new_end = end + int(delta_bp)
    else:
        raise ValueError(f'unsupported terminal expansion direction: {direction}')

    new_len = new_end - new_start + 1
    if new_len < min_tile_bp:
        return None
    if max_tile_bp is not None and new_len > max_tile_bp:
        return None

    out.iloc[tile_idx, start_col] = new_start
    out.iloc[tile_idx, end_col] = new_end
    return out

def apply_boundary_shift(tiles, idxs, delta_left=0, delta_right=0, min_tile_bp=1500):
    out = tiles.copy()
    # Move the seams that touch the problematic block while preserving overlap geometry.
    if idxs[0] > 0 and delta_left:
        shifted = shift_seam(out, idxs[0] - 1, delta_left, min_tile_bp=min_tile_bp)
        if shifted is None:
            return out
        out = shifted
    if idxs[-1] < len(out) - 1 and delta_right:
        shifted = shift_seam(out, idxs[-1], delta_right, min_tile_bp=min_tile_bp)
        if shifted is None:
            return out
        out = shifted
    return out

def block_constraints_ok(tiles, idxs, max_tile, min_tile, min_overlap):
    for i in range(len(tiles)):
        L = int(tiles.iloc[i]['tile_end_1based']) - int(tiles.iloc[i]['tile_start_1based']) + 1
        if L < min_tile or L > max_tile:
            return False
    for i in range(len(tiles) - 1):
        ov = int(tiles.iloc[i]['tile_end_1based']) - int(tiles.iloc[i + 1]['tile_start_1based']) + 1
        if ov < min_overlap:
            return False
    return True

def replace_block(full_tiles, idxs, new_rows):
    before = full_tiles.iloc[:idxs[0]].copy()
    after = full_tiles.iloc[idxs[-1] + 1:].copy()
    block = pd.DataFrame(new_rows)
    out = pd.concat([before, block, after], ignore_index=True)
    out['tile_start_1based'] = out['tile_start_1based'].astype(int)
    out['tile_end_1based'] = out['tile_end_1based'].astype(int)
    return out.sort_values(['tile_start_1based','tile_end_1based']).reset_index(drop=True)

def main():
    a = parse_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    problems = pd.read_csv(a.tiles_problematic_tsv, sep='\t')
    tiles = normalize_tiles(pd.read_csv(a.tile_plan_tsv, sep='\t'))
    selected = select_block(problems, a.block_id)
    if selected is None:
        pd.DataFrame().to_csv(outdir / 'tile_plan_candidates.tsv', sep='\t', index=False)
        pd.DataFrame().to_csv(outdir / 'candidate_plans_summary.tsv', sep='\t', index=False)
        (outdir / 'tiles_revised_summary.json').write_text(json.dumps({'n_candidate_plans': 0}, indent=2))
        return 0
    block_id, block = selected
    idxs = [tiles.index[tiles['tile_id'] == str(t)].tolist()[0] for t in block['tile_id'].astype(str)]
    idxs = sorted(idxs)
    block_tiles = tiles.iloc[idxs].copy()
    block_start = int(block_tiles['tile_start_1based'].min())
    block_end = int(block_tiles['tile_end_1based'].max())
    tile_ids = list(block_tiles['tile_id'].astype(str))
    chrom = str(block_tiles.iloc[0]['chrom'])
    trigger = block['redesign_trigger'].mode().iloc[0]
    plans, plan_rows = [], []
    def add_plan(strategy, strategy_details, full_tiles):
        pid = f'{block_id}_{len(plans)+1:03d}'
        plans.append({'candidate_plan_id': pid, 'redesign_block_id': block_id, 'strategy': strategy, 'strategy_details': strategy_details, 'n_tiles': int(len(full_tiles)), 'block_tile_count': int(len(tile_ids))})
        tmp = full_tiles.copy()
        tmp['candidate_plan_id'] = pid
        tmp['strategy'] = strategy
        tmp['strategy_details'] = strategy_details
        plan_rows.append(tmp)
    add_plan('baseline_keep_current', 'unchanged', tiles)
    if trigger in {'local_boundary_shift','local_retile','long_tile_rescue'}:
        # Shift each seam that touches or lies within the problematic block in 1 kb jumps.
        # Valid seam indices are positional boundaries between adjacent tiles:
        # seam 0 is between tiles 0 and 1, seam N-2 is between tiles N-2 and N-1.
        # Explore every seam that touches or lies within the problematic block.
        first_seam = max(0, idxs[0] - 1)
        last_seam = min(len(tiles) - 2, idxs[-1])
        seam_indices = list(range(first_seam, last_seam + 1)) if last_seam >= first_seam else []
        for seam_idx in seam_indices:
            left_tid = str(tiles.iloc[seam_idx]['tile_id'])
            right_tid = str(tiles.iloc[seam_idx + 1]['tile_id'])
            for jump in range(1, a.max_boundary_jumps + 1):
                delta = jump * a.boundary_shift_jump_bp
                for sign, direction in [(-1, 'upstream'), (1, 'downstream')]:
                    shifted = shift_seam(tiles, seam_idx, sign * delta, min_tile_bp=a.min_tile_length_bp)
                    if shifted is None:
                        continue
                    if block_constraints_ok(shifted, idxs, a.max_tile_size_bp, a.min_tile_length_bp, a.min_overlap_bp):
                        add_plan('local_boundary_shift', f'seam_{left_tid}_{right_tid}_{direction}_{delta}bp', shifted)

        # Allow outward-only expansion of the terminal locus edges when the problematic
        # block touches the first or last tile. This never shrinks the requested target
        # interval; it only explores slightly larger end tiles as rescue candidates.
        if idxs[0] == 0:
            first_tid = str(tiles.iloc[0]['tile_id'])
            for jump in range(1, a.max_boundary_jumps + 1):
                delta = jump * a.boundary_shift_jump_bp
                expanded = expand_terminal_edge(
                    tiles,
                    tile_idx=0,
                    delta_bp=delta,
                    direction='left_outward',
                    min_tile_bp=a.min_tile_length_bp,
                    max_tile_bp=a.max_tile_size_bp,
                )
                if expanded is None:
                    continue
                if block_constraints_ok(expanded, idxs, a.max_tile_size_bp, a.min_tile_length_bp, a.min_overlap_bp):
                    add_plan('local_boundary_shift', f'terminal_left_outward_{first_tid}_{delta}bp', expanded)

        if idxs[-1] == len(tiles) - 1:
            last_tid = str(tiles.iloc[len(tiles) - 1]['tile_id'])
            for jump in range(1, a.max_boundary_jumps + 1):
                delta = jump * a.boundary_shift_jump_bp
                expanded = expand_terminal_edge(
                    tiles,
                    tile_idx=len(tiles) - 1,
                    delta_bp=delta,
                    direction='right_outward',
                    min_tile_bp=a.min_tile_length_bp,
                    max_tile_bp=a.max_tile_size_bp,
                )
                if expanded is None:
                    continue
                if block_constraints_ok(expanded, idxs, a.max_tile_size_bp, a.min_tile_length_bp, a.min_overlap_bp):
                    add_plan('local_boundary_shift', f'terminal_right_outward_{last_tid}_{delta}bp', expanded)
    if trigger in {'local_retile','long_tile_rescue'}:
        current_n = len(tile_ids)
        n_options = sorted({current_n, max(1, current_n - 1), current_n + 1 if current_n > 1 else current_n})
        if trigger == 'long_tile_rescue':
            n_options = sorted({max(1, current_n - 2), max(1, current_n - 1), current_n})
        overlap_options = sorted({a.target_overlap_size_bp, max(a.min_overlap_bp, a.target_overlap_size_bp + 500)})
        for n_tiles in n_options:
            if n_tiles > len(tile_ids):
                continue
            for overlap_bp in overlap_options:
                new_tile_ids = [f"{block_id}_retile_{i+1}" for i in range(n_tiles)]
                rows = retile_interval(new_tile_ids, chrom, block_start, block_end, n_tiles, overlap_bp)
                if not rows:
                    continue
                full = replace_block(tiles, idxs, rows)
                check_idxs = [full.index[full['tile_id'] == tid].tolist()[0] for tid in [r['tile_id'] for r in rows]]
                if block_constraints_ok(full, sorted(check_idxs), a.max_tile_size_bp, a.min_tile_length_bp, a.min_overlap_bp):
                    add_plan('local_retile' if trigger != 'long_tile_rescue' else 'long_tile_rescue', f'n_tiles={n_tiles};overlap_bp={overlap_bp}', full)
    plans_df = pd.DataFrame(plans)
    cand_df = pd.concat(plan_rows, ignore_index=True) if plan_rows else pd.DataFrame()
    if not plans_df.empty and not cand_df.empty:
        # Deduplicate by full-plan geometry so repeated no-op candidates do not clutter search history.
        sigs = []
        changed_ids = []
        for pid in plans_df['candidate_plan_id']:
            sub = cand_df[cand_df['candidate_plan_id'] == pid].copy()
            sig = '|'.join(f"{r.tile_id}:{int(r.tile_start_1based)}-{int(r.tile_end_1based)}" for r in sub.sort_values(['tile_start_1based','tile_end_1based']).itertuples())
            sigs.append(sig)
            base = tiles[['tile_id','tile_start_1based','tile_end_1based']].copy()
            merged = sub[['tile_id','tile_start_1based','tile_end_1based']].copy().rename(columns={'tile_start_1based':'s','tile_end_1based':'e'})
            comp = base.merge(merged, on='tile_id', how='outer')
            changed = comp[(comp['tile_start_1based'] != comp['s']) | (comp['tile_end_1based'] != comp['e']) | comp['tile_start_1based'].isna() | comp['s'].isna()]['tile_id'].astype(str).tolist()
            changed_ids.append(','.join(changed))
        plans_df['geometry_signature'] = sigs
        plans_df['changed_tile_ids'] = changed_ids
        plans_df['n_changed_tiles'] = plans_df['changed_tile_ids'].apply(lambda x: 0 if not x else len([p for p in str(x).split(',') if p]))
        plans_df = plans_df.drop_duplicates(subset=['geometry_signature'], keep='first').head(a.max_candidate_plans)
    keep = set(plans_df['candidate_plan_id'])
    if not cand_df.empty:
        cand_df = cand_df[cand_df['candidate_plan_id'].isin(keep)].copy()
        # Keep legacy and redesigned coordinate columns synchronized so downstream code
        # cannot accidentally fall back to stale original coordinates.
        cand_df['start_1based'] = cand_df['tile_start_1based'].astype(int)
        cand_df['end_1based'] = cand_df['tile_end_1based'].astype(int)
        cand_df['tile_length_bp'] = cand_df['tile_end_1based'] - cand_df['tile_start_1based'] + 1
    plans_df.to_csv(outdir / 'candidate_plans_summary.tsv', sep='\t', index=False)
    cand_df.to_csv(outdir / 'tile_plan_candidates.tsv', sep='\t', index=False)
    legacy = cand_df[cand_df['candidate_plan_id'] == plans_df.iloc[0]['candidate_plan_id']].copy() if not plans_df.empty else pd.DataFrame()
    legacy.to_csv(outdir / 'tiles_revised.tsv', sep='\t', index=False)
    (outdir / 'tiles_revised_summary.json').write_text(json.dumps({'n_candidate_plans': int(len(plans_df)), 'selected_block_id': block_id, 'trigger': trigger}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
