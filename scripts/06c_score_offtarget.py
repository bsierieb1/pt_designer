#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pysam

SPCAS9_PAM_RE = re.compile(r'^[ACGT]GG$', re.IGNORECASE)
NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run FlashFry for candidate SpCas9 guides and emit a guide-level off-target score.'
    )
    parser.add_argument('--candidates-ontarget-tsv', required=True, help='Path to candidates_ontarget.tsv from 06b.')
    parser.add_argument('--locus-json', required=True, help='Path to locus.json from 01_parse_locus.py.')
    parser.add_argument('--fasta', required=True, help='Path to hg38 FASTA for within-locus exact-hit checks.')
    parser.add_argument('--flashfry-jar', required=True, help='Path to FlashFry-assembly-*.jar')
    parser.add_argument('--database', required=True, help='Path prefix to FlashFry database built with enzyme spcas9ngg.')
    parser.add_argument('--outdir', required=True, help='Output directory.')
    parser.add_argument('--java-bin', default='java', help='Java executable. Default: java')
    parser.add_argument('--java-xmx', default='6g', help='Heap passed to Java as -Xmx. Default: 6g')
    parser.add_argument('--max-mismatches', type=int, default=3, help='Maximum mismatches to consider. Default: 3')
    parser.add_argument(
        '--scoring-metrics',
        default='dangerous,hsu2013,doench2016cfd',
        help='Comma-separated FlashFry scoring metrics.'
    )
    parser.add_argument('--tmp-dir', default=None, help='Optional tmp dir for FlashFry scratch files.')
    return parser.parse_args()


def reverse_complement(seq: str) -> str:
    comp = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
    return seq.translate(comp)[::-1].upper()


def load_guides(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    required = [
        'guide_seq', 'chrom', 'strand',
        'protospacer_start_1based', 'protospacer_end_1based',
        'pam_start_1based', 'pam_end_1based',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if 'pam_seq' not in df.columns:
        df['pam_seq'] = 'AGG'

    df = df.drop_duplicates(subset=['guide_seq']).copy()
    df['guide_seq'] = df['guide_seq'].astype(str).str.upper()
    df['pam_seq'] = df['pam_seq'].astype(str).str.upper()
    df['pam_seq'] = df['pam_seq'].map(lambda p: p if SPCAS9_PAM_RE.match(p) else 'AGG')
    df['target_with_pam'] = df['guide_seq'] + df['pam_seq']
    for col in ['protospacer_start_1based', 'protospacer_end_1based', 'pam_start_1based', 'pam_end_1based']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(int)
    return df[
        [
            'guide_seq', 'pam_seq', 'target_with_pam', 'chrom', 'strand',
            'protospacer_start_1based', 'protospacer_end_1based',
            'pam_start_1based', 'pam_end_1based',
        ]
    ]


def load_locus(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text())
    for key in ['chrom', 'start_1based', 'end_1based']:
        if key not in data:
            raise ValueError(f'locus.json missing required key: {key}')
    return data


def write_fasta(df: pd.DataFrame, path: Path) -> None:
    with path.open('w') as fh:
        for i, row in enumerate(df.itertuples(index=False), start=1):
            fh.write(f'>guide_{i}|{row.guide_seq}|{row.pam_seq}\n{row.target_with_pam}\n')


def run_cmd(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s == '' or s.lower() in {'nan', 'none', 'null'}:
        return None
    m = NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _safe_int(value: object, default: int = 0) -> int:
    f = _safe_float(value)
    if f is None or math.isnan(f):
        return default
    return int(round(f))


def _first_present(row: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    lowered = {str(k).lower(): k for k in row.keys()}
    for c in candidates:
        key = lowered.get(c.lower())
        if key is not None:
            return row.get(key)
    return None


def _normalize_score_0_100(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if 0.0 <= value <= 1.0:
        return round(value * 100.0, 4)
    return round(max(0.0, min(100.0, value)), 4)


def _parse_mm_counts(row: Dict[str, str]) -> Dict[int, int]:
    mm_counts: Dict[int, int] = {i: 0 for i in range(5)}
    separate = False
    for mm in range(5):
        value = _first_present(
            row,
            [
                f'{mm}_mismatch',
                f'mismatch_{mm}',
                f'offtargets_{mm}_mismatch',
                f'count_{mm}_mismatch',
            ],
        )
        if value is not None:
            mm_counts[mm] = _safe_int(value)
            separate = True
    if separate:
        return mm_counts

    combined = _first_present(row, ['0-1-2-3-4_mismatch', 'mismatchCounts', 'mismatch_counts'])
    if combined is None:
        return mm_counts
    nums = [int(x) for x in re.findall(r'-?\d+', str(combined))]
    for i, n in enumerate(nums[:5]):
        mm_counts[i] = n
    return mm_counts


def _classify_offtarget(score: float) -> str:
    if score >= 80:
        return 'high'
    if score >= 50:
        return 'medium'
    return 'low'


def _choose_offtarget_score(row: Dict[str, str]) -> Tuple[float, str]:
    cfd = _normalize_score_0_100(
        _safe_float(_first_present(row, ['DoenchCFD_specificityscore', 'doench2016cfd_specificityscore']))
    )
    if cfd is not None:
        return cfd, 'flashfry_doench2016cfd_specificity'

    hsu = _normalize_score_0_100(_safe_float(_first_present(row, ['hsu2013'])))
    if hsu is not None:
        return hsu, 'flashfry_hsu2013'

    minot = _normalize_score_0_100(_safe_float(_first_present(row, ['minot'])))
    if minot is not None:
        return minot, 'flashfry_minot'

    dangerous = _safe_int(_first_present(row, ['dangerous_in_genome', 'in_genome', 'IN_GENOME']), default=1)
    score = 100.0 if dangerous <= 1 else 0.0
    return score, 'flashfry_dangerous_fallback'


def parse_flashfry_scored(scored_path: Path, max_mismatches: int) -> pd.DataFrame:
    columns = [
        'guide_seq', 'pam_seq', 'target_with_pam',
        'n_exact_total', 'n_extra_exact', 'n_mm1', 'n_mm2', 'n_mm3', 'n_mm4',
        'offtarget_score', 'offtarget_method', 'offtarget_class',
        'dangerous_in_genome', 'hsu2013', 'doench2016cfd_specificityscore', 'doench2016cfd_maxot',
        'minot', 'basesdifftoclosesthit', 'closesthitcount', 'flashfry_overflow', 'flashfry_otcount',
    ]
    if not scored_path.exists() or scored_path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, object]] = []
    with scored_path.open('r', newline='') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        for row in reader:
            target = (_first_present(row, ['target', 'sequence', 'guideSequence']) or '').strip().upper()
            if not target:
                continue
            guide_seq = target[:20]
            pam_seq = target[20:23] if len(target) >= 23 else ''
            mm_counts = _parse_mm_counts(row)
            exact = _safe_int(_first_present(row, ['dangerous_in_genome', 'in_genome', 'IN_GENOME']), default=mm_counts.get(0, 0))
            if exact > 0:
                mm_counts[0] = exact
            if max_mismatches < 4:
                mm_counts[4] = 0

            score, method = _choose_offtarget_score(row)
            rows.append(
                {
                    'guide_seq': guide_seq,
                    'pam_seq': pam_seq,
                    'target_with_pam': target,
                    'n_exact_total': int(mm_counts.get(0, 0)),
                    'n_extra_exact': max(0, int(mm_counts.get(0, 0)) - 1),
                    'n_mm1': int(mm_counts.get(1, 0)),
                    'n_mm2': int(mm_counts.get(2, 0)),
                    'n_mm3': int(mm_counts.get(3, 0)),
                    'n_mm4': int(mm_counts.get(4, 0)),
                    'offtarget_score': score,
                    'offtarget_method': method,
                    'offtarget_class': _classify_offtarget(score),
                    'dangerous_in_genome': exact,
                    'hsu2013': _normalize_score_0_100(_safe_float(_first_present(row, ['hsu2013']))),
                    'doench2016cfd_specificityscore': _normalize_score_0_100(
                        _safe_float(_first_present(row, ['DoenchCFD_specificityscore', 'doench2016cfd_specificityscore']))
                    ),
                    'doench2016cfd_maxot': _normalize_score_0_100(
                        _safe_float(_first_present(row, ['DoenchCFD_maxOT', 'doench2016cfd_maxot']))
                    ),
                    'minot': _normalize_score_0_100(_safe_float(_first_present(row, ['minot']))),
                    'basesdifftoclosesthit': _safe_int(_first_present(row, ['basesDiffToClosestHit', 'basesdifftoclosesthit'])),
                    'closesthitcount': _safe_int(_first_present(row, ['closestHitCount', 'closesthitcount'])),
                    'flashfry_overflow': (_first_present(row, ['overflow']) or '').strip(),
                    'flashfry_otcount': _safe_int(_first_present(row, ['otCount', 'otcount'])),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=columns)

    df = df.sort_values(['guide_seq', 'offtarget_score'], ascending=[True, False])
    df = df.drop_duplicates(subset=['guide_seq'], keep='first').reset_index(drop=True)
    return df[columns]


def _find_all(seq: str, query: str) -> List[int]:
    starts: List[int] = []
    start = 0
    while query:
        idx = seq.find(query, start)
        if idx == -1:
            break
        starts.append(idx)
        start = idx + 1
    return starts


def annotate_exact_in_locus_offtargets(guides_df: pd.DataFrame, locus: Dict[str, object], fasta_path: Path) -> pd.DataFrame:
    locus_chrom = str(locus['chrom'])
    locus_start_1based = int(locus['start_1based'])
    locus_end_1based = int(locus['end_1based'])

    fasta = pysam.FastaFile(str(fasta_path))
    try:
        locus_seq = fasta.fetch(locus_chrom, locus_start_1based - 1, locus_end_1based).upper()
    finally:
        fasta.close()

    rows: List[Dict[str, object]] = []
    for row in guides_df.itertuples(index=False):
        target_plus = str(row.target_with_pam).upper()
        target_minus = reverse_complement(target_plus)
        intended_chrom = str(row.chrom)
        intended_strand = str(row.strand)
        if intended_strand == '+':
            intended_start = int(row.protospacer_start_1based)
            intended_end = int(row.pam_end_1based)
        else:
            intended_start = int(row.pam_start_1based)
            intended_end = int(row.protospacer_end_1based)

        sites: List[Dict[str, object]] = []
        for zero_based_start in _find_all(locus_seq, target_plus):
            site_start = locus_start_1based + zero_based_start
            site_end = site_start + len(target_plus) - 1
            sites.append({'chrom': locus_chrom, 'start_1based': site_start, 'end_1based': site_end, 'strand': '+'})
        if target_minus != target_plus:
            for zero_based_start in _find_all(locus_seq, target_minus):
                site_start = locus_start_1based + zero_based_start
                site_end = site_start + len(target_minus) - 1
                sites.append({'chrom': locus_chrom, 'start_1based': site_start, 'end_1based': site_end, 'strand': '-'})

        extra_sites = [
            site for site in sites
            if not (
                site['chrom'] == intended_chrom
                and site['start_1based'] == intended_start
                and site['end_1based'] == intended_end
                and site['strand'] == intended_strand
            )
        ]
        rows.append(
            {
                'guide_seq': row.guide_seq,
                'n_exact_in_locus_offtargets': int(len(extra_sites)),
                'has_exact_in_locus_offtarget': bool(extra_sites),
                'exact_in_locus_offtarget_sites': ';'.join(
                    f"{site['chrom']}:{site['start_1based']}-{site['end_1based']}({site['strand']})"
                    for site in sorted(extra_sites, key=lambda x: (x['chrom'], x['start_1based'], x['end_1based'], x['strand']))
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    guides_df = load_guides(Path(args.candidates_ontarget_tsv))
    exact_in_locus_df = annotate_exact_in_locus_offtargets(
        guides_df=guides_df,
        locus=load_locus(Path(args.locus_json)),
        fasta_path=Path(args.fasta),
    )

    fasta_path = outdir / 'flashfry_guides.fasta'
    discover_path = outdir / 'flashfry_discover.tsv'
    scored_path = outdir / 'flashfry_scored.tsv'
    normalized_path = outdir / 'offtargets_raw.tsv'
    score_path = outdir / 'offtarget_scores.tsv'

    write_fasta(guides_df, fasta_path)

    discover_cmd = [
        args.java_bin, f'-Xmx{args.java_xmx}', '-jar', args.flashfry_jar,
        'discover', '--database', args.database, '--fasta', str(fasta_path),
        '--output', str(discover_path), '--positionOutput', '--maxMismatch', str(args.max_mismatches)
    ]
    if args.tmp_dir:
        discover_cmd.extend(['--tmpLocation', str(Path(args.tmp_dir))])
    run_cmd(discover_cmd)

    score_cmd = [
        args.java_bin, f'-Xmx{args.java_xmx}', '-jar', args.flashfry_jar,
        'score', '--input', str(discover_path), '--output', str(scored_path),
        '--scoringMetrics', args.scoring_metrics, '--database', args.database,
        '--maxMismatch', str(args.max_mismatches)
    ]
    run_cmd(score_cmd)

    flashfry_df = parse_flashfry_scored(scored_path, max_mismatches=args.max_mismatches)
    guide_meta = guides_df[['guide_seq', 'pam_seq', 'target_with_pam']].drop_duplicates(subset=['guide_seq'])
    result_df = guide_meta.merge(flashfry_df, on=['guide_seq', 'pam_seq', 'target_with_pam'], how='left')
    result_df = result_df.merge(exact_in_locus_df, on='guide_seq', how='left')
    result_df['n_exact_in_locus_offtargets'] = pd.to_numeric(
        result_df.get('n_exact_in_locus_offtargets', 0), errors='coerce'
    ).fillna(0).astype(int)
    result_df['has_exact_in_locus_offtarget'] = result_df.get('has_exact_in_locus_offtarget', False).fillna(False).astype(bool)
    result_df['exact_in_locus_offtarget_sites'] = result_df.get('exact_in_locus_offtarget_sites', '').fillna('')
    for col in ['n_exact_total', 'n_extra_exact', 'n_mm1', 'n_mm2', 'n_mm3', 'n_mm4', 'dangerous_in_genome', 'basesdifftoclosesthit', 'closesthitcount', 'flashfry_otcount']:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0).astype(int)
    for col in ['offtarget_score', 'hsu2013', 'doench2016cfd_specificityscore', 'doench2016cfd_maxot', 'minot']:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
    if 'offtarget_score' in result_df.columns:
        result_df['offtarget_score'] = result_df['offtarget_score'].fillna(0.0)
    else:
        result_df['offtarget_score'] = 0.0
    for col, default in [('offtarget_method', 'flashfry_missing_default'), ('offtarget_class', 'unknown'), ('flashfry_overflow', '')]:
        if col in result_df.columns:
            result_df[col] = result_df[col].fillna(default)
        else:
            result_df[col] = default

    result_df.to_csv(normalized_path, sep='\t', index=False)
    result_df.to_csv(score_path, sep='\t', index=False)

    summary = pd.DataFrame([
        {'metric': 'n_guides_input', 'value': int(guides_df.shape[0])},
        {'metric': 'n_guides_scored', 'value': int(result_df.shape[0])},
        {'metric': 'mean_offtarget_score', 'value': round(float(result_df['offtarget_score'].mean()), 4) if not result_df.empty else 0.0},
        {'metric': 'n_high', 'value': int((result_df['offtarget_class'] == 'high').sum()) if not result_df.empty else 0},
        {'metric': 'n_medium', 'value': int((result_df['offtarget_class'] == 'medium').sum()) if not result_df.empty else 0},
        {'metric': 'n_low', 'value': int((result_df['offtarget_class'] == 'low').sum()) if not result_df.empty else 0},
        {'metric': 'n_with_exact_in_locus_offtarget', 'value': int(result_df['has_exact_in_locus_offtarget'].sum()) if not result_df.empty else 0},
    ])
    summary.to_csv(outdir / 'flashfry_summary.tsv', sep='\t', index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
