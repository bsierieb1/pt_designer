#!/usr/bin/env python3
import argparse
import base64
import html
import io
import json
import shutil
from pathlib import Path

import pandas as pd


def maybe_plot_hist(series, title):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return ""
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=20)
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def maybe_plot_multi_hist(series_list, title):
    vals = []
    for s in series_list:
        vals.extend(pd.to_numeric(s, errors="coerce").dropna().tolist())
    return maybe_plot_hist(pd.Series(vals), title) if vals else ""


def table_html(df: pd.DataFrame, max_rows=20) -> str:
    if df is None or df.empty:
        return "<p><em>None</em></p>"
    return df.head(max_rows).to_html(index=False, escape=False)


def load_df(path, sep="\t"):
    p = Path(path) if path else None
    return pd.read_csv(p, sep=sep) if p and p.exists() else pd.DataFrame()


def grouped_tile_svg(df: pd.DataFrame, max_candidates: int = 120) -> str:
    if df is None or df.empty:
        return "<p><em>No tile history available.</em></p>"

    work = df.copy()
    start_col = next((c for c in ["tile_start_1based", "start_1based"] if c in work.columns), None)
    end_col = next((c for c in ["tile_end_1based", "end_1based"] if c in work.columns), None)
    if not start_col or not end_col:
        return "<p><em>No drawable tile coordinates found.</em></p>"

    work[start_col] = pd.to_numeric(work[start_col], errors="coerce")
    work[end_col] = pd.to_numeric(work[end_col], errors="coerce")
    work = work.dropna(subset=[start_col, end_col]).copy()
    if work.empty:
        return "<p><em>No drawable tile coordinates found.</em></p>"

    if "round" not in work.columns:
        work["round"] = "?"
    if "candidate_plan_id" not in work.columns:
        work["candidate_plan_id"] = "candidate"
    if "strategy" not in work.columns:
        work["strategy"] = ""
    if "strategy_details" not in work.columns:
        work["strategy_details"] = ""

    candidate_cols = ["round", "candidate_plan_id", "strategy", "strategy_details"]
    row_df = (
        work.groupby(candidate_cols, dropna=False)
        .agg(
            xmin=(start_col, "min"),
            xmax=(end_col, "max"),
            n_tiles=("tile_id", "count"),
        )
        .reset_index()
        .sort_values(["round", "candidate_plan_id"], kind="stable")
    )

    total_candidates = len(row_df)
    if total_candidates > max_candidates:
        row_df = row_df.head(max_candidates).copy()
        keep_keys = set(tuple(x) for x in row_df[candidate_cols].itertuples(index=False, name=None))
        work = work[work[candidate_cols].apply(tuple, axis=1).isin(keep_keys)].copy()

    xmin = int(work[start_col].min())
    xmax = int(work[end_col].max())
    span = max(1, xmax - xmin + 1)

    row_h = 26
    width = 1350
    left_pad = 340
    right_pad = 24
    height = 34 + row_h * len(row_df)

    pieces = [
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>",
        f"<line x1='{left_pad}' y1='18' x2='{width-right_pad}' y2='18' stroke='#999' stroke-width='1'/>",
    ]

    plot_w = width - left_pad - right_pad

    for i, (_, crow) in enumerate(row_df.iterrows()):
        y = 30 + i * row_h
        label = f"r{crow['round']} {crow['candidate_plan_id']}  ({int(crow['n_tiles'])} tiles)  {crow.get('strategy', '')}  {crow.get('strategy_details', '')}".strip()
        pieces.append(f"<text x='5' y='{y+12}' font-size='11' fill='#333'>{html.escape(label)}</text>")

        mask = (
            (work["round"] == crow["round"]) &
            (work["candidate_plan_id"] == crow["candidate_plan_id"]) &
            (work["strategy"] == crow["strategy"]) &
            (work["strategy_details"] == crow["strategy_details"])
        )
        sub = work.loc[mask].sort_values([start_col, end_col], kind="stable")
        for _, row in sub.iterrows():
            x1 = left_pad + (int(row[start_col]) - xmin) / span * plot_w
            x2 = left_pad + (int(row[end_col]) - xmin) / span * plot_w
            pieces.append(f"<rect x='{x1:.1f}' y='{y}' width='{max(2.0, x2-x1):.1f}' height='12' fill='#87b5ff' stroke='#3b6fb6'/>")
            tile_label = html.escape(str(row.get('tile_id', '')))
            pieces.append(f"<text x='{x1+2:.1f}' y='{y+10}' font-size='9' fill='#103b73'>{tile_label}</text>")

    pieces.append("</svg>")
    note = ""
    if total_candidates > max_candidates:
        note = f"<p class='note'>Showing first {max_candidates} candidate plans in the geometry snapshot.</p>"
    return "".join(pieces) + note


def render_simple_html(context: dict) -> str:
    style = """
    body { font-family: Arial, sans-serif; margin: 24px; color: #222; }
    h1, h2, h3 { margin: 0.4em 0; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 12px; }
    th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: left; vertical-align: top; }
    th { background: #f3f3f3; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    .box { background: #fafafa; border: 1px solid #ddd; padding: 12px 14px; border-radius: 8px; }
    .note { color: #666; font-size: 12px; }
    a { color: #1f5fbf; }
    """
    return f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<title>PacBio PureTarget design summary</title>
<style>{style}</style>
</head>
<body>
<h1>PacBio PureTarget design summary</h1>
<div class='box'>
<p><strong>Locus:</strong> {context.get('locus_label', 'NA')}<br>
<strong>Final tiles:</strong> {context.get('n_tiles', 0)}{context.get('pool_counts_html', '')}</p>
{context.get('design_search_link_html', '')}
</div>

{context.get('pool_summary_section_html', '')}

<h2>Final selected pairs</h2>
{context.get('tiles_table_html', '<p><em>None</em></p>')}

<h2>Guide order table preview</h2>
{context.get('guides_table_html', '<p><em>None</em></p>')}

<h2>Problematic tiles table</h2>
{context.get('problems_table_html', '<p><em>None</em></p>')}

<h2>Guide score distributions</h2>
<div class='grid'>
  <div>{'<img src="data:image/png;base64,' + context['ontarget_hist'] + '">' if context.get('ontarget_hist') else '<p><em>No on-target chart</em></p>'}</div>
  <div>{'<img src="data:image/png;base64,' + context['offtarget_hist'] + '">' if context.get('offtarget_hist') else '<p><em>No off-target chart</em></p>'}</div>
</div>

<h2>Design search history</h2>
<div class='box'>
<p><strong>Rounds completed:</strong> {context.get('search_rounds', 0)}<br>
<strong>Candidates evaluated:</strong> {context.get('search_candidates', 0)}<br>
<strong>Accepted redesign rounds:</strong> {context.get('search_accepted_rounds', 0)}</p>
{context.get('accepted_path_html', '<p><em>No redesign history available.</em></p>')}
</div>

<h3>Round summary</h3>
{context.get('round_summary_html', '<p><em>No round summary available.</em></p>')}

<h3>Candidate evaluations</h3>
{context.get('candidate_eval_html', '<p><em>No candidate evaluation table available.</em></p>')}


<p class='note'>This report is a lightweight HTML summary generated from final design tables. Charts are optional and appear only if matplotlib is available. Use the dedicated design-search page for candidate geometry and redesign history.</p>
</body>
</html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description='Generate human-readable design report.')
    ap.add_argument('--selected-pairs-final-tsv', required=True)
    ap.add_argument('--tile-pool-assignment-tsv', required=False)
    ap.add_argument('--guides-for-ordering-csv', required=False)
    ap.add_argument('--tiles-problematic-tsv', required=False)
    ap.add_argument('--locus-json', required=False)
    ap.add_argument('--design-search-dir', required=False, help='Directory containing Step 9 search-history outputs')
    ap.add_argument('--unpooled', action='store_true', help='Render report without odd/even pool labels.')
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(args.selected_pairs_final_tsv, sep='\t')
    use_pools = bool(args.tile_pool_assignment_tsv) and not args.unpooled
    if use_pools:
        pools = pd.read_csv(args.tile_pool_assignment_tsv, sep='\t')
        df = selected.merge(pools[['tile_id', 'pool', 'pool_label']], on='tile_id', how='left')
    else:
        df = selected.copy()

    guides = pd.read_csv(args.guides_for_ordering_csv) if args.guides_for_ordering_csv and Path(args.guides_for_ordering_csv).exists() else pd.DataFrame()
    problems = pd.read_csv(args.tiles_problematic_tsv, sep='\t') if args.tiles_problematic_tsv and Path(args.tiles_problematic_tsv).exists() else pd.DataFrame()

    locus_label = 'NA'
    if args.locus_json and Path(args.locus_json).exists():
        with open(args.locus_json, 'r') as fh:
            locus = json.load(fh)
        locus_label = locus.get('locus', locus.get('region', 'NA'))

    if use_pools:
        pool_table = (
            df.groupby('pool', dropna=False)
            .agg(n_tiles=('tile_id', 'count'), mean_pair_score=('pair_score', 'mean'))
            .reset_index()
        )
        pool_counts_html = (
            f"<br><strong>Odd pool tiles:</strong> {int((df['pool'] == 'odd').sum())}"
            f"<br><strong>Even pool tiles:</strong> {int((df['pool'] == 'even').sum())}"
        )
        pool_summary_section_html = f"<h2>Pool summary</h2>{table_html(pool_table, max_rows=10)}"
    else:
        pool_table = pd.DataFrame()
        pool_counts_html = ''
        pool_summary_section_html = ''

    ontarget_hist = maybe_plot_multi_hist(
        [df[c] for c in ['left_ontarget_score', 'right_ontarget_score'] if c in df.columns],
        'Guide on-target score distribution',
    )
    if not ontarget_hist:
        for c in ['pair_mean_ontarget_score', 'pair_ontarget_score', 'ontarget_score']:
            if c in df.columns:
                ontarget_hist = maybe_plot_hist(df[c], 'On-target score distribution')
                if ontarget_hist:
                    break

    offtarget_hist = maybe_plot_multi_hist(
        [df[c] for c in ['left_offtarget_score', 'right_offtarget_score'] if c in df.columns],
        'Guide off-target score distribution',
    )
    if not offtarget_hist:
        for c in ['pair_mean_offtarget_score', 'pair_offtarget_score', 'offtarget_score']:
            if c in df.columns:
                offtarget_hist = maybe_plot_hist(df[c], 'Off-target score distribution')
                if offtarget_hist:
                    break

    round_summary = pd.DataFrame()
    candidate_eval = pd.DataFrame()
    accepted_path = pd.DataFrame()
    candidate_tiles = pd.DataFrame()
    design_search_link_html = ''
    design_search_dir = Path(args.design_search_dir) if args.design_search_dir else None
    if design_search_dir and design_search_dir.exists():
        round_summary = load_df(design_search_dir / 'round_summary.tsv')
        candidate_eval = load_df(design_search_dir / 'candidate_evaluation_all_rounds.tsv')
        accepted_path = load_df(design_search_dir / 'accepted_design_path.tsv')
        candidate_tiles = load_df(design_search_dir / 'tile_plan_candidates_all_rounds.tsv')
        copied = outdir / 'design_search'
        copied.mkdir(parents=True, exist_ok=True)
        for name in [
            'round_summary.tsv',
            'candidate_evaluation_all_rounds.tsv',
            'tile_plan_candidates_all_rounds.tsv',
            'tile_plan_candidates_all_rounds.bed',
            'accepted_design_path.tsv',
            'final_tile_plan.bed',
            'design_search_history.html',
        ]:
            src = design_search_dir / name
            if src.exists():
                shutil.copy2(src, copied / name)
        hist_path = copied / 'design_search_history.html'
        if hist_path.exists():
            design_search_link_html = "<p><strong>Search report:</strong> <a href='design_search/design_search_history.html'>Open full design-search history</a></p>"

    context = {
        'locus_label': locus_label,
        'n_tiles': int(len(df)),
        'pool_counts_html': pool_counts_html,
        'pool_summary_section_html': pool_summary_section_html,
        'tiles_table_html': table_html(df, max_rows=30),
        'guides_table_html': table_html(guides, max_rows=40),
        'problems_table_html': table_html(problems, max_rows=30),
        'ontarget_hist': ontarget_hist,
        'offtarget_hist': offtarget_hist,
        'search_rounds': int(round_summary['round'].nunique()) if not round_summary.empty and 'round' in round_summary.columns else 0,
        'search_candidates': int(len(candidate_eval)),
        'search_accepted_rounds': int(len(accepted_path)),
        'accepted_path_html': table_html(accepted_path, max_rows=20),
        'round_summary_html': table_html(round_summary, max_rows=20),
        'candidate_eval_html': table_html(candidate_eval, max_rows=100),
        'candidate_tile_svg': grouped_tile_svg(candidate_tiles, max_candidates=120),
        'design_search_link_html': design_search_link_html,
    }

    html_text = render_simple_html(context)
    html_path = outdir / 'panel_summary.html'
    html_path.write_text(html_text, encoding='utf-8')

    print(f'[DONE] Wrote {html_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
