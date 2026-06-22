#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def first_existing(df: pd.DataFrame, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required columns; tried: {candidates}")
    return None


def collect_warnings(row) -> str:
    warnings = []
    for col, label in [
        ("left_overlaps_common_snp", "left_snp"),
        ("right_overlaps_common_snp", "right_snp"),
        ("left_basic_sequence_warning", "left_seq_warning"),
        ("right_basic_sequence_warning", "right_seq_warning"),
        ("left_has_homopolymer", "left_homopolymer"),
        ("right_has_homopolymer", "right_homopolymer"),
        ("left_is_low_complexity", "left_low_complexity"),
        ("right_is_low_complexity", "right_low_complexity"),
        ("left_guide_too_close_to_target", "left_near_target"),
        ("right_guide_too_close_to_target", "right_near_target"),
        ("pair_has_guide_too_close_to_target", "guide_near_target"),
    ]:
        if col in row.index and bool(row[col]):
            warnings.append(label)

    for col in ["problem_reasons", "rescue_strategy", "rescue_note"]:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            warnings.append(f"{col}={row[col]}")
    return ";".join(warnings)


def write_fasta(df: pd.DataFrame, path: Path) -> None:
    with path.open("w") as handle:
        for _, row in df.iterrows():
            seq = "" if pd.isna(row.get("guide_seq", "")) else str(row.get("guide_seq", "")).strip()
            if not seq:
                continue
            header_parts = [
                f"tile_{row.get('tile_id', '')}",
                f"{row.get('guide_side', '')}",
            ]
            handle.write(">" + "_".join(header_parts) + "\n")
            handle.write(seq + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate final guide ordering table.")
    ap.add_argument("--selected-pairs-final-tsv", required=True)
    ap.add_argument("--tile-pool-assignment-tsv", required=False)
    ap.add_argument("--unpooled", action="store_true", help="Write one combined guide table/FASTA without pool labels.")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.selected_pairs_final_tsv, sep="\t")
    use_pools = bool(args.tile_pool_assignment_tsv) and not args.unpooled
    if use_pools:
        pools = pd.read_csv(args.tile_pool_assignment_tsv, sep="\t")
        df = pairs.merge(pools[["tile_id", "pool", "pool_label"]], on="tile_id", how="left")
    else:
        df = pairs.copy()

    left_guide = first_existing(df, ["left_guide_seq", "left_guide", "guide_seq_left"], required=False)
    right_guide = first_existing(df, ["right_guide_seq", "right_guide", "guide_seq_right"], required=False)
    left_strand = first_existing(df, ["left_strand", "strand_left"], required=False)
    right_strand = first_existing(df, ["right_strand", "strand_right"], required=False)
    left_coord = first_existing(df, ["left_cut_site_1based", "left_cut_site", "cut_site_left_1based"], required=False)
    right_coord = first_existing(df, ["right_cut_site_1based", "right_cut_site", "cut_site_right_1based"], required=False)
    left_on = first_existing(df, ["left_ontarget_score", "ontarget_score_left"], required=False)
    right_on = first_existing(df, ["right_ontarget_score", "ontarget_score_right"], required=False)
    left_off = first_existing(df, ["left_offtarget_score", "offtarget_score_left"], required=False)
    right_off = first_existing(df, ["right_offtarget_score", "offtarget_score_right"], required=False)

    rows = []
    for _, row in df.iterrows():
        if use_pools:
            common = {
                "tile_id": row["tile_id"],
                "pool": row.get("pool", ""),
                "pair_score": row.get("pair_score", ""),
                "warnings": collect_warnings(row),
            }
        else:
            common = {
                "tile_id": row["tile_id"],
                "pair_score": row.get("pair_score", ""),
                "warnings": collect_warnings(row),
            }
        if left_guide:
            rows.append({
                **common,
                "guide_side": "left",
                "guide_seq": row[left_guide],
                "strand": row[left_strand] if left_strand else "",
                "coordinates": row[left_coord] if left_coord else "",
                "on_target_score": row[left_on] if left_on else "",
                "off_target_score": row[left_off] if left_off else "",
            })
        if right_guide:
            rows.append({
                **common,
                "guide_side": "right",
                "guide_seq": row[right_guide],
                "strand": row[right_strand] if right_strand else "",
                "coordinates": row[right_coord] if right_coord else "",
                "on_target_score": row[right_on] if right_on else "",
                "off_target_score": row[right_off] if right_off else "",
            })

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "guides_for_ordering.csv", index=False)
    if use_pools:
        write_fasta(out[out["pool"] == "odd"], outdir / "guides_odd.fasta")
        write_fasta(out[out["pool"] == "even"], outdir / "guides_even.fasta")
    else:
        write_fasta(out, outdir / "guides.fasta")

    print(f"[DONE] Wrote {outdir / 'guides_for_ordering.csv'}")
    if use_pools:
        print(f"[DONE] Wrote {outdir / 'guides_odd.fasta'}")
        print(f"[DONE] Wrote {outdir / 'guides_even.fasta'}")
    else:
        print(f"[DONE] Wrote {outdir / 'guides.fasta'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
