#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate oriented guide candidates with common-SNP overlap and basic sequence QC."
    )
    parser.add_argument(
        "--candidates-oriented-tsv",
        required=True,
        help="Path to candidates_oriented.tsv from Step 5.",
    )
    parser.add_argument(
        "--common-snps-bed",
        required=False,
        default=None,
        help="Optional path to common_snps.bed from Step 1.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--avoid-targets-tsv",
        required=False,
        default=None,
        help="Optional normalized hotspot targets TSV. Guides closer than --min-distance-to-target-bp are flagged.",
    )
    parser.add_argument(
        "--min-distance-to-target-bp",
        type=int,
        default=50,
        help="Minimum required distance between guide+PAM footprint and any hotspot target. Default: 50",
    )
    parser.add_argument(
        "--homopolymer-threshold",
        type=int,
        default=4,
        help="Flag guides containing homopolymers of this length or greater. Default: 4",
    )
    parser.add_argument(
        "--low-complexity-entropy-threshold",
        type=float,
        default=1.5,
        help="Flag guides with Shannon entropy below this threshold. Default: 1.5",
    )
    return parser.parse_args()


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    required = [
        "tile_id",
        "chrom",
        "boundary_type",
        "strand",
        "guide_seq",
        "protospacer_start_0based",
        "protospacer_end_0based",
        "protospacer_start_1based",
        "protospacer_end_1based",
        "pam_start_1based",
        "pam_end_1based",
        "cut_site_1based",
        "distance_cut_to_boundary_bp",
        "distance_abs_to_boundary_bp",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"candidates_oriented.tsv missing required columns: {missing}")

    int_cols = [
        "tile_id",
        "protospacer_start_0based",
        "protospacer_end_0based",
        "protospacer_start_1based",
        "protospacer_end_1based",
        "pam_start_1based",
        "pam_end_1based",
        "cut_site_1based",
        "distance_cut_to_boundary_bp",
        "distance_abs_to_boundary_bp",
    ]
    for col in int_cols:
        df[col] = df[col].astype(int)

    return df.sort_values(
        ["tile_id", "boundary_type", "distance_abs_to_boundary_bp", "cut_site_1based", "guide_seq"]
    ).reset_index(drop=True)


def load_bed(path: Optional[Path], source_name: str) -> pd.DataFrame:
    empty_cols = ["chrom", "start_0based", "end_0based", "name", "score", "strand"]

    if path is None:
        return pd.DataFrame(columns=empty_cols)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=empty_cols)

    try:
        df = pd.read_csv(path, sep="\t", header=None, comment="#")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=empty_cols)

    if df.shape[1] < 3:
        raise ValueError(f"{source_name} BED must have at least 3 columns")

    df = df.iloc[:, :6].copy()
    cols = ["chrom", "start_0based", "end_0based", "name", "score", "strand"][: df.shape[1]]
    df.columns = cols
    for col in ["name", "score", "strand"]:
        if col not in df.columns:
            df[col] = "."

    df["chrom"] = df["chrom"].astype(str).str.strip().map(lambda x: x if x.startswith("chr") else f"chr{x}")
    df["start_0based"] = df["start_0based"].astype(int)
    df["end_0based"] = df["end_0based"].astype(int)
    return df.reset_index(drop=True)


def load_targets(path: Optional[Path]) -> pd.DataFrame:
    cols = ["target_id", "chrom", "start_1based", "end_1based", "label"]
    if path is None or not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(path, sep="\t")
    required = ["target_id", "chrom", "start_1based", "end_1based"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Hotspot targets TSV missing required columns: {missing}")

    out = df.copy()
    out["target_id"] = pd.to_numeric(out["target_id"], errors="coerce").astype(int)
    out["chrom"] = out["chrom"].astype(str).str.strip().map(lambda x: x if x.startswith("chr") else f"chr{x}")
    out["start_1based"] = pd.to_numeric(out["start_1based"], errors="coerce").astype(int)
    out["end_1based"] = pd.to_numeric(out["end_1based"], errors="coerce").astype(int)
    if "label" not in out.columns:
        out["label"] = out["target_id"].map(lambda x: f"target_{x}")
    out["label"] = out["label"].fillna("").astype(str)
    return out[cols].reset_index(drop=True)


def gc_fraction(seq: str) -> float:
    seq = seq.upper()
    atgc = sum(1 for b in seq if b in {"A", "C", "G", "T"})
    if atgc == 0:
        return float("nan")
    gc = sum(1 for b in seq if b in {"G", "C"})
    return gc / atgc


def shannon_entropy(seq: str) -> float:
    from collections import Counter
    import math

    seq = seq.upper()
    counts = Counter(b for b in seq if b in {"A", "C", "G", "T"})
    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for n in counts.values():
        p = n / total
        entropy -= p * math.log2(p)
    return entropy


def longest_homopolymer(seq: str) -> int:
    seq = seq.upper()
    if not seq:
        return 0

    best = 1
    cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def count_overlaps(
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    *,
    start_col: str,
    end_col: str,
) -> pd.Series:
    if candidates.empty or features.empty:
        return pd.Series([0] * len(candidates), index=candidates.index)

    counts = []
    for _, row in candidates.iterrows():
        chrom = row["chrom"]
        start_0 = int(row[start_col])
        end_0 = int(row[end_col])
        sub = features[features["chrom"] == chrom]
        mask = (sub["end_0based"] > start_0) & (sub["start_0based"] < end_0)
        counts.append(int(mask.sum()))

    return pd.Series(counts, index=candidates.index)


def interval_distance_bp(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    if end_a < start_b:
        return max(0, start_b - end_a - 1)
    if end_b < start_a:
        return max(0, start_a - end_b - 1)
    return 0


def annotate_target_distance(
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
    min_distance_to_target_bp: int,
) -> pd.DataFrame:
    out = candidates.copy()
    out["nearest_target_distance_bp"] = pd.NA
    out["nearest_target_id"] = ""
    out["nearest_target_label"] = ""
    out["guide_too_close_to_target"] = False

    if out.empty or targets.empty:
        return out

    distances = []
    target_ids = []
    target_labels = []
    too_close = []

    for _, row in out.iterrows():
        chrom = str(row["chrom"])
        guide_start = int(row["guide_plus_pam_start_0based"]) + 1
        guide_end = int(row["guide_plus_pam_end_0based"])
        sub = targets[targets["chrom"] == chrom]

        best_distance = None
        best_id = ""
        best_label = ""
        for _, target in sub.iterrows():
            dist = interval_distance_bp(
                guide_start,
                guide_end,
                int(target["start_1based"]),
                int(target["end_1based"]),
            )
            if best_distance is None or dist < best_distance:
                best_distance = dist
                best_id = str(int(target["target_id"]))
                best_label = str(target.get("label", ""))

        distances.append(best_distance if best_distance is not None else pd.NA)
        target_ids.append(best_id)
        target_labels.append(best_label)
        too_close.append(best_distance is not None and best_distance < min_distance_to_target_bp)

    out["nearest_target_distance_bp"] = distances
    out["nearest_target_id"] = target_ids
    out["nearest_target_label"] = target_labels
    out["guide_too_close_to_target"] = too_close
    return out


def annotate_sequence_qc(
    df: pd.DataFrame,
    homopolymer_threshold: int,
    low_complexity_entropy_threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    out["guide_length_bp"] = out["guide_seq"].str.len().astype(int)
    out["gc_fraction"] = out["guide_seq"].apply(gc_fraction).round(6)
    out["gc_percent"] = (out["gc_fraction"] * 100).round(2)
    out["shannon_entropy"] = out["guide_seq"].apply(shannon_entropy).round(6)
    out["longest_homopolymer"] = out["guide_seq"].apply(longest_homopolymer).astype(int)
    out["has_homopolymer"] = out["longest_homopolymer"] >= homopolymer_threshold
    out["is_low_complexity"] = out["shannon_entropy"] < low_complexity_entropy_threshold
    out["gc_too_low"] = out["gc_fraction"] < 0.20
    out["gc_too_high"] = out["gc_fraction"] > 0.80
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "n_candidates", "value": int(df.shape[0])},
        {"metric": "n_overlap_common_snp", "value": int(df["n_common_snp_overlaps"].gt(0).sum())},
        {"metric": "n_has_homopolymer", "value": int(df["has_homopolymer"].sum())},
        {"metric": "n_low_complexity", "value": int(df["is_low_complexity"].sum())},
        {"metric": "n_gc_too_low", "value": int(df["gc_too_low"].sum())},
        {"metric": "n_gc_too_high", "value": int(df["gc_too_high"].sum())},
        {"metric": "n_too_close_to_target", "value": int(df.get("guide_too_close_to_target", pd.Series(False, index=df.index)).sum())},
    ]
    return pd.DataFrame(rows)


def summarize_by_tile_boundary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(["tile_id", "boundary_type"], sort=True)
    for (tile_id, boundary_type), sub in grouped:
        rows.append(
            {
                "tile_id": int(tile_id),
                "boundary_type": boundary_type,
                "n_candidates": int(sub.shape[0]),
                "n_overlap_common_snp": int(sub["n_common_snp_overlaps"].gt(0).sum()),
                "n_has_homopolymer": int(sub["has_homopolymer"].sum()),
                "n_low_complexity": int(sub["is_low_complexity"].sum()),
                "n_too_close_to_target": int(sub.get("guide_too_close_to_target", pd.Series(False, index=sub.index)).sum()),
                "best_min_distance_bp": int(sub["distance_abs_to_boundary_bp"].min()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(Path(args.candidates_oriented_tsv))
    common_snps = load_bed(Path(args.common_snps_bed), "common_snps") if args.common_snps_bed else load_bed(None, "common_snps")
    avoid_targets = load_targets(Path(args.avoid_targets_tsv)) if args.avoid_targets_tsv else load_targets(None)

    annotated = candidates.copy()
    annotated["pam_start_0based"] = annotated["pam_start_1based"].astype(int) - 1
    annotated["pam_end_0based"] = annotated["pam_end_1based"].astype(int)
    annotated["guide_plus_pam_start_0based"] = annotated[["protospacer_start_0based", "pam_start_0based"]].min(axis=1).astype(int)
    annotated["guide_plus_pam_end_0based"] = annotated[["protospacer_end_0based", "pam_end_0based"]].max(axis=1).astype(int)

    annotated["n_common_snp_overlaps"] = count_overlaps(
        annotated,
        common_snps,
        start_col="guide_plus_pam_start_0based",
        end_col="guide_plus_pam_end_0based",
    ).astype(int)
    annotated["overlaps_common_snp"] = annotated["n_common_snp_overlaps"] > 0
    annotated = annotate_target_distance(
        annotated,
        avoid_targets,
        min_distance_to_target_bp=args.min_distance_to_target_bp,
    )

    annotated["n_repeat_overlaps"] = 0
    annotated["overlaps_repeat"] = False
    annotated["n_segdup_overlaps"] = 0
    annotated["overlaps_segdup"] = False

    annotated = annotate_sequence_qc(
        annotated,
        homopolymer_threshold=args.homopolymer_threshold,
        low_complexity_entropy_threshold=args.low_complexity_entropy_threshold,
    )

    annotated["basic_sequence_warning"] = (
        annotated["has_homopolymer"]
        | annotated["is_low_complexity"]
        | annotated["gc_too_low"]
        | annotated["gc_too_high"]
    )

    annotated = annotated.sort_values(
        [
            "tile_id",
            "boundary_type",
            "distance_abs_to_boundary_bp",
            "overlaps_common_snp",
            "guide_seq",
        ]
    ).reset_index(drop=True)

    summary_df = summarize(annotated)
    summary_by_tile_boundary_df = summarize_by_tile_boundary(annotated)

    annotated.to_csv(outdir / "candidates_annotated.tsv", sep="\t", index=False)
    summary_df.to_csv(outdir / "annotation_summary.tsv", sep="\t", index=False)
    summary_by_tile_boundary_df.to_csv(
        outdir / "annotation_summary_by_tile_boundary.tsv",
        sep="\t",
        index=False,
    )

    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
