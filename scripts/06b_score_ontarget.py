#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
import pysam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score guide candidates for on-target activity using RS3 when available."
    )
    parser.add_argument(
        "--candidates-annotated-tsv",
        required=True,
        help="Path to candidates_annotated.tsv from Step 6a.",
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Path to hg38 FASTA.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--tracr-rna",
        default="Hsu2013",
        choices=["Hsu2013", "Chen2013"],
        help="RS3 tracrRNA model to use. Default: Hsu2013.",
    )
    parser.add_argument(
        "--keep-fallback",
        action="store_true",
        help="If set, fall back to the local heuristic when RS3 fails. Otherwise exit with an error.",
    )
    return parser.parse_args()


def reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(comp)[::-1].upper()


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = [
        "tile_id",
        "chrom",
        "strand",
        "guide_seq",
        "protospacer_start_1based",
        "protospacer_end_1based",
        "gc_fraction",
        "has_homopolymer",
        "is_low_complexity",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["tile_id", "protospacer_start_1based", "protospacer_end_1based"]:
        df[col] = df[col].astype(int)

    df["guide_seq"] = df["guide_seq"].astype(str).str.upper()
    return df


def fetch_context_30mer(
    fasta: pysam.FastaFile,
    chrom: str,
    strand: str,
    protospacer_start_1based: int,
    protospacer_end_1based: int,
) -> str:
    """
    Return a 30mer centered around the guide region in guide orientation.

    We use:
      4 bp upstream + 20 bp guide + 3 bp PAM + 3 bp downstream
    total = 30 bp
    """
    if strand == "+":
        start_1 = protospacer_start_1based - 4
        end_1 = protospacer_end_1based + 6
        seq = fasta.fetch(chrom, max(0, start_1 - 1), end_1).upper()
        return seq
    else:
        start_1 = protospacer_start_1based - 6
        end_1 = protospacer_end_1based + 4
        seq = fasta.fetch(chrom, max(0, start_1 - 1), end_1).upper()
        return reverse_complement(seq)


def validate_contexts(contexts: Sequence[str], guides: Sequence[str]) -> List[str]:
    issues: List[str] = []
    for i, (context, guide) in enumerate(zip(contexts, guides), start=1):
        if len(context) != 30:
            issues.append(f"row {i}: context length {len(context)} != 30")
            continue
        if guide not in context:
            issues.append(f"row {i}: guide_seq not found inside 30mer context")
        if any(base not in set("ACGTN") for base in context.upper()):
            issues.append(f"row {i}: context contains non-ACGTN characters")
    return issues


def rs3_raw_to_100(raw_score: float) -> float:
    """
    RS3 sequence scores are raw model outputs centered roughly around 0.
    Convert them monotonically onto a 0-100 scale for the rest of the pipeline.

    We use a logistic transform so:
      raw 0.0 -> 50
      large positive -> approaches 100
      large negative -> approaches 0
    """
    return round(100.0 / (1.0 + math.exp(-float(raw_score))), 4)


def try_rs3_score(contexts: List[str], tracr_rna: str) -> Tuple[List[float], List[float]]:
    try:
        from rs3.seq import predict_seq  # type: ignore

        preds = predict_seq(contexts, sequence_tracr=tracr_rna)
        raw_scores = [round(float(x), 6) for x in preds]
        scaled_scores = [rs3_raw_to_100(x) for x in raw_scores]
        return raw_scores, scaled_scores
    except Exception:
        print("[ERROR] RS3 scoring failed:", file=sys.stderr)
        traceback.print_exc()
        return [], []


def fallback_score(row: pd.Series) -> float:
    score = 60.0

    gc = float(row["gc_fraction"])
    if 0.40 <= gc <= 0.70:
        score += 20.0
    elif 0.30 <= gc < 0.40 or 0.70 < gc <= 0.80:
        score += 8.0
    else:
        score -= 10.0

    guide = row["guide_seq"].upper()

    if bool(row["has_homopolymer"]):
        score -= 12.0
    if bool(row["is_low_complexity"]):
        score -= 10.0
    if guide.endswith("GG"):
        score += 3.0
    if guide.startswith("TTTT"):
        score -= 8.0

    return round(max(0.0, min(100.0, score)), 4)


def classify_ontarget(score: float) -> str:
    if score >= 70:
        return "high"
    if score >= 50:
        return "medium"
    return "low"


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_candidates(Path(args.candidates_annotated_tsv))
    fasta = pysam.FastaFile(str(args.fasta))

    contexts = []
    try:
        for _, row in df.iterrows():
            context = fetch_context_30mer(
                fasta=fasta,
                chrom=row["chrom"],
                strand=row["strand"],
                protospacer_start_1based=int(row["protospacer_start_1based"]),
                protospacer_end_1based=int(row["protospacer_end_1based"]),
            )
            contexts.append(context)
    finally:
        fasta.close()

    problems = validate_contexts(contexts, df["guide_seq"].tolist())
    if problems:
        pd.DataFrame({"problem": problems}).to_csv(outdir / "context_validation_errors.tsv", sep="\t", index=False)
        print(f"[WARN] Found {len(problems)} context validation issues. See context_validation_errors.tsv", file=sys.stderr)

    df = df.copy()
    df["context_30mer"] = contexts

    rs3_raw_scores, rs3_scaled_scores = try_rs3_score(
        contexts=df["context_30mer"].tolist(),
        tracr_rna=args.tracr_rna,
    )

    if rs3_scaled_scores:
        if len(rs3_scaled_scores) != len(df):
            raise RuntimeError(
                f"RS3 returned {len(rs3_scaled_scores)} scores for {len(df)} guides."
            )
        df["ontarget_score_raw"] = rs3_raw_scores
        df["ontarget_score"] = rs3_scaled_scores
        df["ontarget_method"] = f"rs3_seq_{args.tracr_rna}"
    else:
        if not args.keep_fallback:
            raise RuntimeError(
                "RS3 scoring failed. Re-run after confirming 'python -c \"from rs3.seq import predict_seq\"' works, "
                "or use --keep-fallback to temporarily fall back to the heuristic."
            )
        df["ontarget_score_raw"] = float("nan")
        df["ontarget_score"] = df.apply(fallback_score, axis=1)
        df["ontarget_method"] = "fallback_heuristic"

    df["ontarget_class"] = df["ontarget_score"].apply(classify_ontarget)

    df.to_csv(outdir / "candidates_ontarget.tsv", sep="\t", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "n_guides", "value": int(df.shape[0])},
            {"metric": "ontarget_mean", "value": round(float(df["ontarget_score"].mean()), 4)},
            {"metric": "ontarget_median", "value": round(float(df["ontarget_score"].median()), 4)},
            {"metric": "n_high", "value": int((df["ontarget_class"] == "high").sum())},
            {"metric": "n_medium", "value": int((df["ontarget_class"] == "medium").sum())},
            {"metric": "n_low", "value": int((df["ontarget_class"] == "low").sum())},
            {"metric": "ontarget_method", "value": df["ontarget_method"].iloc[0]},
        ]
    )
    summary.to_csv(outdir / "ontarget_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
