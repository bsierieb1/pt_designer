#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pysam


LOCUS_RE = re.compile(r"^(chr[\w]+):(\d+)-(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize and annotate a requested hg38 locus."
    )
    parser.add_argument(
        "--locus",
        required=True,
        help="Genomic locus in format chr:start-end, e.g. chr7:117120000-117310000",
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Path to hg38 FASTA file (indexed with .fai).",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--build",
        default="hg38",
        help="Reference build label to store in metadata. Default: hg38",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=200,
        help="Sliding window size for GC/complexity metrics. Default: 200",
    )
    parser.add_argument(
        "--window-step",
        type=int,
        default=100,
        help="Sliding window step for GC/complexity metrics. Default: 100",
    )

    # Optional tabix-indexed annotation files
    parser.add_argument(
        "--common-snps",
        default=None,
        help="Optional bgzip+tabix indexed common SNP file (VCF.gz or BED.gz).",
    )
    parser.add_argument(
        "--repeats",
        default=None,
        help="Optional bgzip+tabix indexed repeats BED.gz.",
    )
    parser.add_argument(
        "--segdups",
        default=None,
        help="Optional bgzip+tabix indexed segmental duplications BED.gz.",
    )
    parser.add_argument(
        "--genes",
        default=None,
        help="Optional bgzip+tabix indexed genes BED.gz or GFF/GTF.gz.",
    )

    return parser.parse_args()


def parse_locus(locus: str) -> Tuple[str, int, int]:
    """
    Parse locus in 1-based inclusive format: chr:start-end
    Returns: chrom, start_1based, end_1based
    """
    m = LOCUS_RE.match(locus.replace(",", ""))
    if not m:
        raise ValueError(
            f"Invalid locus '{locus}'. Expected format like chr7:117120000-117310000"
        )

    chrom = m.group(1)
    start = int(m.group(2))
    end = int(m.group(3))

    if start < 1:
        raise ValueError("Locus start must be >= 1")
    if end < start:
        raise ValueError("Locus end must be >= start")

    return chrom, start, end


def ensure_fasta_index(fasta_path: Path) -> None:
    fai = fasta_path.with_suffix(fasta_path.suffix + ".fai")
    if not fai.exists():
        print(f"[INFO] FASTA index not found. Creating {fai.name}", file=sys.stderr)
        pysam.faidx(str(fasta_path))


def validate_chromosome_in_fasta(fasta_path: Path, chrom: str) -> int:
    fasta = pysam.FastaFile(str(fasta_path))
    try:
        refs = fasta.references
        lengths = fasta.lengths
        if chrom not in refs:
            raise ValueError(
                f"Chromosome '{chrom}' not found in FASTA. Available refs include: "
                f"{', '.join(refs[:10])}{' ...' if len(refs) > 10 else ''}"
            )
        chrom_len = lengths[refs.index(chrom)]
        return chrom_len
    finally:
        fasta.close()


def fetch_sequence(fasta_path: Path, chrom: str, start_1: int, end_1: int) -> str:
    """
    Input locus is 1-based inclusive.
    pysam fetch uses 0-based half-open.
    """
    fasta = pysam.FastaFile(str(fasta_path))
    try:
        seq = fasta.fetch(chrom, start_1 - 1, end_1).upper()
        return seq
    finally:
        fasta.close()


def shannon_entropy(seq: str) -> float:
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


def gc_fraction(seq: str) -> float:
    seq = seq.upper()
    atgc = sum(1 for b in seq if b in {"A", "C", "G", "T"})
    if atgc == 0:
        return float("nan")
    gc = sum(1 for b in seq if b in {"G", "C"})
    return gc / atgc


def make_windows(
    chrom: str,
    start_1: int,
    end_1: int,
    window_size: int,
    window_step: int,
) -> List[Tuple[str, int, int]]:
    """
    Returns windows as 1-based inclusive intervals.
    """
    windows: List[Tuple[str, int, int]] = []
    cur = start_1
    while cur <= end_1:
        w_end = min(cur + window_size - 1, end_1)
        windows.append((chrom, cur, w_end))
        if w_end == end_1:
            break
        cur += window_step
    return windows


def compute_window_metrics(
    fasta_path: Path,
    chrom: str,
    start_1: int,
    end_1: int,
    window_size: int,
    window_step: int,
) -> pd.DataFrame:
    windows = make_windows(chrom, start_1, end_1, window_size, window_step)
    rows = []

    fasta = pysam.FastaFile(str(fasta_path))
    try:
        for w_chrom, w_start, w_end in windows:
            seq = fasta.fetch(w_chrom, w_start - 1, w_end).upper()
            rows.append(
                {
                    "chrom": w_chrom,
                    "start_1based": w_start,
                    "end_1based": w_end,
                    "start_0based": w_start - 1,
                    "end_0based": w_end,
                    "length_bp": len(seq),
                    "gc_fraction": round(gc_fraction(seq), 6),
                    "shannon_entropy": round(shannon_entropy(seq), 6),
                    "n_fraction": round(
                        (sum(1 for b in seq if b == "N") / len(seq)) if seq else float("nan"),
                        6,
                    ),
                }
            )
    finally:
        fasta.close()

    return pd.DataFrame(rows)


def detect_tabix_format(path: Path) -> str:
    """
    Heuristic based on extension.
    """
    name = path.name.lower()
    if name.endswith(".vcf.gz"):
        return "vcf"
    if name.endswith(".bed.gz"):
        return "bed"
    if name.endswith(".gff.gz") or name.endswith(".gtf.gz"):
        return "gff"
    return "generic"


def normalize_record(
    raw_line: str,
    source_name: str,
    file_format: str,
) -> Optional[Dict]:
    """
    Convert one line from a tabix fetch into a standardized dict.

    Outputs 0-based half-open coordinates where possible.
    Keeps raw columns too.
    """
    if not raw_line or raw_line.startswith("#"):
        return None

    fields = raw_line.rstrip("\n").split("\t")

    if file_format == "vcf":
        if len(fields) < 5:
            return None
        chrom = fields[0]
        pos_1 = int(fields[1])
        ref = fields[3]
        alt = fields[4]
        end_1 = pos_1 + len(ref) - 1
        return {
            "source": source_name,
            "feature_type": "variant",
            "chrom": chrom,
            "start_0based": pos_1 - 1,
            "end_0based": end_1,
            "start_1based": pos_1,
            "end_1based": end_1,
            "name": alt,
            "score": ".",
            "strand": ".",
            "raw": raw_line,
        }

    if file_format == "bed":
        if len(fields) < 3:
            return None
        chrom = fields[0]
        start_0 = int(fields[1])
        end_0 = int(fields[2])
        name = fields[3] if len(fields) > 3 else "."
        score = fields[4] if len(fields) > 4 else "."
        strand = fields[5] if len(fields) > 5 else "."
        return {
            "source": source_name,
            "feature_type": "interval",
            "chrom": chrom,
            "start_0based": start_0,
            "end_0based": end_0,
            "start_1based": start_0 + 1,
            "end_1based": end_0,
            "name": name,
            "score": score,
            "strand": strand,
            "raw": raw_line,
        }

    if file_format == "gff":
        if len(fields) < 9:
            return None
        chrom = fields[0]
        feature_type = fields[2]
        start_1 = int(fields[3])
        end_1 = int(fields[4])
        score = fields[5]
        strand = fields[6]
        attrs = fields[8]
        return {
            "source": source_name,
            "feature_type": feature_type,
            "chrom": chrom,
            "start_0based": start_1 - 1,
            "end_0based": end_1,
            "start_1based": start_1,
            "end_1based": end_1,
            "name": attrs,
            "score": score,
            "strand": strand,
            "raw": raw_line,
        }

    # Generic tab-delimited fallback: assume BED-like at least 3 cols
    if len(fields) >= 3:
        chrom = fields[0]
        start_0 = int(fields[1])
        end_0 = int(fields[2])
        name = fields[3] if len(fields) > 3 else "."
        return {
            "source": source_name,
            "feature_type": "interval",
            "chrom": chrom,
            "start_0based": start_0,
            "end_0based": end_0,
            "start_1based": start_0 + 1,
            "end_1based": end_0,
            "name": name,
            "score": ".",
            "strand": ".",
            "raw": raw_line,
        }

    return None


def fetch_tabix_region(
    path: Path,
    chrom: str,
    start_1: int,
    end_1: int,
    source_name: str,
) -> pd.DataFrame:
    """
    Fetch region from a bgzip+tabix indexed file.
    start_1/end_1 are 1-based inclusive.
    tabix fetch uses 0-based, half-open region internally when numeric start/end are given.
    """
    tbi_path = Path(str(path) + ".tbi")
    csi_path = Path(str(path) + ".csi")
    if not tbi_path.exists() and not csi_path.exists():
        raise FileNotFoundError(
            f"Missing tabix index for {path}. Expected {tbi_path.name} or {csi_path.name}"
        )

    file_format = detect_tabix_format(path)
    tbx = pysam.TabixFile(str(path))
    rows: List[Dict] = []
    try:
        contigs = set(tbx.contigs)

        query_chrom = chrom
        if query_chrom not in contigs:
            if chrom.startswith("chr") and chrom[3:] in contigs:
                query_chrom = chrom[3:]
            elif f"chr{chrom}" in contigs:
                query_chrom = f"chr{chrom}"
            else:
                raise ValueError(
                    f"Chromosome '{chrom}' not found in {path.name}. "
                    f"Example contigs in file: {list(tbx.contigs)[:10]}"
                )

        for raw_line in tbx.fetch(query_chrom, start_1 - 1, end_1):
            rec = normalize_record(raw_line, source_name=source_name, file_format=file_format)
            if rec is not None:
                rows.append(rec)
    finally:
        tbx.close()

    if not rows:
        return pd.DataFrame(
            columns=[
                "source",
                "feature_type",
                "chrom",
                "start_0based",
                "end_0based",
                "start_1based",
                "end_1based",
                "name",
                "score",
                "strand",
                "raw",
            ]
        )

    return pd.DataFrame(rows)


def write_fasta(out_fa: Path, chrom: str, start_1: int, end_1: int, seq: str) -> None:
    header = f">{chrom}:{start_1}-{end_1}"
    with out_fa.open("w") as fh:
        fh.write(header + "\n")
        wrap = 60
        for i in range(0, len(seq), wrap):
            fh.write(seq[i : i + wrap] + "\n")


def write_bed_single_interval(out_bed: Path, chrom: str, start_1: int, end_1: int) -> None:
    # BED is 0-based, half-open
    with out_bed.open("w") as fh:
        fh.write(f"{chrom}\t{start_1 - 1}\t{end_1}\n")


def write_annotation_bed(df: pd.DataFrame, out_bed: Path) -> None:
    if df.empty:
        out_bed.write_text("")
        return

    cols = ["chrom", "start_0based", "end_0based", "name", "score", "strand"]
    bed = df.copy()
    for col in ["name", "score", "strand"]:
        if col not in bed.columns:
            bed[col] = "."
    bed[cols].to_csv(out_bed, sep="\t", header=False, index=False)


def summarize_track(df: pd.DataFrame, source_name: str) -> Dict:
    if df.empty:
        return {
            "source": source_name,
            "n_records": 0,
            "total_bp_covered_naive_sum": 0,
        }

    lengths = (df["end_0based"] - df["start_0based"]).clip(lower=0)
    return {
        "source": source_name,
        "n_records": int(df.shape[0]),
        "total_bp_covered_naive_sum": int(lengths.sum()),
    }


def main() -> int:
    args = parse_args()

    base_outdir = Path(args.outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)
    outdir = base_outdir / "01_locus_annotation"
    outdir.mkdir(parents=True, exist_ok=True)

    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    chrom, start_1, end_1 = parse_locus(args.locus)

    ensure_fasta_index(fasta_path)
    chrom_len = validate_chromosome_in_fasta(fasta_path, chrom)

    if end_1 > chrom_len:
        raise ValueError(
            f"Locus end {end_1} exceeds chromosome length {chrom_len} for {chrom}"
        )

    length_bp = end_1 - start_1 + 1

    # Core outputs
    locus_meta = {
        "input_locus": args.locus,
        "build": args.build,
        "chrom": chrom,
        "start_1based": start_1,
        "end_1based": end_1,
        "start_0based": start_1 - 1,
        "end_0based": end_1,
        "length_bp": length_bp,
        "fasta": str(fasta_path.resolve()),
    }

    (outdir / "locus.json").write_text(json.dumps(locus_meta, indent=2))
    write_bed_single_interval(outdir / "locus.bed", chrom, start_1, end_1)

    seq = fetch_sequence(fasta_path, chrom, start_1, end_1)
    write_fasta(outdir / "locus.fa", chrom, start_1, end_1, seq)

    window_df = compute_window_metrics(
        fasta_path=fasta_path,
        chrom=chrom,
        start_1=start_1,
        end_1=end_1,
        window_size=args.window_size,
        window_step=args.window_step,
    )
    window_df.to_csv(outdir / "sequence_windows.tsv", sep="\t", index=False)

    # Optional annotation tracks
    track_specs = [
        ("common_snps", args.common_snps),
        ("repeats", args.repeats),
        ("segdups", args.segdups),
        ("genes", args.genes),
    ]

    merged_tracks: List[pd.DataFrame] = []
    summaries: List[Dict] = []

    for track_name, path_str in track_specs:
        if not path_str:
            continue

        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"{track_name} file not found: {path}")

        df = fetch_tabix_region(
            path=path,
            chrom=chrom,
            start_1=start_1,
            end_1=end_1,
            source_name=track_name,
        )

        df.to_csv(outdir / f"{track_name}.tsv", sep="\t", index=False)
        write_annotation_bed(df, outdir / f"{track_name}.bed")
        summaries.append(summarize_track(df, track_name))

        if not df.empty:
            merged_tracks.append(df)

    if merged_tracks:
        merged_df = pd.concat(merged_tracks, axis=0, ignore_index=True)
    else:
        merged_df = pd.DataFrame(
            columns=[
                "source",
                "feature_type",
                "chrom",
                "start_0based",
                "end_0based",
                "start_1based",
                "end_1based",
                "name",
                "score",
                "strand",
                "raw",
            ]
        )

    merged_df.to_csv(outdir / "locus_annotations.tsv", sep="\t", index=False)
    pd.DataFrame(summaries).to_csv(outdir / "annotation_summary.tsv", sep="\t", index=False)

    print(f"[OK] Wrote normalized locus outputs to: {outdir}", file=sys.stderr)
    print(f"[OK] Locus length: {length_bp} bp", file=sys.stderr)
    print(f"[OK] Sequence windows: {len(window_df)}", file=sys.stderr)

    if summaries:
        for s in summaries:
            print(
                f"[OK] {s['source']}: {s['n_records']} records, "
                f"{s['total_bp_covered_naive_sum']} bp naive summed span",
                file=sys.stderr,
            )
    else:
        print("[INFO] No annotation tracks provided; only locus normalization + sequence metrics were run.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
