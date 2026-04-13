#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


LOCUS_RE = re.compile(r"^(chr[\w]+):(\d+)-(\d+)$")
UCSC_API = "https://api.genome.ucsc.edu"
ENSEMBL_API = "https://rest.ensembl.org"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize and annotate a requested hg38 locus using UCSC/Ensembl remote data sources."
    )
    parser.add_argument(
        "--locus",
        required=True,
        help="Genomic locus in format chr:start-end, e.g. chr7:117120000-117310000",
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
    parser.add_argument(
        "--ucsc-snp-track",
        default="snp151Common",
        help="UCSC common-SNP track to query. Default: snp151Common",
    )
    parser.add_argument(
        "--species",
        default="human",
        help="Ensembl species name for gene annotation. Default: human",
    )
    parser.add_argument(
        "--request-sleep-seconds",
        type=float,
        default=1.0,
        help="Sleep between remote API requests to be gentle on UCSC/Ensembl. Default: 1.0",
    )
    return parser.parse_args()


def parse_locus(locus: str) -> Tuple[str, int, int]:
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


def _http_get_json(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 60) -> dict:
    request = Request(url, headers={"User-Agent": "PacBio-PureTarget-Prototype/1.0", **(headers or {})})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} for {url}\n{body}") from e
    except URLError as e:
        raise RuntimeError(f"Request failed for {url}: {e}") from e


def _http_get_text(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 60) -> str:
    request = Request(url, headers={"User-Agent": "PacBio-PureTarget-Prototype/1.0", **(headers or {})})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} for {url}\n{body}") from e
    except URLError as e:
        raise RuntimeError(f"Request failed for {url}: {e}") from e


def ucsc_get_sequence(genome: str, chrom: str, start_1: int, end_1: int) -> str:
    url = (
        f"{UCSC_API}/getData/sequence?genome={quote(genome)};chrom={quote(chrom)};"
        f"start={start_1 - 1};end={end_1}"
    )
    payload = _http_get_json(url)
    seq = payload.get("dna") or payload.get("seq") or ""
    if not seq:
        raise RuntimeError(f"UCSC sequence API returned no sequence for {chrom}:{start_1}-{end_1}")
    return str(seq).upper()


def ucsc_get_track(genome: str, track: str, chrom: str, start_1: int, end_1: int) -> List[dict]:
    url = (
        f"{UCSC_API}/getData/track?genome={quote(genome)};track={quote(track)};chrom={quote(chrom)};"
        f"start={start_1 - 1};end={end_1}"
    )
    payload = _http_get_json(url)
    items = payload.get(track, [])
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict)]


def ensembl_overlap_region(species: str, region: str, feature: str) -> List[dict]:
    url = f"{ENSEMBL_API}/overlap/region/{quote(species)}/{quote(region)}?feature={quote(feature)}"
    payload = _http_get_json(url, headers={"Content-Type": "application/json", "Accept": "application/json"})
    if not isinstance(payload, list):
        return []
    return [x for x in payload if isinstance(x, dict)]


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
    chrom: str,
    start_1: int,
    end_1: int,
    window_size: int,
    window_step: int,
    full_seq: str,
) -> pd.DataFrame:
    windows = make_windows(chrom, start_1, end_1, window_size, window_step)
    rows = []

    for w_chrom, w_start, w_end in windows:
        offset0 = w_start - start_1
        offset1 = w_end - start_1 + 1
        seq = full_seq[offset0:offset1].upper()
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

    return pd.DataFrame(rows)


def write_fasta(out_fa: Path, chrom: str, start_1: int, end_1: int, seq: str) -> None:
    header = f">{chrom}:{start_1}-{end_1}"
    with out_fa.open("w") as fh:
        fh.write(header + "\n")
        wrap = 60
        for i in range(0, len(seq), wrap):
            fh.write(seq[i : i + wrap] + "\n")


def write_bed_single_interval(out_bed: Path, chrom: str, start_1: int, end_1: int) -> None:
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


def normalize_ucsc_common_snps(items: List[dict], chrom: str) -> pd.DataFrame:
    rows: List[dict] = []
    for item in items:
        start_0 = item.get("chromStart")
        end_0 = item.get("chromEnd")
        if start_0 is None or end_0 is None:
            continue
        name = item.get("name") or item.get("rsId") or "."
        strand = item.get("strand") or "."
        rows.append(
            {
                "source": "common_snps",
                "feature_type": "variant",
                "chrom": item.get("chrom", chrom),
                "start_0based": int(start_0),
                "end_0based": int(end_0),
                "start_1based": int(start_0) + 1,
                "end_1based": int(end_0),
                "name": str(name),
                "score": str(item.get("score", ".")),
                "strand": str(strand),
                "raw": json.dumps(item, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def normalize_ensembl_genes(items: List[dict], chrom: str) -> pd.DataFrame:
    rows: List[dict] = []
    for item in items:
        start_1 = item.get("start")
        end_1 = item.get("end")
        if start_1 is None or end_1 is None:
            continue
        name = item.get("external_name") or item.get("gene_id") or item.get("id") or "."
        strand_val = item.get("strand", 0)
        strand = "+" if strand_val == 1 else "-" if strand_val == -1 else "."
        rows.append(
            {
                "source": "genes",
                "feature_type": "gene",
                "chrom": item.get("seq_region_name", chrom if not chrom.startswith("chr") else chrom[3:]),
                "start_0based": int(start_1) - 1,
                "end_0based": int(end_1),
                "start_1based": int(start_1),
                "end_1based": int(end_1),
                "name": str(name),
                "score": ".",
                "strand": strand,
                "raw": json.dumps(item, sort_keys=True),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["chrom"] = df["chrom"].astype(str).map(lambda x: x if x.startswith("chr") else f"chr{x}")
    return df


def main() -> int:
    args = parse_args()

    if args.build != "hg38":
        raise ValueError("This remote-annotation version currently supports hg38 only.")

    base_outdir = Path(args.outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)
    outdir = base_outdir / "01_locus_annotation"
    outdir.mkdir(parents=True, exist_ok=True)

    chrom, start_1, end_1 = parse_locus(args.locus)
    length_bp = end_1 - start_1 + 1
    region_no_chr = f"{chrom[3:] if chrom.startswith('chr') else chrom}:{start_1}-{end_1}"

    locus_meta = {
        "input_locus": args.locus,
        "build": args.build,
        "chrom": chrom,
        "start_1based": start_1,
        "end_1based": end_1,
        "start_0based": start_1 - 1,
        "end_0based": end_1,
        "length_bp": length_bp,
        "sequence_source": "UCSC getData/sequence",
        "gene_annotation_source": f"Ensembl overlap/region ({args.species}, feature=gene)",
        "common_snp_source": f"UCSC getData/track ({args.ucsc_snp_track})",
    }
    (outdir / "locus.json").write_text(json.dumps(locus_meta, indent=2))
    write_bed_single_interval(outdir / "locus.bed", chrom, start_1, end_1)

    seq = ucsc_get_sequence(args.build, chrom, start_1, end_1)
    write_fasta(outdir / "locus.fa", chrom, start_1, end_1, seq)
    time.sleep(args.request_sleep_seconds)

    window_df = compute_window_metrics(
        chrom=chrom,
        start_1=start_1,
        end_1=end_1,
        window_size=args.window_size,
        window_step=args.window_step,
        full_seq=seq,
    )
    window_df.to_csv(outdir / "sequence_windows.tsv", sep="\t", index=False)

    common_snps_df = normalize_ucsc_common_snps(
        ucsc_get_track(args.build, args.ucsc_snp_track, chrom, start_1, end_1),
        chrom=chrom,
    )
    time.sleep(args.request_sleep_seconds)

    genes_df = normalize_ensembl_genes(
        ensembl_overlap_region(args.species, region_no_chr, "gene"),
        chrom=chrom,
    )

    common_snps_df.to_csv(outdir / "common_snps.tsv", sep="\t", index=False)
    genes_df.to_csv(outdir / "genes.tsv", sep="\t", index=False)
    write_annotation_bed(common_snps_df, outdir / "common_snps.bed")
    write_annotation_bed(genes_df, outdir / "genes.bed")

    merged_tracks: List[pd.DataFrame] = []
    summaries: List[Dict] = []
    for track_name, df in [("common_snps", common_snps_df), ("genes", genes_df)]:
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
    for s in summaries:
        print(
            f"[OK] {s['source']}: {s['n_records']} records, {s['total_bp_covered_naive_sum']} bp naive summed span",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
