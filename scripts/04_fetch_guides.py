#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


UCSC_API = "https://api.genome.ucsc.edu"
MIT_RE = re.compile(r"MIT[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
DOENCH_RE = re.compile(r"Doench[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)%", re.IGNORECASE)
MORENO_RE = re.compile(r"Moreno[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)%", re.IGNORECASE)
BAE_RE = re.compile(r"out[- ]of[- ]frame[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch predesigned UCSC CRISPR guides overlapping tile-boundary windows."
    )
    parser.add_argument(
        "--boundary-windows-tsv",
        required=True,
        help="Path to boundary_windows.tsv from Step 3.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--genome",
        default="hg38",
        help="UCSC genome assembly. Default: hg38",
    )
    parser.add_argument(
        "--ucsc-track",
        default="crisprAllTargets",
        help="UCSC CRISPR track name. Default: crisprAllTargets",
    )
    parser.add_argument(
        "--request-sleep-seconds",
        type=float,
        default=1.0,
        help="Sleep between UCSC API requests. Default: 1.0",
    )
    return parser.parse_args()


def _http_get_json(url: str, timeout: int = 60) -> dict:
    request = Request(url, headers={"User-Agent": "PacBio-PureTarget-Prototype/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} for {url}\n{body}") from e
    except URLError as e:
        raise RuntimeError(f"Request failed for {url}: {e}") from e


def reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(comp)[::-1].upper()


def load_boundary_windows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    required = [
        "tile_id",
        "chrom",
        "boundary_type",
        "desired_boundary_1based",
        "window_start_1based",
        "window_end_1based",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"boundary_windows.tsv missing required columns: {missing}")

    df["tile_id"] = df["tile_id"].astype(int)
    df["desired_boundary_1based"] = df["desired_boundary_1based"].astype(int)
    df["window_start_1based"] = df["window_start_1based"].astype(int)
    df["window_end_1based"] = df["window_end_1based"].astype(int)

    return df.sort_values(["tile_id", "boundary_type"]).reset_index(drop=True)


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


def first_nonnull(item: dict, keys: List[str], default=None):
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return default


def extract_score(text: str, pattern: re.Pattern[str]) -> Optional[float]:
    if not text:
        return None
    m = pattern.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def build_row(
    item: dict,
    chrom: str,
    tile_id: int,
    boundary_type: str,
    desired_boundary_1based: int,
    window_start_1based: int,
    window_end_1based: int,
    window_seq: str,
) -> Optional[Dict]:
    chrom_start = first_nonnull(item, ["chromStart", "start"])
    chrom_end = first_nonnull(item, ["chromEnd", "end"])
    if chrom_start is None or chrom_end is None:
        return None
    chrom_start = int(chrom_start)
    chrom_end = int(chrom_end)
    if chrom_end <= chrom_start:
        return None

    strand = str(first_nonnull(item, ["strand"], "."))
    thick_start = int(first_nonnull(item, ["thickStart"], chrom_start))
    thick_end = int(first_nonnull(item, ["thickEnd"], chrom_end))
    if thick_end <= thick_start:
        # fall back to a simple 20+3 split from the outer span
        if strand == "+":
            thick_start = chrom_start
            thick_end = max(chrom_start, chrom_end - 3)
        elif strand == "-":
            thick_start = min(chrom_end, chrom_start + 3)
            thick_end = chrom_end
        else:
            return None

    guide_start_0 = thick_start
    guide_end_0 = thick_end
    if strand == "+":
        pam_start_0 = guide_end_0
        pam_end_0 = min(chrom_end, guide_end_0 + 3)
    elif strand == "-":
        pam_start_0 = max(chrom_start, guide_start_0 - 3)
        pam_end_0 = guide_start_0
    else:
        return None

    if guide_end_0 <= guide_start_0 or pam_end_0 <= pam_start_0:
        return None

    local_guide_start = guide_start_0 - (window_start_1based - 1)
    local_guide_end = guide_end_0 - (window_start_1based - 1)
    local_pam_start = pam_start_0 - (window_start_1based - 1)
    local_pam_end = pam_end_0 - (window_start_1based - 1)
    if min(local_guide_start, local_guide_end, local_pam_start, local_pam_end) < 0:
        return None
    if max(local_guide_end, local_pam_end) > len(window_seq):
        return None

    guide_plus = window_seq[local_guide_start:local_guide_end]
    pam_plus = window_seq[local_pam_start:local_pam_end]
    if strand == "+":
        guide_seq = guide_plus.upper()
        pam_seq = pam_plus.upper()
        cut_site_1based = guide_end_0 - 2
    else:
        guide_seq = reverse_complement(guide_plus)
        pam_seq = reverse_complement(pam_plus)
        cut_site_1based = guide_start_0 + 3

    mouse_over = str(first_nonnull(item, ["_mouseOver", "mouseOver", "description"], ""))
    score = first_nonnull(item, ["score"], None)
    try:
        mit = float(score) if score not in (None, "") else None
    except ValueError:
        mit = None
    mit = mit if mit is not None else extract_score(mouse_over, MIT_RE)
    doench = extract_score(mouse_over, DOENCH_RE)
    moreno = extract_score(mouse_over, MORENO_RE)
    bae = extract_score(mouse_over, BAE_RE)

    guide_name = str(first_nonnull(item, ["name"], guide_seq))
    return {
        "tile_id": tile_id,
        "chrom": chrom,
        "boundary_type": boundary_type,
        "desired_boundary_1based": desired_boundary_1based,
        "window_start_1based": window_start_1based,
        "window_end_1based": window_end_1based,
        "strand": strand,
        "guide_name": guide_name,
        "protospacer_seq": guide_seq,
        "pam_seq": pam_seq,
        "guide_seq": guide_seq,
        "protospacer_start_1based": guide_start_0 + 1,
        "protospacer_end_1based": guide_end_0,
        "protospacer_start_0based": guide_start_0,
        "protospacer_end_0based": guide_end_0,
        "pam_start_1based": pam_start_0 + 1,
        "pam_end_1based": pam_end_0,
        "pam_start_0based": pam_start_0,
        "pam_end_0based": pam_end_0,
        "cut_site_1based": cut_site_1based,
        "cut_site_0based": cut_site_1based - 1,
        "distance_cut_to_boundary_bp": cut_site_1based - desired_boundary_1based,
        "ucsc_mit_specificity_score": mit,
        "ucsc_doench2016_percentile": doench,
        "ucsc_moreno_mateos_percentile": moreno,
        "ucsc_bae_out_of_frame_score": bae,
        "ucsc_raw_score": score,
        "ucsc_mouseover": mouse_over,
        "ucsc_item_json": json.dumps(item, sort_keys=True),
    }


def write_candidates_bed(df: pd.DataFrame, outpath: Path) -> None:
    if df.empty:
        outpath.write_text("")
        return

    bed = df.copy()
    bed["name"] = (
        "tile"
        + bed["tile_id"].astype(str)
        + "_"
        + bed["boundary_type"].astype(str)
        + "_"
        + bed["strand"].astype(str)
        + "_"
        + bed["guide_seq"].astype(str)
    )
    bed["score"] = pd.to_numeric(bed["ucsc_mit_specificity_score"], errors="coerce").fillna(0).round(0).astype(int)
    bed = bed[
        [
            "chrom",
            "protospacer_start_0based",
            "protospacer_end_0based",
            "name",
            "score",
            "strand",
        ]
    ]
    bed.to_csv(outpath, sep="\t", header=False, index=False)


def main() -> int:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    boundary_df = load_boundary_windows(Path(args.boundary_windows_tsv))

    all_rows: List[Dict] = []
    for _, row in boundary_df.iterrows():
        chrom = row["chrom"]
        tile_id = int(row["tile_id"])
        boundary_type = row["boundary_type"]
        desired_boundary_1based = int(row["desired_boundary_1based"])
        window_start_1based = int(row["window_start_1based"])
        window_end_1based = int(row["window_end_1based"])

        window_seq = ucsc_get_sequence(
            genome=args.genome,
            chrom=chrom,
            start_1=window_start_1based,
            end_1=window_end_1based,
        )
        time.sleep(args.request_sleep_seconds)
        items = ucsc_get_track(
            genome=args.genome,
            track=args.ucsc_track,
            chrom=chrom,
            start_1=window_start_1based,
            end_1=window_end_1based,
        )
        time.sleep(args.request_sleep_seconds)

        for item in items:
            built = build_row(
                item=item,
                chrom=chrom,
                tile_id=tile_id,
                boundary_type=boundary_type,
                desired_boundary_1based=desired_boundary_1based,
                window_start_1based=window_start_1based,
                window_end_1based=window_end_1based,
                window_seq=window_seq,
            )
            if built is not None:
                all_rows.append(built)

    candidates_df = pd.DataFrame(all_rows)
    if candidates_df.empty:
        candidates_df = pd.DataFrame(
            columns=[
                "tile_id",
                "chrom",
                "boundary_type",
                "desired_boundary_1based",
                "window_start_1based",
                "window_end_1based",
                "strand",
                "guide_name",
                "protospacer_seq",
                "pam_seq",
                "guide_seq",
                "protospacer_start_1based",
                "protospacer_end_1based",
                "protospacer_start_0based",
                "protospacer_end_0based",
                "pam_start_1based",
                "pam_end_1based",
                "pam_start_0based",
                "pam_end_0based",
                "cut_site_1based",
                "cut_site_0based",
                "distance_cut_to_boundary_bp",
                "ucsc_mit_specificity_score",
                "ucsc_doench2016_percentile",
                "ucsc_moreno_mateos_percentile",
                "ucsc_bae_out_of_frame_score",
                "ucsc_raw_score",
                "ucsc_mouseover",
                "ucsc_item_json",
            ]
        )
        summary = {
            "n_boundary_windows": int(boundary_df.shape[0]),
            "n_candidates_total": 0,
            "n_candidates_plus": 0,
            "n_candidates_minus": 0,
        }
    else:
        candidates_df = candidates_df.drop_duplicates(
            subset=[
                "tile_id",
                "boundary_type",
                "strand",
                "protospacer_start_0based",
                "protospacer_end_0based",
                "pam_start_0based",
                "pam_end_0based",
                "guide_seq",
            ]
        ).sort_values(
            ["tile_id", "boundary_type", "strand", "cut_site_1based", "guide_seq"]
        ).reset_index(drop=True)
        summary = {
            "n_boundary_windows": int(boundary_df.shape[0]),
            "n_candidates_total": int(candidates_df.shape[0]),
            "n_candidates_plus": int((candidates_df["strand"] == "+").sum()),
            "n_candidates_minus": int((candidates_df["strand"] == "-").sum()),
        }

    candidates_df.to_csv(outdir / "candidates_raw.tsv", sep="\t", index=False)
    write_candidates_bed(candidates_df, outdir / "candidates_raw.bed")

    pd.DataFrame(
        [{"boundary_type": k, "n_candidates": int(v)} for k, v in candidates_df["boundary_type"].value_counts().to_dict().items()]
    ).to_csv(outdir / "candidate_counts_by_boundary_type.tsv", sep="\t", index=False)

    pd.DataFrame(
        [{"strand": k, "n_candidates": int(v)} for k, v in candidates_df["strand"].value_counts().to_dict().items()]
    ).to_csv(outdir / "candidate_counts_by_strand.tsv", sep="\t", index=False)

    (outdir / "candidate_scan_summary.tsv").write_text(
        "\n".join(f"{k}\t{v}" for k, v in summary.items()) + "\n"
    )

    print(pd.Series(summary).to_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
