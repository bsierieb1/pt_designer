#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pysam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan boundary windows for SpCas9 NGG candidate guides."
    )
    parser.add_argument(
        "--boundary-windows-tsv",
        required=True,
        help="Path to boundary_windows.tsv from Step 3.",
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Path to hg38 FASTA file.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )
    return parser.parse_args()


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


def fetch_sequence(
    fasta: pysam.FastaFile,
    chrom: str,
    start_1based: int,
    end_1based: int,
) -> str:
    return fasta.fetch(chrom, start_1based - 1, end_1based).upper()


def scan_spcas9_ngg(
    chrom: str,
    tile_id: int,
    boundary_type: str,
    desired_boundary_1based: int,
    window_start_1based: int,
    window_end_1based: int,
    window_seq: str,
) -> List[Dict]:
    """
    Scan a window for SpCas9 NGG guides on both strands.

    Definitions:
    - Plus-strand candidate:
        protospacer = seq[i:i+20]
        PAM        = seq[i+20:i+23] where PAM[1:3] == 'GG' and PAM[0] can be any base
        cut site   = last protospacer base - 3 bp upstream of PAM = i+17 (0-based in window)
        genomic cut 1-based = window_start_1based + (i + 17)

    - Minus-strand candidate:
        On the plus strand, the PAM appears as CCN.
        If seq[j:j+3] matches CCN, then the minus-strand protospacer is seq[j+3:j+23] on plus,
        reverse-complemented.
        cut site is 3 bp upstream of PAM on the minus strand, which corresponds to j+5 in plus coords.
        genomic cut 1-based = window_start_1based + (j + 5)

    We report:
    - protospacer sequence in 5'->3' guide orientation
    - PAM in target-strand orientation:
        plus strand: NGG
        minus strand: NGG after reverse-complementing CCN
    """
    rows: List[Dict] = []
    seq = window_seq
    n = len(seq)

    # Plus strand: protospacer followed by NGG
    for i in range(0, n - 23 + 1):
        protospacer = seq[i : i + 20]
        pam = seq[i + 20 : i + 23]

        if len(protospacer) != 20 or len(pam) != 3:
            continue
        if "N" in protospacer or "N" in pam:
            continue
        if pam[1:] != "GG":
            continue

        protospacer_start_1based = window_start_1based + i
        protospacer_end_1based = window_start_1based + i + 19
        pam_start_1based = window_start_1based + i + 20
        pam_end_1based = window_start_1based + i + 22
        cut_site_1based = window_start_1based + i + 17

        rows.append(
            {
                "tile_id": tile_id,
                "chrom": chrom,
                "boundary_type": boundary_type,
                "desired_boundary_1based": desired_boundary_1based,
                "window_start_1based": window_start_1based,
                "window_end_1based": window_end_1based,
                "strand": "+",
                "protospacer_seq": protospacer,
                "pam_seq": pam,
                "guide_seq": protospacer,
                "protospacer_start_1based": protospacer_start_1based,
                "protospacer_end_1based": protospacer_end_1based,
                "protospacer_start_0based": protospacer_start_1based - 1,
                "protospacer_end_0based": protospacer_end_1based,
                "pam_start_1based": pam_start_1based,
                "pam_end_1based": pam_end_1based,
                "cut_site_1based": cut_site_1based,
                "cut_site_0based": cut_site_1based - 1,
                "distance_cut_to_boundary_bp": cut_site_1based - desired_boundary_1based,
            }
        )

    # Minus strand: plus-strand pattern is CCN followed by 20 nt
    for j in range(0, n - 23 + 1):
        pam_plus = seq[j : j + 3]          # should be CCN
        protospacer_plus = seq[j + 3 : j + 23]

        if len(protospacer_plus) != 20 or len(pam_plus) != 3:
            continue
        if "N" in protospacer_plus or "N" in pam_plus:
            continue
        if pam_plus[:2] != "CC":
            continue

        guide_seq = reverse_complement(protospacer_plus)
        pam_seq = reverse_complement(pam_plus)  # should become NGG

        protospacer_start_1based = window_start_1based + j + 3
        protospacer_end_1based = window_start_1based + j + 22
        pam_start_1based = window_start_1based + j
        pam_end_1based = window_start_1based + j + 2
        cut_site_1based = window_start_1based + j + 5

        rows.append(
            {
                "tile_id": tile_id,
                "chrom": chrom,
                "boundary_type": boundary_type,
                "desired_boundary_1based": desired_boundary_1based,
                "window_start_1based": window_start_1based,
                "window_end_1based": window_end_1based,
                "strand": "-",
                "protospacer_seq": guide_seq,
                "pam_seq": pam_seq,
                "guide_seq": guide_seq,
                "protospacer_start_1based": protospacer_start_1based,
                "protospacer_end_1based": protospacer_end_1based,
                "protospacer_start_0based": protospacer_start_1based - 1,
                "protospacer_end_0based": protospacer_end_1based,
                "pam_start_1based": pam_start_1based,
                "pam_end_1based": pam_end_1based,
                "cut_site_1based": cut_site_1based,
                "cut_site_0based": cut_site_1based - 1,
                "distance_cut_to_boundary_bp": cut_site_1based - desired_boundary_1based,
            }
        )

    return rows


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
    bed["score"] = "."
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
    fasta = pysam.FastaFile(str(args.fasta))

    all_rows: List[Dict] = []
    try:
        for _, row in boundary_df.iterrows():
            chrom = row["chrom"]
            tile_id = int(row["tile_id"])
            boundary_type = row["boundary_type"]
            desired_boundary_1based = int(row["desired_boundary_1based"])
            window_start_1based = int(row["window_start_1based"])
            window_end_1based = int(row["window_end_1based"])

            seq = fetch_sequence(
                fasta=fasta,
                chrom=chrom,
                start_1based=window_start_1based,
                end_1based=window_end_1based,
            )

            rows = scan_spcas9_ngg(
                chrom=chrom,
                tile_id=tile_id,
                boundary_type=boundary_type,
                desired_boundary_1based=desired_boundary_1based,
                window_start_1based=window_start_1based,
                window_end_1based=window_end_1based,
                window_seq=seq,
            )
            all_rows.extend(rows)
    finally:
        fasta.close()

    candidates_df = pd.DataFrame(all_rows)

    if candidates_df.empty:
        summary = {
            "n_boundary_windows": int(boundary_df.shape[0]),
            "n_candidates_total": 0,
            "n_candidates_plus": 0,
            "n_candidates_minus": 0,
        }
        candidates_df = pd.DataFrame(
            columns=[
                "tile_id",
                "chrom",
                "boundary_type",
                "desired_boundary_1based",
                "window_start_1based",
                "window_end_1based",
                "strand",
                "protospacer_seq",
                "pam_seq",
                "guide_seq",
                "protospacer_start_1based",
                "protospacer_end_1based",
                "protospacer_start_0based",
                "protospacer_end_0based",
                "pam_start_1based",
                "pam_end_1based",
                "cut_site_1based",
                "cut_site_0based",
                "distance_cut_to_boundary_bp",
            ]
        )
    else:
        candidates_df = candidates_df.sort_values(
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
