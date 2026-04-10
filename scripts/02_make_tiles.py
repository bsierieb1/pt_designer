#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class Tile:
    tile_id: int
    chrom: str
    start_1based: int
    end_1based: int

    @property
    def length_bp(self) -> int:
        return self.end_1based - self.start_1based + 1

    @property
    def start_0based(self) -> int:
        return self.start_1based - 1

    @property
    def end_0based(self) -> int:
        return self.end_1based


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create initial overlapping tiles and optimize overlap SNP content."
    )
    parser.add_argument(
        "--locus-json",
        required=True,
        help="Path to locus.json from Step 1.",
    )
    parser.add_argument(
        "--common-snps-bed",
        required=False,
        default=None,
        help="Path to common_snps.bed from Step 1. If omitted, SNP optimization is skipped.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--target-tile-size",
        type=int,
        default=10000,
        help="Desired tile size in bp. Default: 10000",
    )
    parser.add_argument(
        "--target-overlap-size",
        type=int,
        default=2000,
        help="Desired overlap size in bp. Default: 2000",
    )
    parser.add_argument(
        "--min-tile-size",
        type=int,
        default=5000,
        help="Minimum allowed tile size in bp. Default: 5000",
    )
    parser.add_argument(
        "--max-tile-size",
        type=int,
        default=15000,
        help="Maximum allowed tile size in bp. Default: 15000",
    )

    parser.add_argument(
        "--boundary-shift-max",
        type=int,
        default=1000,
        help="Maximum internal boundary shift in either direction during overlap optimization. Default: 1000",
    )
    parser.add_argument(
        "--boundary-shift-step",
        type=int,
        default=100,
        help="Step size for internal boundary optimization. Default: 100",
    )

    return parser.parse_args()


def load_locus_json(path: Path) -> Dict:
    with path.open() as fh:
        data = json.load(fh)

    required = ["chrom", "start_1based", "end_1based", "length_bp"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"locus.json missing required keys: {missing}")

    return data


def load_snps_bed(path: Optional[Path], chrom: str) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["chrom", "start_0based", "end_0based"])

    if not path.exists():
        raise FileNotFoundError(f"common_snps BED not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        comment="#",
        dtype={0: str},
    )

    if df.shape[1] < 3:
        raise ValueError(f"SNP BED must have at least 3 columns: {path}")

    df = df.iloc[:, :6].copy()
    cols = ["chrom", "start_0based", "end_0based", "name", "score", "strand"][: df.shape[1]]
    df.columns = cols

    for col in ["name", "score", "strand"]:
        if col not in df.columns:
            df[col] = "."

    df = df[df["chrom"] == chrom].copy()
    df["start_0based"] = df["start_0based"].astype(int)
    df["end_0based"] = df["end_0based"].astype(int)

    return df.reset_index(drop=True)


def build_initial_tiles(
    chrom: str,
    locus_start_1: int,
    locus_end_1: int,
    target_tile_size: int,
    target_overlap_size: int,
) -> List[Tile]:
    """
    Build a first-pass overlapping tile scaffold with similarly sized tiles.

    Instead of repeatedly stepping forward until we create a short terminal tile,
    we first choose the number of tiles from the locus length and target tile size,
    then distribute tile starts so that tiles stay close to the requested size while
    still fully covering the locus. This lets the effective overlap expand when needed
    to avoid small terminal slivers.
    """
    if target_overlap_size >= target_tile_size:
        raise ValueError("target_overlap_size must be smaller than target_tile_size")

    locus_len = locus_end_1 - locus_start_1 + 1
    if locus_len <= target_tile_size:
        return [Tile(1, chrom, locus_start_1, locus_end_1)]

    n_tiles = max(2, math.ceil(locus_len / target_tile_size))

    # Keep tiles near the requested size and absorb any remainder by increasing the
    # realized overlap between adjacent tiles instead of creating a short final tile.
    step = math.ceil((locus_len - target_tile_size) / (n_tiles - 1))
    step = max(1, step)

    tiles: List[Tile] = []
    for i in range(n_tiles):
        if i == n_tiles - 1:
            start = locus_end_1 - target_tile_size + 1
        else:
            start = locus_start_1 + (i * step)

        start = max(locus_start_1, min(start, locus_end_1 - target_tile_size + 1))
        end = min(start + target_tile_size - 1, locus_end_1)
        tiles.append(Tile(i + 1, chrom, start, end))

    # Enforce monotonic starts in case of heavy clamping near the end of a locus.
    tiles = sorted(tiles, key=lambda t: (t.start_1based, t.end_1based, t.tile_id))

    for i, t in enumerate(tiles, start=1):
        t.tile_id = i

    return tiles


def tile_to_dict(tile: Tile) -> Dict:
    return {
        "tile_id": tile.tile_id,
        "chrom": tile.chrom,
        "start_1based": tile.start_1based,
        "end_1based": tile.end_1based,
        "start_0based": tile.start_0based,
        "end_0based": tile.end_0based,
        "length_bp": tile.length_bp,
    }


def compute_overlaps(tiles: List[Tile], snps_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(len(tiles) - 1):
        left = tiles[i]
        right = tiles[i + 1]

        overlap_start = max(left.start_1based, right.start_1based)
        overlap_end = min(left.end_1based, right.end_1based)

        if overlap_end >= overlap_start:
            overlap_len = overlap_end - overlap_start + 1
            overlap_start_0 = overlap_start - 1
            overlap_end_0 = overlap_end

            if snps_df.empty:
                snp_count = 0
            else:
                # BED overlap test on 0-based half-open coordinates
                mask = (
                    (snps_df["end_0based"] > overlap_start_0)
                    & (snps_df["start_0based"] < overlap_end_0)
                )
                snp_count = int(mask.sum())

            snp_density_per_kb = round((snp_count / overlap_len) * 1000, 6)
        else:
            overlap_start = None
            overlap_end = None
            overlap_start_0 = None
            overlap_end_0 = None
            overlap_len = 0
            snp_count = 0
            snp_density_per_kb = 0.0

        rows.append(
            {
                "left_tile_id": left.tile_id,
                "right_tile_id": right.tile_id,
                "chrom": left.chrom,
                "overlap_start_1based": overlap_start,
                "overlap_end_1based": overlap_end,
                "overlap_start_0based": overlap_start_0,
                "overlap_end_0based": overlap_end_0,
                "overlap_length_bp": overlap_len,
                "overlap_snp_count": snp_count,
                "overlap_snp_density_per_kb": snp_density_per_kb,
            }
        )

    return pd.DataFrame(rows)


def tile_lengths_ok(tiles: List[Tile], min_tile_size: int, max_tile_size: int) -> bool:
    return all(min_tile_size <= t.length_bp <= max_tile_size for t in tiles)


def locus_fully_covered(tiles: List[Tile], locus_start_1: int, locus_end_1: int) -> bool:
    if not tiles:
        return False
    if tiles[0].start_1based > locus_start_1:
        return False
    if tiles[-1].end_1based < locus_end_1:
        return False

    # no gaps
    for i in range(len(tiles) - 1):
        if tiles[i + 1].start_1based > tiles[i].end_1based + 1:
            return False
    return True


def tiles_have_valid_geometry(tiles: List[Tile]) -> bool:
    """
    Enforce sensible left-to-right tile ordering.

    Adjacent tiles may overlap, but they must not be identical or nested. In
    practice that means both starts and ends must increase monotonically across
    the ordered tile list.
    """
    if not tiles:
        return False

    for i in range(len(tiles) - 1):
        left = tiles[i]
        right = tiles[i + 1]

        if not (left.start_1based < right.start_1based):
            return False
        if not (left.end_1based < right.end_1based):
            return False

    return True


def score_overlap_set(overlaps_df: pd.DataFrame) -> Tuple[int, int, int, float]:
    """
    Lexicographic overlap objective:
    1) maximize the number of overlaps that contain at least one common SNP
    2) minimize the number of overlaps that contain zero common SNPs
    3) maximize total SNP count across all overlaps
    4) maximize total SNP density across all overlaps

    This more strongly disfavors designs where one overlap is SNP-rich but another
    overlap is completely SNP-free, which matters for downstream phasing.
    """
    if overlaps_df.empty:
        return 0, 0, 0, 0.0

    n_overlaps_with_snp = int(overlaps_df["overlap_snp_count"].gt(0).sum())
    n_zero_snp_overlaps = int(overlaps_df["overlap_snp_count"].eq(0).sum())
    total_snps = int(overlaps_df["overlap_snp_count"].sum())
    total_density = float(overlaps_df["overlap_snp_density_per_kb"].sum())
    return n_overlaps_with_snp, -n_zero_snp_overlaps, total_snps, total_density


def optimize_internal_boundaries(
    tiles: List[Tile],
    snps_df: pd.DataFrame,
    locus_start_1: int,
    locus_end_1: int,
    min_tile_size: int,
    max_tile_size: int,
    boundary_shift_max: int,
    boundary_shift_step: int,
) -> List[Tile]:
    """
    Optimize one internal boundary at a time.

    Boundary between tile i and tile i+1 is represented by:
      tile_i.end_1based and tile_(i+1).start_1based

    We shift the boundary by moving both together:
      new_left_end = old_left_end + delta
      new_right_start = old_right_start + delta

    This preserves overlap length between those two tiles, but moves the overlap window.
    It also affects the lengths of the two adjacent tiles.

    We greedily accept a shift if it improves the total overlap SNP score.
    """
    if len(tiles) < 2 or snps_df.empty:
        return tiles

    optimized = [Tile(t.tile_id, t.chrom, t.start_1based, t.end_1based) for t in tiles]

    improved = True
    while improved:
        improved = False

        for i in range(len(optimized) - 1):
            base_tiles = [Tile(t.tile_id, t.chrom, t.start_1based, t.end_1based) for t in optimized]
            base_overlaps = compute_overlaps(base_tiles, snps_df)
            base_score = score_overlap_set(base_overlaps)

            best_tiles = base_tiles
            best_score = base_score

            left = base_tiles[i]
            right = base_tiles[i + 1]

            for delta in range(-boundary_shift_max, boundary_shift_max + 1, boundary_shift_step):
                if delta == 0:
                    continue

                candidate_tiles = [Tile(t.tile_id, t.chrom, t.start_1based, t.end_1based) for t in base_tiles]
                cand_left = candidate_tiles[i]
                cand_right = candidate_tiles[i + 1]

                cand_left.end_1based = left.end_1based + delta
                cand_right.start_1based = right.start_1based + delta

                # Preserve ordering and boundary sanity
                if cand_left.end_1based < cand_left.start_1based:
                    continue
                if cand_right.end_1based < cand_right.start_1based:
                    continue
                if cand_left.end_1based >= locus_end_1 + max_tile_size:
                    continue
                if cand_right.start_1based < locus_start_1 - max_tile_size:
                    continue

                # Keep the same locus coverage edges
                candidate_tiles[0].start_1based = optimized[0].start_1based
                candidate_tiles[-1].end_1based = optimized[-1].end_1based

                if not tile_lengths_ok(candidate_tiles, min_tile_size, max_tile_size):
                    continue
                if not locus_fully_covered(candidate_tiles, locus_start_1, locus_end_1):
                    continue
                if not tiles_have_valid_geometry(candidate_tiles):
                    continue

                candidate_overlaps = compute_overlaps(candidate_tiles, snps_df)
                candidate_score = score_overlap_set(candidate_overlaps)

                if candidate_score > best_score:
                    best_tiles = candidate_tiles
                    best_score = candidate_score

            if best_score > base_score:
                optimized = best_tiles
                improved = True

    # final renumber
    for idx, tile in enumerate(optimized, start=1):
        tile.tile_id = idx

    return optimized


def write_bed(df: pd.DataFrame, outpath: Path, kind: str) -> None:
    if df.empty:
        outpath.write_text("")
        return

    if kind == "tiles":
        bed = df[["chrom", "start_0based", "end_0based", "tile_id"]].copy()
        bed["score"] = "."
        bed["strand"] = "."
        bed.to_csv(outpath, sep="\t", header=False, index=False)

    elif kind == "overlaps":
        tmp = df.copy()
        tmp = tmp[tmp["overlap_length_bp"] > 0].copy()
        if tmp.empty:
            outpath.write_text("")
            return
        bed = tmp[
            ["chrom", "overlap_start_0based", "overlap_end_0based", "left_tile_id", "overlap_snp_count"]
        ].copy()
        bed["strand"] = "."
        bed.to_csv(outpath, sep="\t", header=False, index=False)

    else:
        raise ValueError(f"Unsupported BED kind: {kind}")


def main() -> int:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    locus = load_locus_json(Path(args.locus_json))
    chrom = locus["chrom"]
    locus_start_1 = int(locus["start_1based"])
    locus_end_1 = int(locus["end_1based"])

    snps_df = load_snps_bed(
        Path(args.common_snps_bed) if args.common_snps_bed else None,
        chrom=chrom,
    )

    initial_tiles = build_initial_tiles(
        chrom=chrom,
        locus_start_1=locus_start_1,
        locus_end_1=locus_end_1,
        target_tile_size=args.target_tile_size,
        target_overlap_size=args.target_overlap_size,
    )

    initial_tiles_df = pd.DataFrame([tile_to_dict(t) for t in initial_tiles])
    initial_overlaps_df = compute_overlaps(initial_tiles, snps_df)

    optimized_tiles = optimize_internal_boundaries(
        tiles=initial_tiles,
        snps_df=snps_df,
        locus_start_1=locus_start_1,
        locus_end_1=locus_end_1,
        min_tile_size=args.min_tile_size,
        max_tile_size=args.max_tile_size,
        boundary_shift_max=args.boundary_shift_max,
        boundary_shift_step=args.boundary_shift_step,
    )

    optimized_tiles_df = pd.DataFrame([tile_to_dict(t) for t in optimized_tiles])
    optimized_overlaps_df = compute_overlaps(optimized_tiles, snps_df)

    # Summary
    initial_score = score_overlap_set(initial_overlaps_df)
    optimized_score = score_overlap_set(optimized_overlaps_df)

    summary = {
        "chrom": chrom,
        "locus_start_1based": locus_start_1,
        "locus_end_1based": locus_end_1,
        "target_tile_size": args.target_tile_size,
        "target_overlap_size": args.target_overlap_size,
        "min_tile_size": args.min_tile_size,
        "max_tile_size": args.max_tile_size,
        "boundary_shift_max": args.boundary_shift_max,
        "boundary_shift_step": args.boundary_shift_step,
        "n_tiles_initial": int(initial_tiles_df.shape[0]),
        "n_tiles_optimized": int(optimized_tiles_df.shape[0]),
        "initial_total_overlap_snps": int(initial_score[0]),
        "optimized_total_overlap_snps": int(optimized_score[0]),
        "initial_total_overlap_density_sum": float(initial_score[1]),
        "optimized_total_overlap_density_sum": float(optimized_score[1]),
    }

    # Write outputs
    (outdir / "tile_design_summary.json").write_text(json.dumps(summary, indent=2))

    initial_tiles_df.to_csv(outdir / "tiles_initial.tsv", sep="\t", index=False)
    initial_overlaps_df.to_csv(outdir / "tile_overlaps_initial.tsv", sep="\t", index=False)

    optimized_tiles_df.to_csv(outdir / "tiles_phasing_optimized.tsv", sep="\t", index=False)
    optimized_overlaps_df.to_csv(outdir / "tile_overlaps_phasing_optimized.tsv", sep="\t", index=False)

    write_bed(initial_tiles_df, outdir / "tiles_initial.bed", kind="tiles")
    write_bed(optimized_tiles_df, outdir / "tiles_phasing_optimized.bed", kind="tiles")
    write_bed(initial_overlaps_df, outdir / "tile_overlaps_initial.bed", kind="overlaps")
    write_bed(optimized_overlaps_df, outdir / "tile_overlaps_phasing_optimized.bed", kind="overlaps")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
