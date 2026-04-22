#!/usr/bin/env python3
"""
Regenerate thesis histogram PNGs (full scene caption in the plot title) from a prior
``run_triplet_sampling`` output tree: ``<run_dir>/temp_<T>/caption_<id>_edges.csv``.

Example (from repo root):

  python -m thesis_mvp.replot_thesis_entropy_hists \\
    --run_dir thesis_mvp/outputs/20260219_110038 \\
    --out_dir \"thesis/Thesis Template COS/Figures\"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from thesis_mvp.caption_hist_plot import save_caption_edge_histogram, thesis_hist_filename


def load_captions(path: Path) -> list[str]:
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Output directory from run_triplet_sampling (contains temp_* subfolders)",
    )
    p.add_argument(
        "--captions_file",
        type=Path,
        default=Path("thesis_mvp/captions.txt"),
        help="Same captions file used for the run (line order must match caption ids)",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("thesis/Thesis Template COS/Figures"),
        help="Where to write PNGs (default: thesis figures folder)",
    )
    p.add_argument(
        "--pairs",
        type=str,
        default="1:0.7,15:1.0",
        help="Comma-separated caption_id:temperature pairs, e.g. 1:0.7,15:1.0",
    )
    args = p.parse_args()

    if not args.run_dir.is_dir():
        print(f"run_dir is not a directory: {args.run_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.captions_file.is_file():
        print(f"captions_file not found: {args.captions_file}", file=sys.stderr)
        sys.exit(1)

    captions = load_captions(args.captions_file)
    pairs: list[tuple[int, float]] = []
    for part in args.pairs.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            cid_s, temp_s = part.split(":", 1)
            pairs.append((int(cid_s.strip()), float(temp_s.strip())))
        except ValueError:
            print(f"Bad pair (expected id:temp): {part!r}", file=sys.stderr)
            sys.exit(1)

    for cap_id, temp in pairs:
        csv_path = args.run_dir / f"temp_{temp}" / f"caption_{cap_id}_edges.csv"
        if not csv_path.is_file():
            print(f"Missing CSV (run triplet sampling first or fix path): {csv_path}", file=sys.stderr)
            sys.exit(1)
        if cap_id < 1 or cap_id > len(captions):
            print(f"caption_id {cap_id} out of range for captions file (1..{len(captions)})", file=sys.stderr)
            sys.exit(1)

        df = pd.read_csv(csv_path)
        caption = captions[cap_id - 1]
        out_name = thesis_hist_filename(cap_id, temp, "entropy")
        out_path = args.out_dir / out_name
        save_caption_edge_histogram(df, cap_id, temp, caption, out_path, "entropy")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
