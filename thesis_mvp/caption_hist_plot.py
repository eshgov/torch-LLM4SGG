"""
Per-caption edge histogram figures (entropy or probability) with full scene captions in the title.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pandas as pd

PlotType = Literal["entropy", "prob"]


def format_per_caption_hist_title(
    cap_one_based: int,
    temp: float,
    caption: str,
    *,
    wrap_width: int = 56,
) -> str:
    """Title: one header line, then the full scene caption word-wrapped (no truncation)."""
    header = f"Caption {cap_one_based} (T={temp})"
    cap = (caption or "").strip()
    if not cap:
        return header
    wrapped = textwrap.wrap(cap, width=wrap_width)
    if not wrapped:
        wrapped = [cap]
    return header + "\n" + "\n".join(wrapped)


def per_caption_hist_fig_height(caption: str, *, wrap_width: int = 56) -> float:
    """Extra vertical space when the scene caption spans many lines."""
    cap = (caption or "").strip()
    n_lines = len(textwrap.wrap(cap, width=wrap_width)) if cap else 1
    # Histogram body + title block; cap height for very long captions
    return min(10.0, 3.15 + 0.2 * max(2, n_lines + 1))


def save_caption_edge_histogram(
    df: "pd.DataFrame",
    cap_one_based: int,
    temp: float,
    caption: str,
    out_path: Path,
    plot_type: PlotType = "entropy",
    *,
    wrap_width: int = 56,
    dpi: int = 100,
) -> None:
    """Write a single histogram PNG (used by run_triplet_sampling and replot CLI)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        return

    fig_h = per_caption_hist_fig_height(caption, wrap_width=wrap_width)
    _, ax = plt.subplots(figsize=(6, fig_h))

    if plot_type == "entropy":
        ax.hist(df["entropy"], bins=min(20, max(1, len(df))), edgecolor="black", alpha=0.7)
        ax.set_xlabel("Edge entropy H(e)")
    else:
        ax.hist(df["probability"], bins=min(20, max(1, len(df))), edgecolor="black", alpha=0.7)
        ax.set_xlabel("Edge probability P(e)")

    ax.set_ylabel("Count")
    ax.set_title(
        format_per_caption_hist_title(cap_one_based, temp, caption, wrap_width=wrap_width),
        fontsize=9,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def thesis_hist_filename(cap_one_based: int, temp: float, plot_type: PlotType) -> str:
    """Match filenames used in the thesis (e.g. caption_1_hist_entropy_T0p7.png)."""
    ttag = str(temp).replace(".", "p")
    return f"caption_{cap_one_based}_hist_{plot_type}_T{ttag}.png"
