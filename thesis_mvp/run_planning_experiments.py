#!/usr/bin/env python3
"""
Evaluate greedy vs uncertainty-aware replanning on predicate-disagreement cases.

Reads belief graphs from a prior ``run_triplet_sampling`` output directory (temp_*/caption_*_edges.csv),
finds (subject, object) pairs with multiple predicates, and for each pair runs simulated search
with ground_truth = object, target = subject. Writes CSV tables and comparison figures.

Usage (from repo root):
  python -m thesis_mvp.run_planning_experiments --outputs_dir thesis_mvp/outputs/20260219_110038
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from thesis_mvp.belief_graph import predicate_disagreement
from thesis_mvp.search_planner import compare_policies_simulation

Triplet = Tuple[str, str, str]


def load_belief_from_edges_csv(path: Path) -> Tuple[Dict[Triplet, int], Dict[Triplet, float]]:
    """Build edge_counts and edge_probs from caption_*_edges.csv."""
    df = pd.read_csv(path)
    edge_counts: Dict[Triplet, int] = {}
    edge_probs: Dict[Triplet, float] = {}
    for _, row in df.iterrows():
        s, p, o = str(row["subject"]), str(row["predicate"]), str(row["object"])
        edge_counts[(s, p, o)] = int(row["count"])
        edge_probs[(s, p, o)] = float(row["probability"])
    return edge_counts, edge_probs


def parse_caption_id(name: str) -> Optional[int]:
    m = re.match(r"caption_(\d+)_edges\.csv$", name)
    return int(m.group(1)) if m else None


def discover_temp_dirs(outputs_dir: Path) -> List[Tuple[float, Path]]:
    out = []
    for p in sorted(outputs_dir.glob("temp_*")):
        if not p.is_dir():
            continue
        suffix = p.name.replace("temp_", "")
        try:
            t = float(suffix)
        except ValueError:
            continue
        out.append((t, p))
    return sorted(out, key=lambda x: x[0])


def run_experiments(
    outputs_dir: Path,
    spatial_only: bool,
    max_steps: int,
    synonyms_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from thesis_mvp.canonicalize import load_synonyms_file

    synonyms = load_synonyms_file(synonyms_csv) if synonyms_csv else {}

    detail_rows: List[dict] = []
    temp_dirs = discover_temp_dirs(outputs_dir)
    if not temp_dirs:
        print(f"No temp_* subdirectories under {outputs_dir}", file=sys.stderr)
        return pd.DataFrame(), pd.DataFrame()

    summary_path = outputs_dir / "summary.csv"
    summary_df = pd.read_csv(summary_path) if summary_path.exists() else None

    for temp, temp_path in temp_dirs:
        for csv_path in sorted(temp_path.glob("caption_*_edges.csv")):
            cap_id = parse_caption_id(csv_path.name)
            if cap_id is None:
                continue
            edge_counts, edge_probs = load_belief_from_edges_csv(csv_path)
            if not edge_probs:
                continue
            disagreements = predicate_disagreement(edge_counts)
            n_dis = len(disagreements)
            caption_text = ""
            if summary_df is not None:
                sub = summary_df[summary_df["caption_id"] == cap_id].copy()
                sub["temperature"] = sub["temperature"].astype(float)
                sub = sub[abs(sub["temperature"] - float(temp)) < 1e-6]
                if len(sub) > 0:
                    caption_text = str(sub.iloc[0].get("caption_text", ""))[:300]

            for idx, ((s, o), pred_list) in enumerate(disagreements):
                g, u, first_diff = compare_policies_simulation(
                    edge_probs,
                    s,
                    o,
                    synonyms=synonyms,
                    spatial_only=spatial_only,
                    max_steps=max_steps,
                )
                preds_str = ";".join(f"{p}:{c}" for p, c in pred_list)
                detail_rows.append({
                    "temperature": temp,
                    "caption_id": cap_id,
                    "caption_text": caption_text,
                    "disagreement_index": idx,
                    "subject": s,
                    "object": o,
                    "predicates_counts": preds_str,
                    "steps_greedy": g.steps,
                    "steps_uncertainty_aware": u.steps,
                    "success_greedy": g.success,
                    "success_uncertainty_aware": u.success,
                    "first_choice_differs": first_diff,
                    "visit_order_greedy": ";".join(g.visited),
                    "visit_order_ua": ";".join(u.visited),
                    "step_delta_ua_minus_greedy": u.steps - g.steps,
                })

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        return detail, pd.DataFrame()

    # Aggregate by temperature
    agg_rows = []
    for temp in sorted(detail["temperature"].unique()):
        sub = detail[detail["temperature"] == temp]
        n = len(sub)
        wins_ua = int((sub["step_delta_ua_minus_greedy"] < 0).sum())
        wins_g = int((sub["step_delta_ua_minus_greedy"] > 0).sum())
        ties = int((sub["step_delta_ua_minus_greedy"] == 0).sum())
        std_g = float(sub["steps_greedy"].std(ddof=0))
        std_u = float(sub["steps_uncertainty_aware"].std(ddof=0))
        sem_g = std_g / math.sqrt(n) if n > 0 else 0.0
        sem_u = std_u / math.sqrt(n) if n > 0 else 0.0
        agg_rows.append({
            "temperature": temp,
            "n_disagreement_cases": n,
            "mean_steps_greedy": sub["steps_greedy"].mean(),
            "std_steps_greedy": std_g,
            "sem_steps_greedy": sem_g,
            "mean_steps_uncertainty_aware": sub["steps_uncertainty_aware"].mean(),
            "std_steps_uncertainty_aware": std_u,
            "sem_steps_uncertainty_aware": sem_u,
            "median_steps_greedy": sub["steps_greedy"].median(),
            "median_steps_uncertainty_aware": sub["steps_uncertainty_aware"].median(),
            "n_ua_fewer_steps": wins_ua,
            "n_greedy_fewer_steps": wins_g,
            "n_tie_steps": ties,
            "n_first_choice_differs": int(sub["first_choice_differs"].sum()),
            "success_rate_greedy": sub["success_greedy"].mean(),
            "success_rate_ua": sub["success_uncertainty_aware"].mean(),
        })
    aggregate = pd.DataFrame(agg_rows)
    return detail, aggregate


def plot_comparisons(detail: pd.DataFrame, aggregate: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    if not aggregate.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        temps = [str(t) for t in aggregate["temperature"].tolist()]
        x = list(range(len(temps)))
        w = 0.35
        x_g = [i - w / 2 for i in x]
        x_u = [i + w / 2 for i in x]
        ax.bar(
            x_g,
            aggregate["mean_steps_greedy"],
            width=w,
            label="Greedy (argmax P̃)",
            edgecolor="black",
            yerr=aggregate["sem_steps_greedy"].fillna(0),
            capsize=3,
        )
        ax.bar(
            x_u,
            aggregate["mean_steps_uncertainty_aware"],
            width=w,
            label="UA (P̃ × (H_pred+δ))",
            edgecolor="black",
            yerr=aggregate["sem_steps_uncertainty_aware"].fillna(0),
            capsize=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"T={t}" for t in temps])
        ax.set_ylabel("Mean simulated steps to goal")
        ax.set_title("Mean steps ± standard error (same cases per temperature)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "planning_mean_steps_by_temp.png", dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(7, 4))
        cats = ["UA fewer steps", "Tie", "Greedy fewer steps"]
        xb = list(range(len(temps)))
        bottoms = [0.0] * len(temps)
        ua_vals = aggregate["n_ua_fewer_steps"].tolist()
        tie_vals = aggregate["n_tie_steps"].tolist()
        g_vals = aggregate["n_greedy_fewer_steps"].tolist()
        stacks = [ua_vals, tie_vals, g_vals]
        colors = ["#2ca02c", "#7f7f7f", "#d62728"]
        for vals, c, lab in zip(stacks, colors, cats):
            ax.bar(xb, vals, bottom=bottoms, label=lab, color=c, edgecolor="black")
            bottoms = [bottoms[i] + vals[i] for i in range(len(temps))]
        ax.set_xticks(xb)
        ax.set_xticklabels([f"T={t}" for t in temps])
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Count of disagreement cases")
        ax.set_title("Paired comparison: who used fewer search steps?")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "planning_win_counts_stacked.png", dpi=150)
        plt.close()

    if len(detail) >= 2:
        _plot_step_delta_distribution(detail, out_dir)


def _plot_step_delta_distribution(detail: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart of paired difference (UA steps − greedy steps) per temperature."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = "step_delta_ua_minus_greedy"
    dmin = int(detail[col].min())
    dmax = int(detail[col].max())
    delta_range = list(range(dmin, dmax + 1))
    temps = sorted(detail["temperature"].unique())
    n_t = len(temps)
    fig, axes = plt.subplots(1, n_t, figsize=(2.6 * n_t, 3.4), sharey=True, squeeze=False)
    for ax, t in zip(axes[0], temps):
        sub = detail[detail["temperature"] == t]
        counts = sub[col].value_counts()
        ys = [int(counts.get(d, 0)) for d in delta_range]
        colors = [
            "#2ca02c" if d < 0 else "#7f7f7f" if d == 0 else "#d62728" for d in delta_range
        ]
        ax.bar(delta_range, ys, color=colors, edgecolor="black", width=0.88)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.45)
        ax.set_title(f"$T={t}$", fontsize=11)
        ax.set_xlabel(r"$\Delta$ steps (UA $-$ greedy)")
        ax.set_xticks(delta_range)
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].set_ylabel("Number of cases")
    fig.suptitle(
        "Paired difference in search depth (negative: UA used fewer steps)",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "planning_step_delta_by_temp.png", dpi=150, bbox_inches="tight")
    plt.close()


def write_synthetic_outputs(base: Path, k: int = 20) -> None:
    """
    Minimal run_triplet_sampling-like tree so experiments always have example disagreement cases.
    Scenario: target man, true location door; desk competes. Greedy may visit desk first; UA may differ.
    """
    base.mkdir(parents=True, exist_ok=True)
    rows = [
        ("man", "by", "door", 10),
        ("man", "next to", "door", 6),
        ("man", "on", "desk", 8),
        ("man", "near", "desk", 5),
        ("man", "in", "room", 4),
    ]
    # One temperature folder is enough for a slide demo; bar chart code still works with a single T.
    for temp in (0.7,):
        tdir = base / f"temp_{temp}"
        tdir.mkdir(parents=True, exist_ok=True)
        lines = ["subject,predicate,object,count,probability,entropy"]
        for s, p, o, c in rows:
            prob = c / k
            if prob <= 0 or prob >= 1:
                h = 0.0
            else:
                h = -(prob * math.log2(prob) + (1 - prob) * math.log2(1 - prob))
            lines.append(f"{s},{p},{o},{c},{prob:.6f},{h}")
        (tdir / "caption_1_edges.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_lines = [
        "caption_id,caption_text,temperature,runs,n_unique_edges,global_entropy,mean_entropy_per_edge,n_predicate_disagreements,n_entity_variants,timestamp",
        f'1,"Synthetic: man by/next to door; desk competes.",0.7,{k},5,0.5,0.1,2,0,demo',
    ]
    (base / "summary.csv").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedy vs uncertainty-aware planning on predicate-disagreement cases from sampling outputs",
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default=None,
        help="Directory from run_triplet_sampling (contains temp_*/caption_*_edges.csv)",
    )
    parser.add_argument(
        "--synthetic_demo",
        action="store_true",
        help="Write a small synthetic output tree under thesis_mvp/outputs/planning_synthetic_demo and run experiments on it",
    )
    parser.add_argument(
        "--spatial_only",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Match search_planner spatial predicate filter (default true)",
    )
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--synonyms_file", type=str, default=None)
    parser.add_argument(
        "--eval_out",
        type=str,
        default=None,
        help="Subfolder name under outputs_dir for results (default: planning_eval)",
    )
    args = parser.parse_args()

    if args.synthetic_demo:
        repo_mvp = Path(__file__).resolve().parent
        outputs_dir = (repo_mvp / "outputs" / "planning_synthetic_demo").resolve()
        write_synthetic_outputs(outputs_dir, k=20)
        print(f"Wrote synthetic belief graphs under {outputs_dir}")
    elif args.outputs_dir:
        outputs_dir = Path(args.outputs_dir).resolve()
    else:
        print("Provide --outputs_dir or use --synthetic_demo", file=sys.stderr)
        sys.exit(1)

    if not outputs_dir.is_dir():
        print(f"Not a directory: {outputs_dir}", file=sys.stderr)
        sys.exit(1)

    eval_dir = outputs_dir / (args.eval_out or "planning_eval")
    spatial_only = args.spatial_only.lower() == "true"

    detail, aggregate = run_experiments(
        outputs_dir,
        spatial_only=spatial_only,
        max_steps=args.max_steps,
        synonyms_csv=args.synonyms_file,
    )

    detail_path = eval_dir / "planning_experiment_detail.csv"
    agg_path = eval_dir / "planning_experiment_aggregate.csv"
    eval_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False)
    aggregate.to_csv(agg_path, index=False)
    print(f"Wrote {detail_path} ({len(detail)} rows)")
    print(f"Wrote {agg_path}")

    meta = {
        "outputs_dir": str(outputs_dir),
        "spatial_only": spatial_only,
        "max_steps": args.max_steps,
        "n_detail_rows": len(detail),
        "n_temperatures": len(aggregate),
    }
    with open(eval_dir / "planning_experiment_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    plot_comparisons(detail, aggregate, eval_dir)
    for name in (
        "planning_mean_steps_by_temp.png",
        "planning_win_counts_stacked.png",
        "planning_step_delta_by_temp.png",
    ):
        p = eval_dir / name
        if p.exists():
            print(f"Wrote {p}")

    if detail.empty:
        print(
            "\nNo predicate-disagreement cases found in this output directory. "
            "Run triplet sampling at T>0 with more captions / K runs, then re-run.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
