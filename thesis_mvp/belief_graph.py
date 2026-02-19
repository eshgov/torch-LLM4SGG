"""
Belief graph: from K runs of triplet extraction, compute edge counts,
probabilities P(e)=count/K, and entropy H(e) = -P*log(P) - (1-P)*log(1-P).
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd


def build_belief_graph(
    run_triplets: List[List[Tuple[str, str, str]]], k: int
) -> Tuple[Dict[Tuple[str, str, str], int], Dict[Tuple[str, str, str], float], Dict[Tuple[str, str, str], float]]:
    """
    run_triplets: list of K lists, each list is the triplets from one run.
    k: number of runs (for P(e) = count/K).

    Returns:
        edge_counts: (s, p, o) -> count
        edge_probs: (s, p, o) -> P(e)
        edge_entropies: (s, p, o) -> H(e)
    """
    edge_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for triplets in run_triplets:
        for t in triplets:
            edge_counts[t] += 1

    edge_probs: Dict[Tuple[str, str, str], float] = {}
    edge_entropies: Dict[Tuple[str, str, str], float] = {}
    for e, count in edge_counts.items():
        p = count / k
        edge_probs[e] = p
        # Binary entropy: H(p) = -p*log2(p) - (1-p)*log2(1-p); use natural log if preferred, same ranking
        if p <= 0 or p >= 1:
            h = 0.0
        else:
            h = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        edge_entropies[e] = h

    return dict(edge_counts), edge_probs, edge_entropies


def global_entropy(edge_entropies: Dict[Tuple[str, str, str], float]) -> float:
    """Sum of H(e) over all edges."""
    return sum(edge_entropies.values())


def predicate_disagreement(
    edge_counts: Dict[Tuple[str, str, str], int]
) -> List[Tuple[Tuple[str, str], List[Tuple[str, int]]]]:
    """
    Same (subject, object) with multiple predicates. Returns list of
    ((subject, object), [(predicate, count), ...]) for pairs with >1 predicate.
    """
    so_to_preds: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (s, p, o), count in edge_counts.items():
        so_to_preds[(s, o)][p] = count

    out = []
    for (s, o), pred_counts in so_to_preds.items():
        if len(pred_counts) <= 1:
            continue
        out.append(((s, o), sorted(pred_counts.items(), key=lambda x: -x[1])))
    return out


def to_dataframe(
    edge_counts: Dict[Tuple[str, str, str], int],
    edge_probs: Dict[Tuple[str, str, str], float],
    edge_entropies: Dict[Tuple[str, str, str], float],
) -> pd.DataFrame:
    """One row per edge: subject, predicate, object, count, probability, entropy."""
    rows = []
    for (s, p, o) in edge_counts:
        rows.append({
            "subject": s,
            "predicate": p,
            "object": o,
            "count": edge_counts[(s, p, o)],
            "probability": edge_probs[(s, p, o)],
            "entropy": edge_entropies[(s, p, o)],
        })
    if not rows:
        return pd.DataFrame(columns=["subject", "predicate", "object", "count", "probability", "entropy"])
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def summary_stats_per_caption(
    edge_counts: Dict[Tuple[str, str, str], int],
    edge_entropies: Dict[Tuple[str, str, str], float],
    k: int,
) -> Dict[str, float]:
    """Mean #unique edges, mean entropy, etc. for one caption."""
    n_edges = len(edge_counts)
    total_entropy = sum(edge_entropies.values())
    return {
        "n_unique_edges": n_edges,
        "global_entropy": total_entropy,
        "mean_entropy_per_edge": total_entropy / n_edges if n_edges else 0.0,
        "k": k,
    }


def aggregate_summary(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Across captions: mean of n_unique_edges, mean global_entropy, etc."""
    if not stats_list:
        return {}
    n = len(stats_list)
    return {
        "mean_n_unique_edges": sum(s["n_unique_edges"] for s in stats_list) / n,
        "mean_global_entropy": sum(s["global_entropy"] for s in stats_list) / n,
        "mean_entropy_per_edge": sum(s["mean_entropy_per_edge"] for s in stats_list) / n,
        "n_captions": n,
    }
