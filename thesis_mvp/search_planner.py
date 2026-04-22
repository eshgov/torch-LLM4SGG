"""
Language-guided object search planning over a belief graph (thesis MVP).

**Location marginal.** Fix target entity t. For each candidate location ℓ (object in a triplet),
aggregate belief mass M(ℓ) = sum_p P(t, p, ℓ) over edges, optionally restricted to spatial
predicates. Normalize to P̃(ℓ) = M(ℓ) / sum_{ℓ'} M(ℓ').

**Greedy baseline (paper):** At each decision step, choose ℓ* in argmax_ℓ P̃(ℓ) (myopic
maximum-likelihood location). After a failed query at ℓ, remove ℓ from the support, recompute
M and P̃ on the remaining locations, and repeat. Same replanning dynamics for both policies
below; only the argmax objective differs.

**Uncertainty-aware:** ℓ* = argmax_ℓ P̃(ℓ) · (H(p|t,ℓ) + δ), where H(p|t,ℓ) is the Shannon
entropy (bits) of the conditional predicate distribution from edges (t,·,ℓ), and δ>0 is a small
prior so ties degrade to near-greedy when all H=0. This prefers locations that are both
plausible and **predicate-split**, and can disagree with argmax P̃(ℓ).

The older variance rule argmax P̃(1−P̃) is kept as :func:`uncertainty_aware_variance_choice`.

**Predicate disagreement** cases are built in ``belief_graph.predicate_disagreement``;
``run_planning_experiments`` evaluates the two policies on those (s, o) scenarios.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple

from thesis_mvp.canonicalize import DEFAULT_SYNONYMS, canonicalize_entity

Triplet = Tuple[str, str, str]
EdgeProbs = Dict[Triplet, float]

# Predicates treated as "locating" the subject at/with respect to the object (open list).
DEFAULT_SPATIAL_TOKENS: Tuple[str, ...] = (
    "on",
    "in",
    "inside",
    "under",
    "near",
    "next to",
    "by",
    "beside",
    "along",
    "against",
    "into",
    "onto",
    "toward",
    "towards",
    "behind",
    "in front of",
    "above",
    "below",
    "between",
    "sitting on",
    "standing on",
    "lying on",
    "resting on",
    "attached to",
    "close to",
    "around",
)


def _merge_synonyms(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    merged = {k.lower(): v.lower() for k, v in DEFAULT_SYNONYMS.items()}
    if extra:
        for k, v in extra.items():
            merged[k.lower()] = v.lower()
    return merged


def canonicalize_target(target: str, synonyms: Optional[Dict[str, str]] = None) -> str:
    return canonicalize_entity(target, _merge_synonyms(synonyms), singularize=True)


def _predicate_is_spatial(pred: str, spatial_tokens: Iterable[str] = DEFAULT_SPATIAL_TOKENS) -> bool:
    """Multi-word phrases: substring match. Single-word tokens: whole-word match only."""
    p = pred.strip().lower()
    if not p:
        return False
    words = set(re.findall(r"[a-z0-9]+", p))
    for tok in spatial_tokens:
        t = tok.strip().lower()
        if not t:
            continue
        if " " in t:
            if t in p:
                return True
        else:
            if t in words:
                return True
    return False


def raw_location_masses(
    edge_probs: EdgeProbs,
    target_subject: str,
    spatial_only: bool = True,
    spatial_tokens: Tuple[str, ...] = DEFAULT_SPATIAL_TOKENS,
) -> Dict[str, float]:
    """
    Unnormalized location masses M(ℓ) = sum_p P(t, p, ℓ) for fixed target t = target_subject.
    If spatial_only, only predicates matching spatial_tokens contribute (same as thesis text).
    """
    masses: Dict[str, float] = {}
    for (s, p, o), prob in edge_probs.items():
        if s != target_subject:
            continue
        if spatial_only and not _predicate_is_spatial(p, spatial_tokens):
            continue
        masses[o] = masses.get(o, 0.0) + float(prob)
    return masses


def normalize_masses(masses: Dict[str, float]) -> Dict[str, float]:
    """P̃(ℓ) = M(ℓ) / sum M over remaining support."""
    total = sum(masses.values())
    if total <= 0:
        return {}
    return {loc: m / total for loc, m in masses.items()}


def location_marginal_P_tilde(
    edge_probs: EdgeProbs,
    target_subject: str,
    spatial_only: bool = True,
    spatial_tokens: Tuple[str, ...] = DEFAULT_SPATIAL_TOKENS,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Return (M, P̃): raw masses M(ℓ) and normalized location marginal P̃(ℓ) for target t.
    This is the distribution the greedy baseline argmax-es over at each step.
    """
    raw = raw_location_masses(edge_probs, target_subject, spatial_only, spatial_tokens)
    return raw, normalize_masses(raw)


def greedy_baseline_argmax_P_tilde(P_tilde: Dict[str, float]) -> Optional[str]:
    """
    Greedy baseline: ℓ* ∈ argmax_ℓ P̃(ℓ). Tie-break is deterministic (max over (P̃, ℓ) ordering).
    """
    if not P_tilde:
        return None
    return max(P_tilde.items(), key=lambda x: x[1])[0]


def greedy_choice(P_tilde: Dict[str, float]) -> Optional[str]:
    """Alias for :func:`greedy_baseline_argmax_P_tilde` (backward compatible)."""
    return greedy_baseline_argmax_P_tilde(P_tilde)


def predicate_entropy_bits_for_location(
    edge_probs: EdgeProbs,
    target_subject: str,
    location: str,
    spatial_only: bool,
    spatial_tokens: Tuple[str, ...] = DEFAULT_SPATIAL_TOKENS,
) -> float:
    """
    Entropy H(p | t, ℓ) in bits: normalize empirical masses over predicates p for edges (t, p, ℓ).
    Uses the same spatial filter as location marginals when spatial_only is True.
    """
    pred_mass: Dict[str, float] = {}
    for (s, p, o), prob in edge_probs.items():
        if s != target_subject or o != location:
            continue
        if spatial_only and not _predicate_is_spatial(p, spatial_tokens):
            continue
        pred_mass[p] = pred_mass.get(p, 0.0) + float(prob)
    tot = sum(pred_mass.values())
    if tot <= 0:
        return 0.0
    h = 0.0
    for m in pred_mass.values():
        q = m / tot
        if q > 0:
            h -= q * math.log2(q)
    return h


def uncertainty_aware_predicate_entropy_choice(
    P_tilde: Dict[str, float],
    edge_probs: EdgeProbs,
    target_subject: str,
    spatial_only: bool = True,
    delta: float = 0.02,
) -> Optional[str]:
    """
    Argmax_ℓ P̃(ℓ) · (H(p|t,ℓ) + δ). Tie-break: higher H, then higher P̃(ℓ).
    Can pick a lower-P̃ location if predicate entropy is high enough.
    """
    if not P_tilde:
        return None
    best_loc: Optional[str] = None
    best_score = -1.0
    best_h = -1.0
    best_p = -1.0
    for loc, ptilde in P_tilde.items():
        h_pred = predicate_entropy_bits_for_location(
            edge_probs, target_subject, loc, spatial_only
        )
        score = ptilde * (h_pred + delta)
        if score > best_score + 1e-15:
            best_score = score
            best_h = h_pred
            best_p = ptilde
            best_loc = loc
        elif abs(score - best_score) <= 1e-15:
            if h_pred > best_h + 1e-15 or (
                abs(h_pred - best_h) <= 1e-15 and ptilde > best_p
            ):
                best_score = score
                best_h = h_pred
                best_p = ptilde
                best_loc = loc
    return best_loc


def uncertainty_aware_variance_choice(P_tilde: Dict[str, float]) -> Optional[str]:
    """Argmax P̃(ℓ)(1 - P̃(ℓ)); tie-break by higher P̃. (Legacy heuristic.)"""
    if not P_tilde:
        return None
    best_loc: Optional[str] = None
    best_score = -1.0
    best_p = -1.0
    for loc, p in P_tilde.items():
        score = p * (1.0 - p)
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and p > best_p):
            best_score = score
            best_loc = loc
            best_p = p
    return best_loc


def uncertainty_aware_choice(
    P_tilde: Dict[str, float],
    edge_probs: EdgeProbs,
    target_subject: str,
    spatial_only: bool = True,
) -> Optional[str]:
    """Default UA policy: predicate-entropy-weighted marginal (see module doc)."""
    return uncertainty_aware_predicate_entropy_choice(
        P_tilde, edge_probs, target_subject, spatial_only=spatial_only
    )


def plan_locations(
    edge_probs: EdgeProbs,
    target: str,
    synonyms: Optional[Dict[str, str]] = None,
    spatial_only: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float], Optional[str], Optional[str]]:
    """
    Returns (M, P̃, greedy_baseline_ℓ, uncertainty_aware_ℓ) for canonicalized target.
    Greedy ℓ is argmax P̃ (baseline definition above).
    """
    subj = canonicalize_target(target, synonyms)
    raw, P_tilde = location_marginal_P_tilde(edge_probs, subj, spatial_only=spatial_only)
    return (
        raw,
        P_tilde,
        greedy_baseline_argmax_P_tilde(P_tilde),
        uncertainty_aware_choice(P_tilde, edge_probs, subj, spatial_only=spatial_only),
    )


def update_masses_after_reject(
    masses: Dict[str, float],
    rejected: str,
) -> Dict[str, float]:
    """Remove ℓ from mass dict and return normalized P̃ (helper; simulation uses raw M + normalize each step)."""
    out = {k: v for k, v in masses.items() if k != rejected}
    return normalize_masses(out)


@dataclass
class SearchSimulationResult:
    target: str
    ground_truth: str
    policy: str
    steps: int
    visited: List[str]
    success: bool


def simulate_search(
    edge_probs: EdgeProbs,
    target: str,
    ground_truth_location: str,
    policy: Literal["greedy", "uncertainty_aware"],
    synonyms: Optional[Dict[str, str]] = None,
    spatial_only: bool = True,
    max_steps: int = 50,
) -> SearchSimulationResult:
    """
    Sequential search with replanning: after each failed query, drop that ℓ from M(·),
    renormalize to P̃, and choose the next ℓ. Policy ``greedy`` uses argmax P̃ at every step;
    ``uncertainty_aware`` uses argmax P̃·(H_pred+δ) over predicate entropy at (t,ℓ).
    Success when chosen ℓ equals ground truth.
    """
    subj = canonicalize_target(target, synonyms)
    truth = canonicalize_target(ground_truth_location, synonyms)
    raw = raw_location_masses(edge_probs, subj, spatial_only=spatial_only)
    visited: List[str] = []

    for step in range(max_steps):
        P_tilde = normalize_masses(raw)
        if not P_tilde:
            return SearchSimulationResult(
                target=subj,
                ground_truth=truth,
                policy=policy,
                steps=step,
                visited=visited,
                success=False,
            )
        if policy == "greedy":
            loc = greedy_baseline_argmax_P_tilde(P_tilde)
        else:
            loc = uncertainty_aware_choice(P_tilde, edge_probs, subj, spatial_only=spatial_only)
        assert loc is not None
        visited.append(loc)
        if loc == truth:
            return SearchSimulationResult(
                target=subj,
                ground_truth=truth,
                policy=policy,
                steps=step + 1,
                visited=visited,
                success=True,
            )
        raw = {k: v for k, v in raw.items() if k != loc}

    return SearchSimulationResult(
        target=subj,
        ground_truth=truth,
        policy=policy,
        steps=len(visited),
        visited=visited,
        success=False,
    )


def compare_policies_simulation(
    edge_probs: EdgeProbs,
    target: str,
    ground_truth_location: str,
    synonyms: Optional[Dict[str, str]] = None,
    spatial_only: bool = True,
    max_steps: int = 50,
) -> Tuple[SearchSimulationResult, SearchSimulationResult, bool]:
    """
    Run greedy baseline (argmax P̃ each step) vs uncertainty-aware with identical elimination
    / replanning after failed queries. Returns (greedy_result, ua_result, first_choice_differs).
    """
    g = simulate_search(
        edge_probs,
        target,
        ground_truth_location,
        "greedy",
        synonyms=synonyms,
        spatial_only=spatial_only,
        max_steps=max_steps,
    )
    u = simulate_search(
        edge_probs,
        target,
        ground_truth_location,
        "uncertainty_aware",
        synonyms=synonyms,
        spatial_only=spatial_only,
        max_steps=max_steps,
    )
    first_diff = (g.visited[0] != u.visited[0]) if (g.visited and u.visited) else False
    return g, u, first_diff
