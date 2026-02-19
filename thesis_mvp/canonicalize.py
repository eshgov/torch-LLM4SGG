"""
Lightweight canonicalization for subject/object and normalization for predicates
before counting edges. Reduces artificial disagreement from "phone" vs "cell phone", etc.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

# Hardcoded synonym map (entity terms): from -> to (lowercase)
DEFAULT_SYNONYMS: Dict[str, str] = {
    "cell phone": "phone",
    "cellphone": "phone",
    "mobile phone": "phone",
    "sofa": "couch",
    "tv": "television",
    "telly": "television",
    "woman's feet": "feet",
    "womens feet": "feet",
    "man's feet": "feet",
    "mens feet": "feet",
    "wooden cabinets": "cabinets",
    "white blanket": "blanket",
    "cup of coffee": "coffee",
    "vase of flowers": "vase",
    "double bed": "bed",
    "park bench": "bench",
    "coffee table": "table",
    "nightstands": "nightstand",
    "night stands": "nightstand",
}

# Simple plural -> singular (safe set only; don't aggressive stem)
SIMPLE_PLURALS: Dict[str, str] = {
    "clocks": "clock",
    "men": "man",
    "women": "woman",
    "feet": "foot",
    "pillows": "pillow",
    "plates": "plate",
    "forks": "fork",
    "toys": "toy",
    "curtains": "curtain",
    "ducks": "duck",
    "kites": "kite",
}

AUXILIARY_PREFIX = re.compile(
    r"^\s*(is|are|was|were)\s+",
    re.IGNORECASE,
)


def _collapse_whitespace(s: str) -> str:
    return " ".join(s.split()).strip()


def _strip_determiners(s: str) -> str:
    s = s.strip()
    for prefix in ("a ", "an ", "the "):
        if s.lower().startswith(prefix):
            s = s[len(prefix) :].strip()
            break
    return s


def canonicalize_entity(
    token: str,
    synonyms: Dict[str, str],
    singularize: bool = True,
) -> str:
    """
    Lowercase, strip determiners, collapse whitespace, optional synonym map,
    optional simple plural -> singular.
    """
    if not token or not token.strip():
        return token
    s = token.strip().lower()
    s = _strip_determiners(s)
    s = _collapse_whitespace(s)
    if s in synonyms:
        s = synonyms[s]
    if singularize and s in SIMPLE_PLURALS:
        s = SIMPLE_PLURALS[s]
    return _collapse_whitespace(s)


def normalize_predicate(pred: str) -> str:
    """Lowercase, strip leading is/are/was/were, collapse whitespace."""
    if not pred or not pred.strip():
        return pred
    s = pred.strip().lower()
    s = _collapse_whitespace(s)
    s = AUXILIARY_PREFIX.sub("", s).strip()
    return _collapse_whitespace(s)


def load_synonyms_file(path: str) -> Dict[str, str]:
    """Load 'from,to' CSV; keys/values lowercased."""
    out: Dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return out
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",", 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                out[parts[0].lower()] = parts[1].lower()
    return out


def canonicalize_triplets(
    run_triplets: List[List[Tuple[str, str, str]]],
    synonyms: Dict[str, str],
    singularize: bool = True,
) -> Tuple[List[List[Tuple[str, str, str]]], int]:
    """
    Apply entity canonicalization and predicate normalization to all triplets.
    Returns (canonicalized run_triplets, n_entity_variants) where n_entity_variants
    is the number of subject or object slots that were changed (approximate).
    """
    merged = {k.lower(): v.lower() for k, v in DEFAULT_SYNONYMS.items()}
    for k, v in synonyms.items():
        merged[k.lower()] = v.lower()

    out_runs: List[List[Tuple[str, str, str]]] = []
    n_changed = 0
    for triplets in run_triplets:
        new_triplets = []
        for s, p, o in triplets:
            s_c = canonicalize_entity(s, merged, singularize)
            o_c = canonicalize_entity(o, merged, singularize)
            p_c = normalize_predicate(p)
            if s != s_c:
                n_changed += 1
            if o != o_c:
                n_changed += 1
            new_triplets.append((s_c, p_c, o_c))
        out_runs.append(new_triplets)
    return out_runs, n_changed
