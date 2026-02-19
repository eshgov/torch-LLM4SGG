"""
Robust parsing of LLM triplet output into (subject, predicate, object) tuples.
Handles angle-bracket format <subj, pred, obj> and bullet/list formats.
"""

import re
from typing import List, Tuple


def normalize_whitespace(s: str) -> str:
    """Collapse internal whitespace and strip."""
    return " ".join(s.split()).strip()


def parse_triplets_from_response(response: str) -> List[Tuple[str, str, str]]:
    """
    Parse raw LLM response into list of (subject, predicate, object) triplets.
    Tries angle-bracket pattern first, then fallback patterns (bullets, numbered).
    """
    triplets = []
    response = response.strip()

    # --- 1. Angle-bracket format: <subject, predicate, object> or <subject,predicate,object>
    # Match < ... , ... , ... > with possible newlines; be careful with commas inside
    angle_pattern = re.compile(
        r"<\s*([^,<>]+?)\s*,\s*([^,<>]+?)\s*,\s*([^<>]+?)\s*>",
        re.IGNORECASE | re.DOTALL,
    )
    for m in angle_pattern.finditer(response):
        subj = normalize_whitespace(m.group(1))
        pred = normalize_whitespace(m.group(2))
        obj = normalize_whitespace(m.group(3))
        if subj and pred and obj:
            triplets.append((subj, pred, obj))

    if triplets:
        return _dedupe_triplets(triplets)

    # --- 2. "triplet(s) are" / "triplets:" then list of <s, p, o> or "s, p, o"
    # After "Answer:" take the rest and try again with relaxed angle pattern
    if "Answer:" in response:
        answer_part = response.split("Answer:")[-1].strip()
        for m in angle_pattern.finditer(answer_part):
            subj = normalize_whitespace(m.group(1))
            pred = normalize_whitespace(m.group(2))
            obj = normalize_whitespace(m.group(3))
            if subj and pred and obj:
                triplets.append((subj, pred, obj))
        if triplets:
            return _dedupe_triplets(triplets)

        # Bullet-style in answer: "- subject, predicate, object" or "• subject, predicate, object"
        line_pattern = re.compile(
            r"[\-\•\*]\s*([^,\n]+),\s*([^,\n]+),\s*([^\n]+)",
            re.IGNORECASE,
        )
        for m in line_pattern.finditer(answer_part):
            subj = normalize_whitespace(m.group(1))
            pred = normalize_whitespace(m.group(2))
            obj = normalize_whitespace(m.group(3))
            if subj and pred and obj:
                triplets.append((subj, pred, obj))

    if triplets:
        return _dedupe_triplets(triplets)

    # --- 3. Any line containing exactly two commas (s, p, o)
    lines = response.replace("Answer:", "\n").split("\n")
    two_comma = re.compile(r"^[\s\-\•\*]*([^,]+),\s*([^,]+),\s*([^,]+)\s*$")
    for line in lines:
        line = line.strip()
        m = two_comma.match(line)
        if m:
            subj = normalize_whitespace(m.group(1))
            pred = normalize_whitespace(m.group(2))
            obj = normalize_whitespace(m.group(3))
            if subj and pred and obj and not subj.startswith("<"):
                triplets.append((subj, pred, obj))

    return _dedupe_triplets(triplets)


def _dedupe_triplets(triplets: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """Return unique triplets preserving order of first occurrence."""
    seen = set()
    out = []
    for t in triplets:
        key = (t[0].lower(), t[1].lower(), t[2].lower())
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out
