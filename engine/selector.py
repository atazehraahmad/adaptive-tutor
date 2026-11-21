"""Adaptive item selection.

Diagnostic mode maximises information gain: ask about the concept we are least
certain of, weighted by how much of the syllabus depends on it. Practice mode
targets the weakest concept the learner is actually ready for, at a difficulty
that should yield roughly a 70% success rate, which is where learning is
fastest and motivation survives.
"""
from __future__ import annotations

import math
import random

from .bkt import LearnerModel
from .graph import ConceptGraph

TARGET_SUCCESS = 0.70


def centrality(graph: ConceptGraph, cid: str) -> float:
    """Worth of resolving this node: how much downstream syllabus rests on it."""
    return 1.0 + math.log1p(len(graph.descendants(cid)))


def information_value(lm: LearnerModel, cid: str) -> float:
    return lm.uncertainty(cid) * centrality(lm.graph, cid)


def success_probability(mastery: float, difficulty: int) -> float:
    """Expected P(correct) for a learner at `mastery` on an item of `difficulty`.

    Each difficulty has its own guess floor and its own exponent: a hard item
    demands disproportionately more mastery before it is reliably solved.
    """
    floor, k = {1: (0.30, 0.70), 2: (0.25, 1.20), 3: (0.20, 2.00)}[
        max(1, min(3, int(difficulty)))
    ]
    return floor + (1.0 - floor) * (max(0.0, min(1.0, mastery)) ** k)


def choose_difficulty(mastery: float) -> int:
    """Difficulty whose expected success rate sits nearest the target."""
    return min((1, 2, 3), key=lambda d: abs(success_probability(mastery, d) - TARGET_SUCCESS))


def next_diagnostic_concept(lm: LearnerModel, asked: list[str],
                            spread_chapters: bool = True,
                            rng: random.Random | None = None) -> str | None:
    """Highest-information unasked concept, nudged to cover chapters evenly.

    With an rng, the pick is sampled from the top few near-equal candidates, so
    two students do not sit an identical paper while the items stay informative.
    """
    pool = [c for c in lm.graph.ids() if c not in asked]
    if not pool:
        return None
    seen_chapters = [lm.graph.concepts[c].chapter for c in asked]
    counts = {ch: seen_chapters.count(ch) for ch in lm.graph.chapters()}

    def score(cid: str) -> float:
        s = information_value(lm, cid)
        if spread_chapters:
            s *= 1.0 / (1.0 + counts[lm.graph.concepts[cid].chapter])
        return s

    if rng is None:
        return max(pool, key=score)
    ranked = sorted(pool, key=score, reverse=True)
    top = ranked[:3]
    weights = [score(c) for c in top]
    total = sum(weights)
    if total <= 0:
        return rng.choice(top)
    return rng.choices(top, weights=weights, k=1)[0]


def next_practice_item(lm: LearnerModel, recent: list[str] | None = None,
                       rng: random.Random | None = None) -> tuple[str, int] | None:
    """Weakest ready concept, avoiding immediate repetition, plus a difficulty."""
    rng = rng or random
    recent = recent or []
    candidates = lm.graph.frontier(lm.mastery) or lm.graph.ids()
    fresh = [c for c in candidates if c not in recent[-3:]] or candidates
    ranked = sorted(fresh, key=lambda c: (lm.mastery[c], -centrality(lm.graph, c)))
    # soft pick from the three weakest so a session does not feel robotic
    cid = rng.choice(ranked[:3]) if len(ranked) >= 3 else ranked[0]
    return cid, choose_difficulty(lm.mastery[cid])


def diagnostic_plan(lm: LearnerModel, n: int = 10) -> list[str]:
    """Pre-compute a diagnostic sequence against the prior (no answers yet)."""
    asked: list[str] = []
    for _ in range(n):
        nxt = next_diagnostic_concept(lm, asked)
        if nxt is None:
            break
        asked.append(nxt)
    return asked
