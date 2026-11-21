"""Bayesian Knowledge Tracing with prerequisite propagation over a concept graph.

The learner model holds one mastery probability per concept. Every answer
produces a Bayesian posterior update on the concept the item targets, and a
damped propagation along prerequisite edges, so evidence at one node moves
belief about concepts the learner was never directly asked about.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .graph import ConceptGraph


@dataclass(frozen=True)
class BKTParams:
    p_init: float = 0.25      # prior mastery before any evidence
    p_transit: float = 0.12   # chance of learning from one attempt
    p_guess: float = 0.25     # 4-option MCQ floor
    p_slip: float = 0.10      # careless error on a known concept

    def calibrate(self, difficulty: int) -> tuple[float, float]:
        """Difficulty-adjusted (guess, slip).

        A harder item is easier to slip on and harder to fluke, so evidence
        from it is weighted differently. difficulty is 1 (easy) to 3 (hard).
        """
        d = max(1, min(3, int(difficulty))) - 1
        guess = self.p_guess * (1.0 - 0.30 * d)
        slip = self.p_slip * (1.0 + 0.55 * d)
        return _clip(guess, 0.02, 0.5), _clip(slip, 0.02, 0.5)


# propagation strengths
UP_SUCCESS_DECAY = 0.90   # per level, when success implies prerequisites known
UP_FAILURE_DECAY = 0.55   # per level, when failure casts doubt upstream
UP_FAILURE_WEIGHT = 0.60  # failure evidence is weaker than success evidence
DOWN_FAILURE_DECAY = 0.35 # a broken prerequisite drags dependents down slightly


def _clip(x: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, x))


@dataclass
class UpdateStep:
    """One line of the audit trail, so the UI can show its working."""
    concept: str
    before: float
    after: float
    reason: str

    @property
    def delta(self) -> float:
        return self.after - self.before


@dataclass
class UpdateTrace:
    concept: str
    correct: bool
    difficulty: int
    guess: float
    slip: float
    posterior: float
    steps: list[UpdateStep] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["steps"] = [asdict(s) | {"delta": s.delta} for s in self.steps]
        return d


class LearnerModel:
    def __init__(self, graph: ConceptGraph, params: BKTParams | None = None,
                 mastery: dict[str, float] | None = None):
        self.graph = graph
        self.params = params or BKTParams()
        self.mastery: dict[str, float] = {
            cid: self.params.p_init for cid in graph.ids()
        }
        if mastery:
            self.mastery.update({k: v for k, v in mastery.items() if k in self.mastery})
        self.attempts: dict[str, int] = dict.fromkeys(graph.ids(), 0)

    # -- core ---------------------------------------------------------
    def observe(self, concept: str, correct: bool, difficulty: int = 2) -> UpdateTrace:
        if concept not in self.mastery:
            raise KeyError(f"unknown concept {concept}")
        guess, slip = self.params.calibrate(difficulty)
        p = self.mastery[concept]

        # Bayes: P(knows | observation)
        if correct:
            num = p * (1 - slip)
            den = num + (1 - p) * guess
        else:
            num = p * slip
            den = num + (1 - p) * (1 - guess)
        posterior = _clip(num / den if den else p)

        # learning opportunity: the attempt itself may teach
        updated = _clip(posterior + (1 - posterior) * self.params.p_transit)

        trace = UpdateTrace(
            concept=concept, correct=correct, difficulty=difficulty,
            guess=round(guess, 3), slip=round(slip, 3), posterior=round(posterior, 4),
        )
        trace.steps.append(UpdateStep(
            concept, round(p, 4), round(updated, 4),
            f"direct evidence: {'correct' if correct else 'incorrect'} "
            f"on a difficulty-{difficulty} item",
        ))
        self.mastery[concept] = updated
        self.attempts[concept] += 1

        self._propagate(concept, updated, correct, trace)
        return trace

    # -- propagation ---------------------------------------------------
    def _propagate(self, concept: str, new_p: float, correct: bool,
                   trace: UpdateTrace) -> None:
        """Move belief along prerequisite edges.

        Success is the logically stronger signal: solving a problem about a
        concept is near-proof that its prerequisites are in place, so we raise
        prerequisites toward the new value. Failure is ambiguous (the gap could
        be local), so it is propagated upward only weakly, and a little way
        downward to concepts that depend on the shaky one.
        """
        name = self.graph.concepts[concept].name
        for ancestor, depth in self.graph.ancestors(concept):
            before = self.mastery[ancestor]
            if correct:
                implied = new_p * (UP_SUCCESS_DECAY ** depth)
                after = max(before, implied)
                if after - before > 1e-4:
                    trace.steps.append(UpdateStep(
                        ancestor, round(before, 4), round(after, 4),
                        f"solving \u201c{name}\u201d implies this prerequisite "
                        f"is in place (depth {depth})",
                    ))
                    self.mastery[ancestor] = after
            else:
                drop = (1 - new_p) * UP_FAILURE_WEIGHT * (UP_FAILURE_DECAY ** depth)
                after = _clip(before * (1 - drop))
                if before - after > 1e-4:
                    trace.steps.append(UpdateStep(
                        ancestor, round(before, 4), round(after, 4),
                        f"the gap on \u201c{name}\u201d may originate here "
                        f"(depth {depth})",
                    ))
                    self.mastery[ancestor] = after

        if not correct:
            for child, depth in self.graph.descendants(concept):
                before = self.mastery[child]
                drop = (1 - new_p) * 0.25 * (DOWN_FAILURE_DECAY ** depth)
                after = _clip(before * (1 - drop))
                if before - after > 1e-4:
                    trace.steps.append(UpdateStep(
                        child, round(before, 4), round(after, 4),
                        f"this builds on \u201c{name}\u201d (depth {depth})",
                    ))
                    self.mastery[child] = after

    # -- reporting -----------------------------------------------------
    def weakest(self, n: int = 5, only_ready: bool = True) -> list[tuple[str, float]]:
        pool = self.graph.frontier(self.mastery) if only_ready else self.graph.ids()
        if not pool:
            pool = self.graph.ids()
        return sorted(((c, self.mastery[c]) for c in pool), key=lambda kv: kv[1])[:n]

    def uncertainty(self, cid: str) -> float:
        """Shannon-style uncertainty; peaks at p = 0.5, where a question tells us most."""
        p = self.mastery[cid]
        return 4 * p * (1 - p)

    def chapter_mastery(self) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for cid, c in self.graph.concepts.items():
            out.setdefault(c.chapter, []).append(self.mastery[cid])
        return {k: sum(v) / len(v) for k, v in out.items()}

    def overall(self) -> float:
        return sum(self.mastery.values()) / len(self.mastery)

    def snapshot(self) -> dict:
        return {"mastery": dict(self.mastery), "attempts": dict(self.attempts)}

    @classmethod
    def restore(cls, graph: ConceptGraph, snap: dict,
                params: BKTParams | None = None) -> "LearnerModel":
        lm = cls(graph, params, snap.get("mastery"))
        lm.attempts.update(snap.get("attempts", {}))
        return lm
