"""Misconception detection and tracking.

Two independent signals feed one posterior per misconception:

1. Structural. Every distractor in the bank is authored against a named
   misconception, so selecting it is direct evidence that needs no parsing.
2. Reasoning. The learner's free-text explanation is classified against the
   misconception taxonomy for that concept by the configured analyser.

The structural signal is cheap and always available; the reasoning signal adds
depth and catches a learner who picked the right answer for the wrong reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .graph import ConceptGraph

PRIOR_A, PRIOR_B = 1.0, 2.0   # weak Beta prior: misconceptions are not assumed


@dataclass
class MisconceptionState:
    hits: float = 0.0         # evidence the learner holds it
    opportunities: float = 0.0  # times it could have shown and did not
    last_source: str = ""

    @property
    def prevalence(self) -> float:
        a = PRIOR_A + self.hits
        b = PRIOR_B + max(0.0, self.opportunities - self.hits)
        return a / (a + b)

    @property
    def confident(self) -> bool:
        return self.hits >= 2 and self.prevalence > 0.50


@dataclass
class MisconceptionTracker:
    graph: ConceptGraph
    state: dict[str, MisconceptionState] = field(default_factory=dict)

    def _get(self, mid: str) -> MisconceptionState:
        return self.state.setdefault(mid, MisconceptionState())

    def record_attempt(self, concept: str, chosen_misconception: str | None,
                       reasoning_misconceptions: list[str] | None = None,
                       available: list[str] | None = None) -> list[str]:
        """Log one attempt. Returns the misconception ids that fired.

        `available` is the set of misconceptions this particular question could
        actually reveal, i.e. those attached to its distractors. Only those count
        as an opportunity: a question that never offered the trap is no evidence
        that the student has avoided it.
        """
        fired: list[str] = []
        probes = (available if available is not None
                  else self.graph.concepts[concept].misconceptions)
        for mid in probes:
            self._get(mid).opportunities += 1

        if chosen_misconception:
            st = self._get(chosen_misconception)
            st.hits += 1.0
            st.last_source = "distractor"
            fired.append(chosen_misconception)

        for mid in reasoning_misconceptions or []:
            st = self._get(mid)
            # reasoning evidence is strong but not certain; weight below a
            # distractor pick unless it corroborates one
            st.hits += 0.7 if mid not in fired else 0.3
            st.last_source = "reasoning" if mid not in fired else "both"
            if mid not in fired:
                fired.append(mid)
        return fired

    def active(self, threshold: float = 0.35) -> list[tuple[str, float]]:
        out = [(mid, st.prevalence) for mid, st in self.state.items()
               if st.hits > 0 and st.prevalence >= threshold]
        return sorted(out, key=lambda kv: -kv[1])

    def label(self, mid: str) -> str:
        return self.graph.misconceptions[mid].label

    def remedy(self, mid: str) -> str:
        return self.graph.misconceptions[mid].remedy

    def snapshot(self) -> dict:
        return {k: vars(v) for k, v in self.state.items()}

    @classmethod
    def restore(cls, graph: ConceptGraph, snap: dict) -> "MisconceptionTracker":
        t = cls(graph)
        for k, v in (snap or {}).items():
            t.state[k] = MisconceptionState(**v)
        return t
