"""Concept graph: loading, prerequisite traversal, readiness."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Concept:
    id: str
    name: str
    chapter: str
    prereq: list[str] = field(default_factory=list)
    misconceptions: list[str] = field(default_factory=list)


@dataclass
class Misconception:
    id: str
    label: str
    remedy: str


class ConceptGraph:
    def __init__(self, payload: dict):
        self.syllabus: str = payload["syllabus"]
        self.concepts: dict[str, Concept] = {
            c["id"]: Concept(**c) for c in payload["concepts"]
        }
        self.misconceptions: dict[str, Misconception] = {
            m["id"]: Misconception(**m) for m in payload["misconceptions"]
        }
        self._dependents: dict[str, list[str]] = {cid: [] for cid in self.concepts}
        for c in self.concepts.values():
            for p in c.prereq:
                self._dependents[p].append(c.id)
        self._validate()

    # -- construction -------------------------------------------------
    @classmethod
    def load(cls, path: Path | str | None = None) -> "ConceptGraph":
        path = Path(path) if path else DATA / "concept_graph.json"
        return cls(json.loads(path.read_text()))

    def _validate(self) -> None:
        for c in self.concepts.values():
            for p in c.prereq:
                if p not in self.concepts:
                    raise ValueError(f"{c.id} has unknown prerequisite {p}")
            for m in c.misconceptions:
                if m not in self.misconceptions:
                    raise ValueError(f"{c.id} has unknown misconception {m}")
        if self._has_cycle():
            raise ValueError("concept graph contains a prerequisite cycle")

    def _has_cycle(self) -> bool:
        WHITE, GREY, BLACK = 0, 1, 2
        colour = dict.fromkeys(self.concepts, WHITE)

        def visit(node: str) -> bool:
            colour[node] = GREY
            for nxt in self._dependents[node]:
                if colour[nxt] == GREY:
                    return True
                if colour[nxt] == WHITE and visit(nxt):
                    return True
            colour[node] = BLACK
            return False

        return any(colour[n] == WHITE and visit(n) for n in self.concepts)

    # -- traversal ----------------------------------------------------
    def ids(self) -> list[str]:
        return list(self.concepts)

    def chapters(self) -> list[str]:
        seen: list[str] = []
        for c in self.concepts.values():
            if c.chapter not in seen:
                seen.append(c.chapter)
        return seen

    def dependents(self, cid: str) -> list[str]:
        return list(self._dependents[cid])

    @lru_cache(maxsize=None)
    def ancestors(self, cid: str) -> tuple[tuple[str, int], ...]:
        """Prerequisites reachable upstream, paired with their depth (1 = direct)."""
        out: dict[str, int] = {}
        frontier = [(p, 1) for p in self.concepts[cid].prereq]
        while frontier:
            node, depth = frontier.pop()
            if node in out and out[node] <= depth:
                continue
            out[node] = depth
            frontier.extend((p, depth + 1) for p in self.concepts[node].prereq)
        return tuple(sorted(out.items(), key=lambda kv: kv[1]))

    @lru_cache(maxsize=None)
    def descendants(self, cid: str) -> tuple[tuple[str, int], ...]:
        out: dict[str, int] = {}
        frontier = [(d, 1) for d in self._dependents[cid]]
        while frontier:
            node, depth = frontier.pop()
            if node in out and out[node] <= depth:
                continue
            out[node] = depth
            frontier.extend((d, depth + 1) for d in self._dependents[node])
        return tuple(sorted(out.items(), key=lambda kv: kv[1]))

    def topo_order(self) -> list[str]:
        indeg = {cid: len(c.prereq) for cid, c in self.concepts.items()}
        queue = [cid for cid, d in indeg.items() if d == 0]
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for nxt in self._dependents[node]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        return order

    # -- pedagogy -----------------------------------------------------
    def readiness(self, cid: str, mastery: dict[str, float]) -> float:
        """How prepared the learner is to be taught this concept at all.

        A concept with unmet prerequisites should not be drilled; the gap
        upstream has to be closed first. Root concepts are always ready.
        """
        prereq = self.concepts[cid].prereq
        if not prereq:
            return 1.0
        return sum(mastery.get(p, 0.0) for p in prereq) / len(prereq)

    def frontier(self, mastery: dict[str, float], ready_at: float = 0.55,
                 known_at: float = 0.80) -> list[str]:
        """Concepts the learner is ready for but has not yet mastered."""
        return [
            cid for cid in self.topo_order()
            if mastery.get(cid, 0.0) < known_at
            and self.readiness(cid, mastery) >= ready_at
        ]
