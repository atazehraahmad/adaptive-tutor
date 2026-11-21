"""Authored question bank: retrieval, difficulty fallback, grading."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Option:
    text: str
    correct: bool = False
    misconception: str | None = None


@dataclass
class Question:
    id: str
    concept: str
    difficulty: int
    stem: str
    options: list[Option]
    explanation: str = ""

    @property
    def correct_index(self) -> int:
        return next(i for i, o in enumerate(self.options) if o.correct)

    @property
    def correct_text(self) -> str:
        return self.options[self.correct_index].text

    def grade(self, chosen: int) -> tuple[bool, str | None]:
        opt = self.options[chosen]
        return opt.correct, (None if opt.correct else opt.misconception)


def option_order(qid: str, student_id: str, n: int = 4) -> list[int]:
    """Stable per-student display order for a question's options.

    The authored bank lists the correct option first, so it must never be shown
    in authored order. Seeding on (student, question) keeps the arrangement
    identical across page reloads -- the choices must not move while the student
    is reading them -- while differing between students and questions.
    """
    idx = list(range(n))
    random.Random(f"{student_id}:{qid}").shuffle(idx)
    return idx


class QuestionBank:
    def __init__(self, questions: list[Question]):
        self.questions = questions
        self.by_concept: dict[str, list[Question]] = {}
        for q in questions:
            self.by_concept.setdefault(q.concept, []).append(q)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "QuestionBank":
        path = Path(path) if path else DATA / "question_bank.json"
        raw = json.loads(path.read_text())["questions"]
        return cls([
            Question(
                id=q["id"], concept=q["concept"], difficulty=q["difficulty"],
                stem=q["stem"], explanation=q.get("explanation", ""),
                options=[Option(**o) for o in q["options"]],
            ) for q in raw
        ])

    def get(self, qid: str) -> Question | None:
        return next((q for q in self.questions if q.id == qid), None)

    def pick(self, concept: str, difficulty: int = 2,
             exclude: set[str] | None = None,
             rng: random.Random | None = None) -> Question | None:
        """Closest-difficulty unseen item for a concept, else any unseen item."""
        rng = rng or random
        exclude = exclude or set()
        pool = [q for q in self.by_concept.get(concept, []) if q.id not in exclude]
        if not pool:
            pool = self.by_concept.get(concept, [])
            if not pool:
                return None
        best = min(abs(q.difficulty - difficulty) for q in pool)
        return rng.choice([q for q in pool if abs(q.difficulty - difficulty) == best])

    def coverage(self) -> dict[str, int]:
        return {c: len(v) for c, v in self.by_concept.items()}
