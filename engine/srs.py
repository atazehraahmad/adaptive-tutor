"""Spaced repetition: an SM-2 variant coupled to the learner model.

Standard SM-2 schedules on recall quality alone. Here the interval is also
scaled by mastery, so a concept the model believes is shaky comes back sooner
even after a lucky correct answer, and a well-established concept stretches out
faster.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, timedelta

MIN_EASE, MAX_EASE = 1.3, 2.8


@dataclass
class Card:
    concept: str
    ease: float = 2.3
    interval: int = 0          # days
    reps: int = 0
    lapses: int = 0
    due: str = ""              # ISO date

    def is_due(self, today: date | None = None) -> bool:
        if not self.due:
            return True
        return date.fromisoformat(self.due) <= (today or date.today())

    def days_until_due(self, today: date | None = None) -> int:
        if not self.due:
            return 0
        return (date.fromisoformat(self.due) - (today or date.today())).days


def review(card: Card, correct: bool, mastery: float,
           today: date | None = None) -> Card:
    today = today or date.today()
    # quality 0-5, blending the outcome with how sure the model is
    quality = (3.0 + 2.0 * mastery) if correct else (1.5 * mastery)

    if correct:
        card.reps += 1
        if card.reps == 1:
            # a card relearned after a lapse need not restart from one day if
            # the model is confident the concept is actually understood
            base = 2 if mastery >= 0.5 else 1
        elif card.reps == 2:
            base = 3
        else:
            base = round(card.interval * card.ease)
        # mastery stretches or compresses the gap
        card.interval = max(1, round(base * (0.55 + 0.9 * mastery)))
    else:
        card.reps = 0
        card.lapses += 1
        card.interval = 1

    card.ease = min(MAX_EASE, max(
        MIN_EASE,
        card.ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)),
    ))
    card.due = (today + timedelta(days=card.interval)).isoformat()
    return card


class Scheduler:
    def __init__(self, cards: dict[str, dict] | None = None):
        self.cards: dict[str, Card] = {
            k: Card(**v) for k, v in (cards or {}).items()
        }

    def card(self, concept: str) -> Card:
        return self.cards.setdefault(concept, Card(concept=concept))

    def record(self, concept: str, correct: bool, mastery: float,
               today: date | None = None) -> Card:
        return review(self.card(concept), correct, mastery, today)

    def due_now(self, today: date | None = None) -> list[Card]:
        return sorted(
            (c for c in self.cards.values() if c.is_due(today)),
            key=lambda c: (c.due or "", -c.lapses),
        )

    def upcoming(self, today: date | None = None) -> list[Card]:
        return sorted(
            (c for c in self.cards.values() if not c.is_due(today)),
            key=lambda c: c.due,
        )

    def snapshot(self) -> dict:
        return {k: asdict(v) for k, v in self.cards.items()}
