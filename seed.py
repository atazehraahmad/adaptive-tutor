"""Generate a synthetic class by running simulated learners through the real pipeline.

Nothing here writes mastery numbers directly. Each synthetic student has a hidden
true ability and a set of misconceptions they actually hold; they answer items
sampled from the same bank, and their stored profile is whatever the learner
model infers. The teacher dashboard is therefore showing genuine model output.
"""
from __future__ import annotations

import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.bank import QuestionBank
from engine.bkt import LearnerModel
from engine.graph import ConceptGraph
from engine.misconception import MisconceptionTracker
from engine.selector import next_practice_item, success_probability
from engine.srs import Scheduler
from engine import store

FIRST = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Ananya", "Vihaan", "Ishita",
         "Arjun", "Sara", "Neel", "Tara", "Dev", "Riya", "Aditya", "Nisha",
         "Kiran", "Zoya", "Varun", "Leela", "Omkar", "Pari", "Yash", "Anika",
         "Rahul", "Juhi", "Siddharth", "Maya", "Nikhil", "Trisha"]
LAST = ["Sharma", "Iyer", "Khan", "Patel", "Nair", "Bose", "Reddy", "Gupta",
        "Menon", "Shah", "Joshi", "Das", "Rao", "Verma", "Pillai"]


def simulate(seed: int, graph: ConceptGraph, bank: QuestionBank,
             n_items: int | None = None) -> dict:
    rng = random.Random(seed)
    # engagement varies: some students have practised far more than others
    n_items = n_items if n_items is not None else rng.randint(40, 95)
    name = f"{FIRST[seed % len(FIRST)]} {LAST[(seed * 7) % len(LAST)]}"
    sid = f"s{seed:03d}"
    section = "10-A" if seed % 2 == 0 else "10-B"

    # hidden ground truth
    ability = min(0.95, max(0.08, rng.gauss(0.5, 0.2)))
    true_mastery = {
        cid: min(0.97, max(0.03, ability + rng.gauss(0, 0.14)))
        for cid in graph.ids()
    }
    # a few chapters are genuinely weak for this student
    for ch in rng.sample(graph.chapters(), k=rng.choice([1, 2])):
        for cid, c in graph.concepts.items():
            if c.chapter == ch:
                true_mastery[cid] = max(0.03, true_mastery[cid] - rng.uniform(0.2, 0.4))
    held = set(rng.sample(list(graph.misconceptions), k=rng.randint(1, 4)))

    lm = LearnerModel(graph)
    tracker = MisconceptionTracker(graph)
    sched = Scheduler()
    history: list[dict] = []
    seen: set[str] = set()
    recent: list[str] = []
    day = date.today() - timedelta(days=28)

    for i in range(n_items):
        pick = next_practice_item(lm, recent, rng)
        if not pick:
            break
        concept, difficulty = pick
        q = bank.pick(concept, difficulty, seen, rng)
        if q is None:
            recent.append(concept)
            continue
        seen.add(q.id)
        recent.append(concept)

        p = success_probability(true_mastery[concept], q.difficulty)
        # a held misconception makes an error far likelier on items that probe it
        probes = {o.misconception for o in q.options if o.misconception}
        if probes & held:
            p *= 0.45
        correct = rng.random() < p

        if correct:
            chosen = q.correct_index
        else:
            trap = [i for i, o in enumerate(q.options)
                    if o.misconception in held and not o.correct]
            wrong = [i for i, o in enumerate(q.options) if not o.correct]
            chosen = rng.choice(trap) if trap else rng.choice(wrong)

        ok, mis = q.grade(chosen)
        lm.observe(concept, ok, q.difficulty)
        tracker.record_attempt(concept, mis, [], list(probes))
        if i % max(2, n_items // 24) == 0:
            day += timedelta(days=1)
        sched.record(concept, ok, lm.mastery[concept], day)
        history.append({"q": q.id, "concept": concept, "correct": ok,
                        "difficulty": q.difficulty, "misconception": mis,
                        "date": day.isoformat()})

    return {
        "id": sid, "name": name, "section": section, "synthetic": True,
        "model": lm.snapshot(), "misconceptions": tracker.snapshot(),
        "schedule": sched.snapshot(), "history": history,
        "held_misconceptions": sorted(held),
    }


def main(n: int = 30) -> None:
    graph, bank = ConceptGraph.load(), QuestionBank.load()
    out = store.DATA / "students"
    for f in out.glob("s*.json"):
        f.unlink()
    for i in range(1, n + 1):
        rec = simulate(i, graph, bank)
        store.save(rec["id"], rec)
    print(f"seeded {n} synthetic students into {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
