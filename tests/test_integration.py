import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import store
from engine.bank import QuestionBank
from engine.bkt import LearnerModel
from engine.graph import ConceptGraph
from engine.llm import TemplateBackend
from engine.misconception import MisconceptionTracker
from engine.selector import next_practice_item
from engine.srs import Scheduler

G, B, LLM = ConceptGraph.load(), QuestionBank.load(), TemplateBackend()


def test_bank_covers_every_concept_at_two_difficulties():
    cov = B.coverage()
    assert set(cov) == set(G.ids())
    assert all(n >= 2 for n in cov.values())


def test_every_distractor_is_tagged_or_deliberately_untagged():
    for q in B.questions:
        assert sum(o.correct for o in q.options) == 1, q.id
        for o in q.options:
            if o.misconception:
                assert o.misconception in G.misconceptions, q.id
                assert o.misconception in G.concepts[q.concept].misconceptions, q.id


def test_distractor_pick_is_detected_without_any_text():
    lm, tr = LearnerModel(G), MisconceptionTracker(G)
    q = B.get("Q053")                       # 6 and 3 ohm in parallel
    wrong = next(i for i, o in enumerate(q.options)
                 if o.misconception == "M_PARALLEL_ADDS")
    ok, mis = q.grade(wrong)
    assert not ok and mis == "M_PARALLEL_ADDS"
    fired = tr.record_attempt(q.concept, mis, [])
    assert fired == ["M_PARALLEL_ADDS"]


def test_right_answer_wrong_reasoning_is_caught():
    tr = MisconceptionTracker(G)
    q = B.get("Q053")
    ok, mis = q.grade(q.correct_index)
    assert ok and mis is None
    found = LLM.classify_reasoning("i just added them and got lucky", q.concept,
                                   G.concepts[q.concept].misconceptions)
    fired = tr.record_attempt(q.concept, None, found)
    assert "M_PARALLEL_ADDS" in fired, "reasoning signal should fire on a correct answer"


def test_repeated_misconception_becomes_confident():
    tr = MisconceptionTracker(G)
    for _ in range(4):
        tr.record_attempt("C27", "M_PARALLEL_ADDS", [])
    assert tr.state["M_PARALLEL_ADDS"].confident
    assert any(m == "M_PARALLEL_ADDS" for m, _ in tr.active())


def test_full_session_updates_and_persists():
    sid = "pytest_user"
    lm, tr, sc = LearnerModel(G), MisconceptionTracker(G), Scheduler()
    history, seen = [], set()
    for _ in range(15):
        cid, diff = next_practice_item(lm, [h["concept"] for h in history])
        q = B.pick(cid, diff, seen)
        assert q is not None
        seen.add(q.id)
        ok, mis = q.grade(q.correct_index)
        lm.observe(cid, ok, q.difficulty)
        tr.record_attempt(cid, mis, [])
        sc.record(cid, ok, lm.mastery[cid], date.today())
        history.append({"q": q.id, "concept": cid, "correct": ok})

    store.save(sid, {"id": sid, "name": "Pytest", "section": "x",
                     "model": lm.snapshot(), "misconceptions": tr.snapshot(),
                     "schedule": sc.snapshot(), "history": history})
    back = store.load(sid)
    assert back and len(back["history"]) == 15
    lm2 = LearnerModel.restore(G, back["model"])
    assert lm2.mastery == lm.mastery
    assert lm2.overall() > LearnerModel(G).overall(), "all-correct run should raise mastery"
    (store.STUDENTS / f"{sid}.json").unlink()


def test_selector_respects_prerequisites():
    lm = LearnerModel(G)
    picks = {next_practice_item(lm, [])[0] for _ in range(40)}
    deep = {"C14", "C15", "C18"}        # need long prerequisite chains
    assert not (picks & deep), f"selector offered un-ready concepts: {picks & deep}"


def test_mastering_prerequisites_unlocks_downstream():
    lm = LearnerModel(G)
    for cid in ("C01", "C02", "C04", "C05", "C03", "C08", "C12", "C13", "C06"):
        for _ in range(6):
            lm.observe(cid, True, 3)
    assert "C14" in G.frontier(lm.mastery), "lens formula should unlock once its chain is known"


def test_authored_bank_has_correct_answer_first_so_display_must_shuffle():
    """Documents why option_order exists: the raw bank is positionally biased."""
    first = sum(1 for q in B.questions if q.correct_index == 0)
    assert first == len(B.questions), "bank authored correct-first by construction"


def test_displayed_option_order_is_not_positionally_biased():
    from engine.bank import option_order

    positions = {0: 0, 1: 0, 2: 0, 3: 0}
    for sid in (f"u{i}" for i in range(40)):
        for q in B.questions:
            order = option_order(q.id, sid)
            positions[order.index(q.correct_index)] += 1
    total = sum(positions.values())
    for slot, n in positions.items():
        share = n / total
        assert 0.18 < share < 0.32, f"slot {slot} holds the answer {share:.0%} of the time"


def test_option_order_is_stable_for_one_student_and_question():
    from engine.bank import option_order
    assert option_order("Q001", "u1") == option_order("Q001", "u1")


def test_option_order_differs_across_students():
    from engine.bank import option_order
    orders = {tuple(option_order("Q001", f"u{i}")) for i in range(12)}
    assert len(orders) > 1, "every student would see the same arrangement"


def test_diagnostic_differs_between_students():
    import random
    from engine.selector import next_diagnostic_concept
    papers = set()
    for sid in range(8):
        lm, asked = LearnerModel(G), []
        for i in range(10):
            asked.append(next_diagnostic_concept(
                lm, asked, rng=random.Random(f"s{sid}:diag:{i}")))
        papers.add(tuple(asked))
    assert len(papers) > 1, "all students sat an identical diagnostic"
