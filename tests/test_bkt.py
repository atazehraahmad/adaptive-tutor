import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.graph import ConceptGraph
from engine.bkt import LearnerModel, BKTParams

G = ConceptGraph.load()


def fresh(**kw):
    return LearnerModel(G, BKTParams(**kw))


def test_posterior_matches_hand_calculation():
    lm = fresh(p_init=0.40, p_transit=0.0, p_guess=0.25, p_slip=0.10)
    tr = lm.observe("C01", correct=True, difficulty=1)
    # P = .4*.9 / (.4*.9 + .6*.25) = .36/.51 = .70588
    assert abs(tr.posterior - 0.70588) < 1e-4, tr.posterior
    assert abs(lm.mastery["C01"] - 0.70588) < 1e-4


def test_wrong_answer_lowers_mastery():
    lm = fresh(p_init=0.60, p_transit=0.0)
    before = lm.mastery["C22"]
    lm.observe("C22", correct=False, difficulty=2)
    assert lm.mastery["C22"] < before


def test_transit_adds_learning():
    a = fresh(p_init=0.40, p_transit=0.0).observe("C01", True, 1).posterior
    lm = fresh(p_init=0.40, p_transit=0.20)
    lm.observe("C01", True, 1)
    assert lm.mastery["C01"] > a


def test_hard_correct_is_stronger_evidence_than_easy_correct():
    easy, hard = fresh(p_init=0.3), fresh(p_init=0.3)
    easy.observe("C01", True, 1)
    hard.observe("C01", True, 3)
    assert hard.mastery["C01"] > easy.mastery["C01"]


def test_easy_wrong_is_stronger_evidence_than_hard_wrong():
    easy, hard = fresh(p_init=0.7), fresh(p_init=0.7)
    easy.observe("C01", False, 1)
    hard.observe("C01", False, 3)
    assert easy.mastery["C01"] < hard.mastery["C01"]


def test_success_lifts_prerequisites():
    lm = fresh(p_init=0.25)
    for _ in range(4):
        lm.observe("C18", True, 3)      # myopia / hypermetropia
    for anc, _ in G.ancestors("C18"):
        assert lm.mastery[anc] > 0.25, anc


def test_failure_drags_prerequisites_down():
    lm = fresh(p_init=0.70)
    lm.observe("C14", False, 2)          # lens formula
    assert lm.mastery["C06"] < 0.70      # mirror sign convention, depth 1
    assert lm.mastery["C13"] < 0.70


def test_propagation_decays_with_depth():
    lm = fresh(p_init=0.70)
    lm.observe("C18", False, 2)
    depths = dict(G.ancestors("C18"))
    near = [a for a, d in depths.items() if d == 1]
    far = [a for a, d in depths.items() if d >= 4]
    drop_near = min(0.70 - lm.mastery[a] for a in near)
    drop_far = max(0.70 - lm.mastery[a] for a in far)
    assert drop_near > drop_far


def test_bounds_hold_under_long_runs():
    lm = fresh()
    for i in range(200):
        lm.observe("C27", correct=bool(i % 3), difficulty=(i % 3) + 1)
    assert all(0.0 < v < 1.0 for v in lm.mastery.values())


def test_repeated_success_converges_high():
    lm = fresh()
    for _ in range(8):
        lm.observe("C24", True, 2)
    assert lm.mastery["C24"] > 0.95


def test_uncertainty_peaks_at_half():
    lm = fresh()
    lm.mastery["C01"], lm.mastery["C02"] = 0.5, 0.95
    assert lm.uncertainty("C01") > lm.uncertainty("C02")


def test_readiness_gates_the_frontier():
    lm = fresh(p_init=0.05)
    lm.mastery["C01"] = 0.95
    front = G.frontier(lm.mastery)
    assert "C02" in front            # prerequisite C01 is known
    assert "C14" not in front        # deep chain still unmet


def test_snapshot_roundtrip():
    lm = fresh()
    lm.observe("C24", True, 2)
    back = LearnerModel.restore(G, lm.snapshot())
    assert back.mastery == lm.mastery and back.attempts == lm.attempts


def test_trace_is_auditable():
    lm = fresh(p_init=0.6)
    tr = lm.observe("C14", False, 2)
    assert tr.steps[0].concept == "C14"
    assert len(tr.steps) > 1
    assert all(s.reason for s in tr.steps)
