import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.graph import ConceptGraph
from engine.llm import GeminiBackend, TemplateBackend, get_backend, register_labels

G = ConceptGraph.load()
register_labels(G)

MISC = [{"id": "M_PARALLEL_ADDS", "label": "adds resistances in parallel"},
        {"id": "M_CURRENT_SAME_PARALLEL", "label": "same current in each branch"}]

GOOD = json.dumps({
    "stem": "Two 4 ohm resistors are connected in parallel. The total is:",
    "options": [
        {"text": "2 ohm", "correct": True, "misconception": None},
        {"text": "8 ohm", "correct": False, "misconception": "M_PARALLEL_ADDS"},
        {"text": "4 ohm", "correct": False, "misconception": "M_CURRENT_SAME_PARALLEL"},
        {"text": "16 ohm", "correct": False, "misconception": None}],
    "explanation": "Reciprocals add, so the total is half of one resistor.",
})


def backend(reply, record=None):
    b = GeminiBackend("test-key")

    def fake(prompt, max_tokens=900):
        if record is not None:
            record.append(prompt)
        if isinstance(reply, Exception):
            raise reply
        return reply

    b._call = fake
    return b


def test_parses_a_well_formed_response():
    q = backend(GOOD).generate_question("Resistors in parallel", 2, MISC)
    assert q["stem"].startswith("Two 4 ohm")
    assert sum(o["correct"] for o in q["options"]) == 1
    assert q["options"][1]["misconception"] == "M_PARALLEL_ADDS"


def test_tolerates_markdown_fenced_json():
    q = backend(f"```json\n{GOOD}\n```").generate_question("x", 2, MISC)
    assert q is not None


def test_tolerates_preamble_before_json():
    q = backend("Sure! Here is your question:\n" + GOOD).generate_question("x", 2, MISC)
    assert q is not None


def test_rejects_wrong_option_count():
    bad = json.loads(GOOD)
    bad["options"] = bad["options"][:3]
    assert backend(json.dumps(bad)).generate_question("x", 2, MISC) is None


def test_rejects_two_correct_answers():
    bad = json.loads(GOOD)
    bad["options"][1]["correct"] = True
    assert backend(json.dumps(bad)).generate_question("x", 2, MISC) is None


def test_strips_invented_misconception_ids():
    bad = json.loads(GOOD)
    bad["options"][1]["misconception"] = "M_NOT_A_REAL_ID"
    q = backend(json.dumps(bad)).generate_question("x", 2, MISC)
    assert q["options"][1]["misconception"] is None


def test_generation_failure_returns_none_not_an_exception():
    assert backend(RuntimeError("503")).generate_question("x", 2, MISC) is None


def test_reasoning_falls_back_offline_on_failure():
    b = backend(RuntimeError("network down"))
    assert b.classify_reasoning("i just added them", "C27",
                                ["M_PARALLEL_ADDS"]) == ["M_PARALLEL_ADDS"]


def test_reasoning_keeps_only_offered_ids():
    b = backend('["M_PARALLEL_ADDS", "M_MADE_UP"]')
    assert b.classify_reasoning("x", "C27", ["M_PARALLEL_ADDS"]) == ["M_PARALLEL_ADDS"]


def test_explain_falls_back_to_offline_text():
    b = backend(RuntimeError("timeout"))
    out = b.explain("Resistors in parallel", "Q?", "2 ohm", None)
    assert "2 ohm" in out


def test_prompt_carries_the_misconception_labels():
    seen = []
    backend(GOOD, seen).generate_question("Resistors in parallel", 3, MISC)
    assert "M_PARALLEL_ADDS" in seen[0] and "adds resistances in parallel" in seen[0]
    assert "difficulty 3" in seen[0]


def test_no_key_means_offline_backend():
    assert isinstance(get_backend(None), TemplateBackend)


def test_key_selects_gemini():
    assert isinstance(get_backend("some-key"), GeminiBackend)
