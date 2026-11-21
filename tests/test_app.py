import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def run(**state):
    at = AppTest.from_file(APP, default_timeout=60)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def test_home_page_renders():
    at = run()
    assert not at.exception, at.exception


def test_student_landing_has_no_exception():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Student").run()
    assert not at.exception, at.exception


def test_teacher_class_overview():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Teacher").run()
    assert not at.exception, at.exception
    assert any("Teacher dashboard" in t.value for t in at.title)


def test_teacher_misconceptions_page():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Teacher").run()
    at.sidebar.radio[1].set_value("Misconceptions").run()
    assert not at.exception, at.exception


def test_teacher_student_detail():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Teacher").run()
    at.sidebar.radio[1].set_value("Student detail").run()
    assert not at.exception, at.exception


def test_student_pages_render_for_seeded_student():
    for page in ("Diagnostic", "Practice", "My progress"):
        at = AppTest.from_file(APP, default_timeout=60).run()
        at.sidebar.radio[0].set_value("Student").run()
        opts = at.sidebar.selectbox[0].options
        target = next(o for o in opts if o != "Add a new student")
        at.sidebar.selectbox[0].set_value(target).run()
        at.sidebar.radio[1].set_value(page).run()
        assert not at.exception, f"{page}: {at.exception}"


def test_practice_answer_loop_end_to_end():
    """Select a student, answer a question, confirm feedback and a model update."""
    import json
    from engine import store

    sid = "uitest"
    store.save(sid, {"id": sid, "name": "UI Test", "section": "10-Z",
                     "synthetic": False, "model": {}, "misconceptions": {},
                     "schedule": {}, "history": []})
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.sidebar.radio[0].set_value("Student").run()
    target = next(o for o in at.sidebar.selectbox[0].options if "uitest" in o)
    at.sidebar.selectbox[0].set_value(target).run()
    at.sidebar.radio[1].set_value("Practice").run()
    assert not at.exception, at.exception

    answer_widget = next(r for r in at.radio if r.label == "Select your answer")
    answer_widget.set_value(answer_widget.options[0]).run()
    assert not at.exception, at.exception

    submit = next(b for b in at.button if "Submit" in b.label)
    submit.click().run()
    assert not at.exception, at.exception

    saved = store.load(sid)
    assert len(saved["history"]) == 1, "answer was not recorded"
    assert saved["model"]["mastery"], "learner model was not persisted"
    assert any("Next question" in b.label for b in at.button), "no feedback stage shown"

    (store.STUDENTS / f"{sid}.json").unlink()


def test_choosing_the_correct_option_grades_correct_after_shuffling():
    """The index remap must survive display shuffling, in both directions."""
    from engine import store
    from engine.bank import QuestionBank

    bank = QuestionBank.load()
    for sid, want_correct in (("uitest_ok", True), ("uitest_bad", False)):
        store.save(sid, {"id": sid, "name": sid, "section": "10-Z",
                         "synthetic": False, "model": {}, "misconceptions": {},
                         "schedule": {}, "history": []})
        at = AppTest.from_file(APP, default_timeout=90).run()
        at.sidebar.radio[0].set_value("Student").run()
        at.sidebar.selectbox[0].set_value(
            next(o for o in at.sidebar.selectbox[0].options if sid in o)).run()
        at.sidebar.radio[1].set_value("Practice").run()

        stem = at.subheader[0].value
        q = next(x for x in bank.questions if x.stem == stem)
        target = (q.correct_text if want_correct
                  else next(o.text for o in q.options if not o.correct))

        widget = next(r for r in at.radio if r.label == "Select your answer")
        assert target in widget.options, "displayed options do not match the bank"
        widget.set_value(target).run()
        next(b for b in at.button if "Submit" in b.label).click().run()
        assert not at.exception, at.exception

        saved = store.load(sid)
        assert saved["history"][0]["correct"] is want_correct, (
            f"{sid}: picked {'correct' if want_correct else 'wrong'} option, "
            f"graded {saved['history'][0]['correct']}")
        (store.STUDENTS / f"{sid}.json").unlink()
