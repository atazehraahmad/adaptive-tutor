"""Adaptive exam-prep tutor with a learner model. Streamlit front end."""
from __future__ import annotations

import random as _random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt
import pandas as pd
import streamlit as st

from engine import store
from engine.bank import QuestionBank, option_order
from engine.bkt import LearnerModel
from engine.graph import ConceptGraph
from engine.bank import Option, Question
from engine.llm import GeminiBackend, get_backend, register_labels
from engine.misconception import MisconceptionTracker
from engine.selector import (choose_difficulty, diagnostic_plan,
                             next_diagnostic_concept, next_practice_item,
                             success_probability)
from engine.srs import Scheduler
from engine.viz import (MASTERY_HIGH, MASTERY_LOW, MASTERY_MID,
                        concept_graph_svg, mastery_colour)

# Categorical hues for the five chapters, in fixed order, validated for
# colour-vision deficiency separation and contrast against a white surface.
CHAPTER_HUES = ["#4150D8", "#1E8FB5", "#BE7320", "#9B5BC8", "#1F9469"]
MASTERY_SCALE = alt.Scale(domain=[0, 0.5, 1],
                          range=[MASTERY_LOW, MASTERY_MID, MASTERY_HIGH])


def chapter_scale(graph) -> alt.Scale:
    return alt.Scale(domain=graph.chapters(), range=CHAPTER_HUES)

st.set_page_config(page_title="Adaptive Tutor", page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;450;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

  html, body, [class*="css"], .stMarkdown, button, input, textarea, select {
      font-family: 'IBM Plex Sans', system-ui, sans-serif;
  }
  .block-container { padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1340px; }

  h1 { font-weight: 600; letter-spacing: -0.03em; color: #16202B; line-height: 1.12; }
  h2 { font-weight: 600; letter-spacing: -0.02em; color: #16202B; }
  h3, h4 { font-weight: 600; letter-spacing: -0.01em; color: #16202B; }
  p, li { color: #2B3845; }

  /* numeric readouts are instrument readings, so they are set in mono */
  [data-testid="stMetricValue"] {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 1.55rem; font-weight: 600; color: #16202B;
  }
  [data-testid="stMetricLabel"] p {
      font-size: 0.82rem; color: #5B6B77; font-weight: 450;
  }
  [data-testid="stMetric"] {
      background: #FFFFFF; border: 1px solid #DDE5EA;
      border-radius: 4px; padding: 13px 16px;
  }

  .masthead {
      border: 1px solid #DDE5EA; border-bottom: none;
      border-radius: 6px 6px 0 0;
      padding: 30px 32px 6px 32px; background: #FFFFFF;
  }
  .graphframe {
      border: 1px solid #DDE5EA; border-radius: 0 0 6px 6px;
      padding: 4px 26px 16px 26px; background: #FFFFFF; margin-bottom: 30px;
  }
  .toprule {
      border: none; border-top: 2px solid #16202B;
      margin: 0 0 14px 0; opacity: 1;
  }
  .toprule + h4 { margin-top: 0; }
  .masthead h1 { font-size: 2.15rem; margin: 0 0 10px 0; max-width: 24ch; }
  .masthead .lede {
      color: #42505D; font-size: 1.02rem; line-height: 1.6;
      max-width: 68ch; margin: 0 0 22px 0;
  }
  .facts {
      font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
      color: #5B6B77; margin: 18px 0 2px 0;
  }
  .facts b { color: #16202B; font-weight: 600; }

  .note {
      border-left: 3px solid #2F3BB3; padding: 2px 0 2px 14px;
      color: #42505D; margin: 6px 0 16px 0;
  }
  .scale { display:flex; align-items:center; gap:9px; color:#5B6B77; font-size:0.78rem; }

  /* answer options read as a list of choices, not a cramped form control */
  .main [data-testid="stRadioOption"], section.main [data-testid="stRadioOption"],
  [data-testid="stAppViewContainer"] [data-testid="stRadioOption"] {
      border: 1px solid #DDE5EA; border-radius: 5px;
      padding: 10px 14px; margin-bottom: 7px; width: 100%;
      transition: border-color .12s ease, background .12s ease;
  }
  [data-testid="stAppViewContainer"] [data-testid="stRadioOption"]:hover {
      border-color: #2F3BB3; background: #F8FAFD;
  }
  [data-testid="stRadioOption"] > div:last-child { width: 100%; }

  section[data-testid="stSidebar"] { border-right: 1px solid #E4EAEE; }
  section[data-testid="stSidebar"] [data-testid="stRadioOption"] {
      border: none !important; padding: 3px 0 !important;
      margin-bottom: 1px !important; background: none !important;
  }

  [data-testid="stDataFrame"] { border-radius: 4px; }
  .stButton button { border-radius: 4px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

DIAGNOSTIC_LEN = 10


# ----------------------------------------------------------------- resources
@st.cache_resource
def resources():
    graph, bank = ConceptGraph.load(), QuestionBank.load()
    register_labels(graph)
    return graph, bank


GRAPH, BANK = resources()


def api_key() -> str | None:
    """A key typed into the sidebar wins; otherwise deployment secrets or env."""
    if st.session_state.get("gemini_key"):
        return st.session_state["gemini_key"]
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return None


LLM = get_backend(api_key())
AI_ON = isinstance(LLM, GeminiBackend)


# ------------------------------------------------------------- session state
def blank_student(sid: str, name: str, section: str) -> dict:
    return {"id": sid, "name": name, "section": section, "synthetic": False,
            "model": LearnerModel(GRAPH).snapshot(), "misconceptions": {},
            "schedule": {}, "history": [], "diagnostic_done": False}


def load_state(sid: str) -> None:
    rec = store.load(sid)
    if rec is None:
        return
    st.session_state.student = rec
    st.session_state.lm = LearnerModel.restore(GRAPH, rec["model"])
    st.session_state.tracker = MisconceptionTracker.restore(GRAPH, rec.get("misconceptions", {}))
    st.session_state.sched = Scheduler(rec.get("schedule", {}))
    st.session_state.q = None
    st.session_state.feedback = None


def persist() -> None:
    s = st.session_state.student
    s["model"] = st.session_state.lm.snapshot()
    s["misconceptions"] = st.session_state.tracker.snapshot()
    s["schedule"] = st.session_state.sched.snapshot()
    store.save(s["id"], s)


def seen_ids() -> set[str]:
    return {h["q"] for h in st.session_state.student["history"]}


def answer(concept: str, q, chosen: int, reasoning: str) -> dict:
    """Grade, update every model, persist. Returns a feedback bundle."""
    lm, tracker, sched = st.session_state.lm, st.session_state.tracker, st.session_state.sched
    correct, distractor_mis = q.grade(chosen)

    candidates = GRAPH.concepts[concept].misconceptions
    reasoning_mis = LLM.classify_reasoning(reasoning, concept, candidates) if reasoning else []
    probed = [o.misconception for o in q.options if o.misconception]
    fired = tracker.record_attempt(concept, distractor_mis, reasoning_mis, probed)

    trace = lm.observe(concept, correct, q.difficulty)
    sched.record(concept, correct, lm.mastery[concept], date.today())
    st.session_state.student["history"].append({
        "q": q.id, "concept": concept, "correct": correct,
        "difficulty": q.difficulty, "misconception": distractor_mis,
        "reasoning": reasoning[:300], "date": date.today().isoformat(),
    })
    persist()
    return {"correct": correct, "trace": trace, "fired": fired,
            "distractor": distractor_mis, "reasoning_mis": reasoning_mis, "q": q}


# ------------------------------------------------------------------ fragments
def legend_bar(n: int = 9) -> str:
    cells = "".join(
        f'<span style="display:inline-block;width:19px;height:8px;'
        f'background:{mastery_colour(i / (n - 1))}"></span>' for i in range(n))
    return f'<span style="display:inline-flex;border-radius:2px;overflow:hidden">{cells}</span>'


def generated_question(concept: str, difficulty: int) -> Question | None:
    """Ask the model for a fresh item on this concept, validated before use."""
    c = GRAPH.concepts[concept]
    spec = LLM.generate_question(
        c.name, difficulty,
        [{"id": m, "label": GRAPH.misconceptions[m].label} for m in c.misconceptions])
    if not spec:
        return None
    try:
        opts = [Option(text=str(o["text"]), correct=bool(o.get("correct")),
                       misconception=o.get("misconception")) for o in spec["options"]]
    except Exception:
        return None
    if len(opts) != 4 or sum(o.correct for o in opts) != 1:
        return None
    n = st.session_state.get("gen_count", 0) + 1
    st.session_state["gen_count"] = n
    return Question(id=f"AI{n:03d}", concept=concept, difficulty=difficulty,
                    stem=str(spec["stem"]), options=opts,
                    explanation=str(spec.get("explanation", "")))


def mastery_badge(v: float) -> str:
    return "🟢" if v >= 0.75 else ("🟡" if v >= 0.45 else "🔴")


def render_trace(fb: dict) -> None:
    trace = fb["trace"]
    direct = trace.steps[0]
    arrow = "▲" if direct.delta >= 0 else "▼"
    st.markdown(
        f"**{GRAPH.concepts[trace.concept].name}** "
        f"{direct.before:.0%} → {direct.after:.0%}  {arrow} {abs(direct.delta):.1%}"
    )
    with st.expander(f"How the learner model updated ({len(trace.steps)} concepts affected)"):
        st.caption(
            f"Bayesian update with difficulty-{trace.difficulty} parameters: "
            f"P(guess)={trace.guess}, P(slip)={trace.slip} → posterior {trace.posterior:.3f}, "
            f"then a learning increment and propagation along prerequisite edges."
        )
        rows = [{
            "Concept": GRAPH.concepts[s.concept].name,
            "Before": round(s.before, 3), "After": round(s.after, 3),
            "Change": round(s.delta, 3), "Why": s.reason,
        } for s in trace.steps]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if len(trace.steps) > 1:
            st.info(
                f"The learner answered one question, but belief about "
                f"{len(trace.steps) - 1} other concept(s) moved too, because the "
                f"concept graph says they are connected."
            )


def render_feedback(fb: dict) -> None:
    q = fb["q"]
    if fb["correct"]:
        st.success(f"Correct — {q.correct_text}")
    else:
        st.error(f"Not right. The answer is: {q.correct_text}")
    if AI_ON:
        mis = fb["fired"][0] if fb["fired"] else None
        with st.spinner("Writing an explanation for you"):
            written = LLM.explain(
                GRAPH.concepts[q.concept].name, q.stem, q.correct_text,
                {"label": GRAPH.misconceptions[mis].label} if mis else None)
        st.write(written)
    elif q.explanation:
        st.write(q.explanation)

    for mid in fb["fired"]:
        src = ("your reasoning" if mid in fb["reasoning_mis"] and mid != fb["distractor"]
               else "the option you chose")
        st.warning(
            f"**Misconception detected** (from {src}): "
            f"{GRAPH.misconceptions[mid].label}\n\n{GRAPH.misconceptions[mid].remedy}"
        )
    if fb["reasoning_mis"] and fb["correct"]:
        st.caption("Right answer, but the reasoning suggests a gap worth closing.")
    render_trace(fb)


def ask(concept: str, difficulty: int, key: str, header: str) -> None:
    """Render one question and handle submission."""
    if st.session_state.get("q") is None:
        q = BANK.pick(concept, difficulty, seen_ids())
        if q is None:
            st.info("No further items available for this concept.")
            return
        st.session_state.q = q
    q = st.session_state.q

    st.caption(f"{header} in {GRAPH.concepts[q.concept].chapter} — "
               f"{GRAPH.concepts[q.concept].name}, difficulty {q.difficulty} of 3")
    st.subheader(q.stem)
    order = option_order(q.id, st.session_state.student["id"], len(q.options))
    labels = [q.options[i].text for i in order]
    choice = st.radio("Select your answer", labels, index=None, key=f"opt_{key}_{q.id}")
    reasoning = st.text_area(
        "Explain your reasoning (optional, but it lets the tutor find *why* you went wrong)",
        key=f"txt_{key}_{q.id}", height=80, placeholder="e.g. I added the two resistances together",
    )
    if st.button("Submit answer", type="primary", disabled=choice is None, key=f"sub_{key}_{q.id}"):
        original = order[labels.index(choice)]
        st.session_state.feedback = answer(q.concept, q, original, reasoning or "")
        st.rerun()


# ------------------------------------------------------------------- sidebar
st.sidebar.title("🎓 Adaptive Tutor")
st.sidebar.caption(GRAPH.syllabus)
role = st.sidebar.radio("View", ["Home", "Student", "Teacher"])

if role == "Home":
    page = None
elif role == "Student":
    people = store.list_students()
    names = {f"{p['name']} ({p['id']})": p["id"] for p in people}
    # an existing student first, so the view is never an empty form on arrival
    options = list(names) + ["Add a new student"]
    pick = st.sidebar.selectbox("Student", options)
    if pick == "Add a new student":
        with st.sidebar.form("new"):
            nm = st.text_input("Name", "Test Student")
            sec = st.text_input("Section", "10-A")
            if st.form_submit_button("Create"):
                sid = f"u{abs(hash(nm)) % 9000 + 1000}"
                store.save(sid, blank_student(sid, nm, sec))
                load_state(sid)
                st.rerun()
    else:
        sid = names[pick]
        if st.session_state.get("student", {}).get("id") != sid:
            load_state(sid)
    page = st.sidebar.radio("Page", ["Diagnostic", "Practice", "My progress"])
else:
    page = st.sidebar.radio("Page", ["Class overview", "Misconceptions", "Student detail"])
    with st.sidebar.expander("Manage data"):
        demo_n = sum(1 for r in store.load_all() if r.get("synthetic"))
        st.caption(f"{demo_n} seeded demo student(s) are in the class.")
        confirm = st.checkbox("Yes, remove the demo students")
        if st.button("Remove demo students", disabled=not confirm):
            for r in store.load_all():
                if r.get("synthetic"):
                    (store.STUDENTS / f"{r['id']}.json").unlink(missing_ok=True)
            st.success("Removed. Re-create them any time with: python seed.py 30")
            st.rerun()

st.sidebar.divider()
with st.sidebar.expander("AI model", expanded=not AI_ON):
    st.caption(
        "Paste a Google AI Studio key to switch on live question generation and "
        "reasoning analysis. Without one the app runs on its authored bank."
    )
    typed = st.text_input("Gemini API key", type="password",
                          value=st.session_state.get("gemini_key", ""),
                          placeholder="AIza...")
    if typed != st.session_state.get("gemini_key", ""):
        st.session_state["gemini_key"] = typed
        st.session_state.pop("ai_status", None)
        st.rerun()
    if AI_ON:
        if "ai_status" not in st.session_state:
            with st.spinner("Checking the key"):
                st.session_state["ai_status"] = LLM.check()
        ok, msg = st.session_state["ai_status"]
        (st.success if ok else st.error)(msg)

if AI_ON and st.session_state.get("ai_status", (False, ""))[0]:
    st.sidebar.caption("Questions and reasoning analysis: Gemini, live")
else:
    st.sidebar.caption("Questions and reasoning analysis: offline bank")


# -------------------------------------------------------------- landing page
if role == "Home":
    n_students = len(store.list_students())
    all_recs = store.load_all()
    if all_recs:
        avg = {cid: sum(r["model"]["mastery"].get(cid, 0.25) for r in all_recs) / len(all_recs)
               for cid in GRAPH.ids()}
    else:
        avg = {cid: 0.25 for cid in GRAPH.ids()}

    st.markdown(f"""
    <div class="masthead">
      <h1>Every answer updates a model of the student</h1>
      <p class="lede">Most tutors grade the question in front of them. This one keeps a
         mastery probability for all {len(GRAPH.ids())} concepts in the syllabus, linked by
         what has to be understood first. Answer one question and belief moves across
         the whole structure, including concepts the student was never asked.</p>
      <p class="facts">
        <b>{len(GRAPH.ids())}</b> concepts &nbsp;&nbsp;
        <b>{len(BANK.questions)}</b> questions &nbsp;&nbsp;
        <b>{len(GRAPH.misconceptions)}</b> tracked misconceptions &nbsp;&nbsp;
        <b>{n_students}</b> students
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div class="graphframe">{concept_graph_svg(GRAPH, avg)}'
        f'<div class="scale">Class mastery across the syllabus '
        f'<span>low</span>{legend_bar()}<span>high</span></div></div>',
        unsafe_allow_html=True)

    st.write("")
    a, b, c = st.columns(3, gap="large")
    with a:
        st.markdown('<hr class="toprule">', unsafe_allow_html=True)
        st.markdown("#### Belief travels along the edges")
        st.write("Mastery is a probability updated by Bayes' rule, with the question's "
                 "difficulty deciding how much one answer is worth. Because concepts "
                 "sit in a prerequisite graph, a single answer typically shifts belief "
                 "about eight or nine of them.")
    with b:
        st.markdown('<hr class="toprule">', unsafe_allow_html=True)
        st.markdown("#### Wrong answers name the reason")
        st.write("Every incorrect option was written to match a specific misconception, "
                 "so the choice is the diagnosis. Written reasoning is read as well, "
                 "which catches a correct answer reached the wrong way.")
    with c:
        st.markdown('<hr class="toprule">', unsafe_allow_html=True)
        st.markdown("#### Teachers see the class, not the marks")
        st.write("A mastery heatmap across every student and concept, and misconceptions "
                 "ranked by how many students hold them. Nothing had to be graded to "
                 "produce either.")

    st.write("")
    left, right = st.columns(2, gap="medium")
    with left.container(border=True):
        st.markdown("#### Start as a student")
        st.write("Sit the adaptive diagnostic, then practise against your weakest "
                 "concepts. Open the update panel after any answer to see exactly "
                 "which concepts moved and why.")
        st.caption("Choose Student in the sidebar.")
    with right.container(border=True):
        st.markdown("#### Open the teacher view")
        st.write("Find the concepts the class is weakest on and the misconceptions "
                 "worth addressing in the next lesson.")
        st.caption("Choose Teacher in the sidebar.")

    st.caption(f"{GRAPH.syllabus}. Question generation and reasoning analysis "
               f"run on the {LLM.name} backend.")
    st.stop()


# ------------------------------------------------------------- student pages
if role == "Student":
    if "student" not in st.session_state:
        st.info("Pick or create a student in the sidebar to begin.")
        st.stop()

    S = st.session_state.student
    lm = st.session_state.lm
    st.title(S["name"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall mastery", f"{lm.overall():.0%}")
    c2.metric("Questions answered", len(S["history"]))
    c3.metric("Concepts attempted", sum(1 for v in lm.attempts.values() if v))
    c4.metric("Active misconceptions", len(st.session_state.tracker.active()))
    st.divider()

    # ---------------------------------------------------------- diagnostic
    if page == "Diagnostic":
        st.header("Diagnostic test")
        asked = [h["concept"] for h in S["history"] if h.get("phase") == "diag"]
        done = S.get("diagnostic_done")

        if not S["history"] and not done:
            st.write(
                "A short adaptive test that places you on the concept graph. Each "
                "item is chosen to be the most *informative* one left, so ten "
                "questions give an estimate across the whole syllabus rather than "
                "ten answers about one chapter."
            )
            plan = diagnostic_plan(lm, DIAGNOSTIC_LEN)
            st.caption("Planned opening sequence (it re-plans after every answer):")
            st.write(" → ".join(GRAPH.concepts[c].name for c in plan[:5]) + " → …")

        n_done = len([h for h in S["history"]])
        if done or n_done >= DIAGNOSTIC_LEN:
            S["diagnostic_done"] = True
            persist()
            st.success("Diagnostic complete. Your starting profile is set.")
            weak = lm.weakest(5, only_ready=False)
            st.subheader("Where you are weakest")
            for cid, v in weak:
                st.write(f"{mastery_badge(v)} **{GRAPH.concepts[cid].name}** — {v:.0%}")
            st.page_link("app.py", label="Head to Practice from the sidebar →")
        else:
            st.progress(n_done / DIAGNOSTIC_LEN, f"Question {n_done + 1} of {DIAGNOSTIC_LEN}")
            if st.session_state.get("feedback"):
                render_feedback(st.session_state.feedback)
                if st.button("Next question", type="primary"):
                    st.session_state.feedback = None
                    st.session_state.q = None
                    st.rerun()
            else:
                already = [h["concept"] for h in S["history"]]
                cid = next_diagnostic_concept(
                    lm, already,
                    rng=_random.Random(f"{S['id']}:diag:{len(already)}"),
                )
                ask(cid, choose_difficulty(lm.mastery[cid]), "diag", "Diagnostic")

    # ------------------------------------------------------------ practice
    elif page == "Practice":
        st.header("Practice")
        due = st.session_state.sched.due_now()
        due_concepts = [c.concept for c in due]
        modes = ["Target my weak concepts", f"Spaced review ({len(due)} due)"]
        if AI_ON:
            modes.append("Generate a new question")
        mode = st.radio("Mode", modes, horizontal=True)
        if st.session_state.get("feedback"):
            render_feedback(st.session_state.feedback)
            card = st.session_state.sched.card(st.session_state.feedback["q"].concept)
            st.caption(f"Next review of this concept scheduled in {card.interval} day(s).")
            if st.button("Next question", type="primary"):
                st.session_state.feedback = None
                st.session_state.q = None
                st.rerun()
        else:
            if mode.startswith("Spaced") and due_concepts:
                cid = due_concepts[0]
                diff = choose_difficulty(lm.mastery[cid])
                hdr = "Spaced review"
            else:
                cid, diff = next_practice_item(lm, [h["concept"] for h in S["history"]])
                hdr = "Targeted practice"

            if mode.startswith("Generate") and st.session_state.get("q") is None:
                with st.spinner("Writing a question for your weakest concept"):
                    fresh = generated_question(cid, diff)
                if fresh:
                    st.session_state.q = fresh
                    hdr = "Written for you just now by Gemini"
                else:
                    st.warning("The model did not return a usable question. "
                               "Showing one from the bank instead.")
            exp = success_probability(lm.mastery[cid], diff)
            st.caption(
                f"Selected because mastery here is {lm.mastery[cid]:.0%} and its "
                f"prerequisites are met. Difficulty {diff} targets a ~{exp:.0%} success rate."
            )
            ask(cid, diff, "prac", hdr)

    # --------------------------------------------------------- my progress
    else:
        st.header("My progress")
        frontier = set(GRAPH.frontier(lm.mastery)[:6])
        st.markdown(concept_graph_svg(GRAPH, lm.mastery, highlight=frontier),
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="scale">Your mastery across the syllabus '
            f'<span>low</span>{legend_bar()}<span>high</span>'
            f'&nbsp;&nbsp;Outlined concepts are what you are ready to learn next.</div>',
            unsafe_allow_html=True)
        st.write("")
        rows = [{"Concept": c.name, "Chapter": c.chapter, "Mastery": lm.mastery[cid],
                 "Attempts": lm.attempts[cid],
                 "Ready": round(GRAPH.readiness(cid, lm.mastery), 2)}
                for cid, c in GRAPH.concepts.items()]
        df = pd.DataFrame(rows)
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("Mastery:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("Concept:N", sort="-x", title=None),
            color=alt.Color("Mastery:Q", scale=MASTERY_SCALE, legend=None),
            tooltip=["Concept", "Chapter", alt.Tooltip("Mastery", format=".0%"), "Attempts"],
        ).properties(height=620)
        left, right = st.columns([3, 2])
        left.altair_chart(chart, use_container_width=True)

        with right:
            st.subheader("Review queue")
            due = st.session_state.sched.due_now()
            if due:
                for c in due[:8]:
                    st.write(f"🔔 {GRAPH.concepts[c.concept].name} "
                             f"({c.reps} reps, {c.lapses} lapses)")
            else:
                st.caption("Nothing due today.")
            up = st.session_state.sched.upcoming()
            if up:
                st.caption("Upcoming")
                for c in up[:6]:
                    st.write(f"🗓 {GRAPH.concepts[c.concept].name} — in {c.days_until_due()}d")

            st.subheader("Misconceptions to clear")
            act = st.session_state.tracker.active()
            if not act:
                st.caption("None detected yet.")
            for mid, p in act[:6]:
                st.write(f"**{GRAPH.misconceptions[mid].label}** · confidence {p:.0%}")
                st.caption(GRAPH.misconceptions[mid].remedy)


# -------------------------------------------------------------- teacher pages
else:
    ALL = store.load_all()
    if not ALL:
        st.warning("No students yet. Run `python seed.py 30` to populate a class.")
        st.stop()

    models = {s["id"]: LearnerModel.restore(GRAPH, s["model"]) for s in ALL}

    if page == "Class overview":
        st.title("Teacher dashboard")
        st.caption(f"{len(ALL)} students in {GRAPH.syllabus}")
        overall = [m.overall() for m in models.values()]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Class mastery", f"{sum(overall) / len(overall):.0%}")
        c2.metric("Students below 40%", sum(1 for v in overall if v < 0.40))
        c3.metric("Answers logged", sum(len(s["history"]) for s in ALL))
        c4.metric("Distinct misconceptions live",
                  len({m for s in ALL for m, v in s.get("misconceptions", {}).items()
                       if v["hits"] >= 2}))

        st.subheader("Weakest concepts across the class")
        conc = [{"Concept": c.name, "Chapter": c.chapter,
                 "Class mastery": sum(models[s["id"]].mastery[cid] for s in ALL) / len(ALL)}
                for cid, c in GRAPH.concepts.items()]
        cdf = pd.DataFrame(conc).sort_values("Class mastery").head(12)
        st.altair_chart(
            alt.Chart(cdf).mark_bar().encode(
                x=alt.X("Class mastery:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                y=alt.Y("Concept:N", sort="x", title=None),
                color=alt.Color("Chapter:N", scale=chapter_scale(GRAPH),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["Concept", "Chapter", alt.Tooltip("Class mastery", format=".0%")],
            ).properties(height=340), use_container_width=True)

        st.subheader("Mastery heatmap")
        st.caption("Every student against every concept. Red blocks are where the class needs teaching.")
        cells = [{"Student": s["name"], "Concept": GRAPH.concepts[cid].name,
                  "Chapter": GRAPH.concepts[cid].chapter,
                  "Mastery": models[s["id"]].mastery[cid]}
                 for s in ALL for cid in GRAPH.ids()]
        hdf = pd.DataFrame(cells)
        order = [GRAPH.concepts[c].name for c in GRAPH.ids()]
        st.altair_chart(
            alt.Chart(hdf).mark_rect().encode(
                x=alt.X("Concept:N", sort=order, axis=alt.Axis(labelAngle=-55, labelLimit=170)),
                y=alt.Y("Student:N", title=None),
                color=alt.Color("Mastery:Q", scale=MASTERY_SCALE,
                                legend=alt.Legend(format=".0%", title="Mastery")),
                tooltip=["Student", "Concept", "Chapter", alt.Tooltip("Mastery", format=".0%")],
            ).properties(height=620), use_container_width=True)

    elif page == "Misconceptions":
        st.title("Class-wide misconceptions")
        st.caption("Ranked by how many students the model is confident hold them.")
        counts: dict[str, int] = {}
        holders: dict[str, list[str]] = {}
        for s in ALL:
            for mid, v in s.get("misconceptions", {}).items():
                if v["hits"] >= 2:
                    counts[mid] = counts.get(mid, 0) + 1
                    holders.setdefault(mid, []).append(s["name"])
        if not counts:
            st.info("No misconceptions confidently detected yet.")
            st.stop()
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        mdf = pd.DataFrame([{"Misconception": GRAPH.misconceptions[m].label,
                             "Students": n, "Share": n / len(ALL)} for m, n in ranked[:14]])
        st.altair_chart(
            alt.Chart(mdf).mark_bar().encode(
                x=alt.X("Students:Q"),
                y=alt.Y("Misconception:N", sort="-x", title=None,
                        axis=alt.Axis(labelLimit=420)),
                color=alt.value("#B0432F"),
                tooltip=["Misconception", "Students", alt.Tooltip("Share", format=".0%")],
            ).properties(height=440), use_container_width=True)

        st.subheader("Teaching actions")
        for mid, n in ranked[:5]:
            with st.expander(f"{n} students — {GRAPH.misconceptions[mid].label}"):
                st.write(f"**Correction:** {GRAPH.misconceptions[mid].remedy}")
                affected = [GRAPH.concepts[c].name for c in GRAPH.ids()
                            if mid in GRAPH.concepts[c].misconceptions]
                st.write("**Surfaces in:** " + ", ".join(affected))
                st.write("**Students:** " + ", ".join(sorted(holders[mid])))

    else:
        st.title("Student detail")
        who = st.selectbox("Student", [s["name"] for s in ALL])
        rec = next(s for s in ALL if s["name"] == who)
        lm = models[rec["id"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall mastery", f"{lm.overall():.0%}")
        c2.metric("Answers", len(rec["history"]))
        acc = sum(1 for h in rec["history"] if h["correct"]) / max(1, len(rec["history"]))
        c3.metric("Accuracy", f"{acc:.0%}")

        st.subheader("Chapter mastery")
        ch = pd.DataFrame([{"Chapter": k, "Mastery": v} for k, v in lm.chapter_mastery().items()])
        st.altair_chart(
            alt.Chart(ch).mark_bar().encode(
                x=alt.X("Mastery:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                y=alt.Y("Chapter:N", sort="-x", title=None),
                color=alt.Color("Mastery:Q", scale=MASTERY_SCALE, legend=None),
                tooltip=["Chapter", alt.Tooltip("Mastery", format=".0%")],
            ).properties(height=200), use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Weakest concepts")
            for cid, v in lm.weakest(8, only_ready=False):
                st.write(f"{mastery_badge(v)} {GRAPH.concepts[cid].name} — {v:.0%}")
        with right:
            st.subheader("Detected misconceptions")
            tr = MisconceptionTracker.restore(GRAPH, rec.get("misconceptions", {}))
            act = tr.active()
            if not act:
                st.caption("None detected.")
            for mid, p in act[:8]:
                st.write(f"• {GRAPH.misconceptions[mid].label} ({p:.0%})")

        st.subheader("Recent answers")
        hist = pd.DataFrame([{
            "Date": h["date"], "Concept": GRAPH.concepts[h["concept"]].name,
            "Difficulty": h["difficulty"], "Correct": "✓" if h["correct"] else "✗",
            "Misconception": GRAPH.misconceptions[h["misconception"]].label
                             if h.get("misconception") else "",
        } for h in rec["history"][-20:][::-1]])
        st.dataframe(hist, width="stretch", hide_index=True)
