# Adaptive exam-prep tutor with a learner model

CBSE Class 10 Physics. The system does not just answer questions — it maintains a
probabilistic model of what the student knows, concept by concept, and uses it to
choose what to ask next.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Runs fully offline with no API key. Add a Gemini key in the sidebar to switch
on live question generation (see **The language model** below). A class of 30 seeded students ships with the
repo so the teacher dashboard has data on first launch.

### Managing student records

```bash
python manage.py list          # show every stored student
python manage.py clear-demo    # remove the seeded demo students, keep real ones
python manage.py remove s004   # remove one
python seed.py 30              # regenerate the demo class
```

The demo students can also be removed from the app: Teacher view → sidebar → Manage data.
Records are plain JSON under `data/students/`.

### A note on the question bank

Options are shuffled per student per question at display time (`engine/bank.py`,
`option_order`). The authored bank lists the correct option first, so showing it in
authored order would make the answer always the first choice. A regression test asserts
the correct answer lands in each of the four slots roughly a quarter of the time.

## The seven requirements

| Requirement | Where it lives | What it actually does |
|---|---|---|
| Diagnostic test | `engine/selector.py`, Diagnostic page | 10 adaptive items chosen by information gain, not a fixed paper |
| Concept graph | `data/concept_graph.json` | 37 concepts, 5 chapters, prerequisite edges, 32 named misconceptions |
| Learner model | `engine/bkt.py` | Bayesian Knowledge Tracing per concept + propagation along prerequisite edges |
| Targeted question generation | `engine/selector.py`, `engine/llm.py` | Weakest *ready* concept, difficulty tuned to a ~70% success rate |
| Misconception detection | `engine/misconception.py` | Two signals: misconception-tagged distractors, plus analysis of written reasoning |
| Spaced repetition | `engine/srs.py` | SM-2 variant where the interval is scaled by model mastery, not just recall quality |
| Teacher dashboard | `app.py` | Class mastery heatmap, weakest concepts, misconceptions ranked by how many students hold them |

## What makes the learner model more than a score counter

**1. Bayesian updating.** Each concept carries a mastery probability. An answer is
evidence, not a verdict:

```
correct:    P' = p(1-slip) / [ p(1-slip) + (1-p)·guess ]
incorrect:  P' = p·slip    / [ p·slip    + (1-p)(1-guess) ]
then:       p  = P' + (1-P')·P(transit)
```

`guess` and `slip` are calibrated by item difficulty, so a correct answer on a hard
item moves belief further than a correct answer on an easy one, and a wrong answer on
an easy item is more damaging than one on a hard item. Both are tested.

**2. Propagation across the graph.** One answer updates many concepts. Success is the
logically stronger signal — solving a lens-formula problem is near-proof the sign
convention is understood — so prerequisites are raised toward the new value, decaying
with distance. Failure is ambiguous, so it is propagated upward weakly and a little way
downward. Every update returns an audit trail the UI renders, so you can see exactly
which concepts moved and why.

**3. Readiness gating.** A concept is only offered once its prerequisites are met. The
tutor will not drill lens formula at a student who cannot do the sign convention; it
teaches the prerequisite first.

## Misconception detection

Every distractor in the bank is authored against a named misconception, so selecting it
is direct evidence requiring no text parsing. On top of that, the student's written
reasoning is classified against that concept's misconception taxonomy — which catches
the student who picked the right answer for the wrong reason. Evidence accumulates in a
Beta posterior per misconception, so one slip is not a diagnosis but a pattern is.

## The language model

`engine/llm.py` defines one interface with three implementations, so the rest of the
app never knows which is in use.

| Backend | When it runs | What it does |
|---|---|---|
| `TemplateBackend` | no key present | authored bank, plus phrase matching on written reasoning |
| `GeminiBackend` | a Google AI Studio key is supplied | writes new questions on demand, reads written reasoning, explains mistakes |
| `AnthropicBackend` | `ANTHROPIC_API_KEY` is set | the same, via Claude |

Paste a key into the sidebar (**AI model**), or set `GEMINI_API_KEY` as an environment
variable, or add it to `.streamlit/secrets.toml` when deploying. The sidebar verifies
the key and reports what went wrong if it fails.

With a key live, Practice gains a **Generate a new question** mode that writes a fresh
item targeted at the student's weakest ready concept, with every distractor tagged to a
misconception from the taxonomy. Explanations after each answer are written for that
student and address the specific mistake they made.

Three things make this safe to demo. Generated questions are validated before display —
wrong option count, more than one correct answer, or an invented misconception id all
cause the item to be rejected rather than shown. Every model call falls back to the
offline path on any failure, so a dead network degrades the product instead of breaking
it. And the key never enters the source: it is read from the sidebar, the environment,
or deployment secrets.

Uses no SDK, only the standard library, so there is nothing extra to install. If the
model name is retired, the backend asks the API which models the key can use and
retries once.

## Tests

```bash
python -m pytest tests/ -q      # 28 tests
```

Covering the posterior against hand calculation, difficulty calibration monotonicity,
propagation direction and decay, prerequisite gating, persistence round-trips, and a
full click-through of the answer loop in the UI.

## Interface

Three views, chosen in the sidebar.

**Home** opens on the concept graph itself, drawn as SVG with every node filled by class
mastery (`engine/viz.py`). Prerequisites run left to right, so the structure the learner
model reasons over is the first thing anyone sees.

**Student** holds the diagnostic, the practice loop and a progress page that redraws the
same graph with that student's own mastery, outlining the concepts they are ready to
learn next.

**Teacher** holds the class overview, the misconception ranking and per-student detail.

### Colour

Mastery is a diverging scale (below par, at par, above par) with a hueless midpoint,
shared between the SVG graph and every chart so one colour means one thing throughout.
The five chapter hues are a fixed categorical order validated for colour-vision
deficiency separation and contrast against the page surface, not a default scheme.

Typography is IBM Plex Sans, with IBM Plex Mono reserved for numeric readouts, since
those are instrument readings. The font is loaded from Google Fonts at runtime and
falls back to the system sans if the network is unavailable.

## Layout

```
app.py                  Streamlit UI (student + teacher)
seed.py                 synthetic class, generated through the real pipeline
engine/graph.py         concept graph, traversal, readiness
engine/bkt.py           learner model
engine/selector.py      adaptive item selection, difficulty calibration
engine/misconception.py misconception tracking
engine/srs.py           spaced repetition
engine/bank.py          question bank
engine/llm.py           pluggable generation backend
engine/store.py         flat-file persistence
engine/viz.py           concept-graph SVG rendering
manage.py               list and remove student records
data/                   concept graph, 74 questions, student records
```
