# Demo script — 4 minutes

Open with the dashboard, not the chat. The dashboard is what no other team will have.

## 0:00 — The problem (25 s)

Open on **Home**. The concept graph is already on screen.

"Every AI tutor can answer a physics question. None of them can tell you what the
student actually understands. This is the syllabus as our system sees it: 37 concepts,
wired by what has to be understood first, each one coloured by how well the class knows
it. Everything else follows from this picture."

## 0:20 — Teacher dashboard (60 s)

Teacher → Class overview. Point at the heatmap.

"Thirty students, thirty-seven concepts. Every cell is a probability our model inferred,
not a test score anyone entered."

Teacher → Misconceptions. Point at the top bar.

"Half this class believes magnetic field lines can intersect. The teacher never had to
grade anything to learn that — it came out of which *wrong options* students picked."

Open the top expander, show the correction and the named student list.

## 1:20 — The learner model (90 s) — this is the part that wins

Student → pick a seeded student → Practice.

Point at the caption above the question: *"Selected because mastery here is X% and its
prerequisites are met. Difficulty 2 targets a ~70% success rate."*

"It picked this concept and this difficulty. Not randomly — the weakest concept the
student is actually *ready* for, at the difficulty where learning is fastest."

Answer it **wrong, deliberately, choosing a tagged distractor**. Then expand
**"How the learner model updated"**.

"One question. Nine concepts moved. We ran a Bayesian update on the concept itself, then
propagated along the prerequisite graph — the student never answered a question about
these, but we now believe less about them, and the further away, the weaker the effect.
That's the table: before, after, and why."

## 2:50 — Misconception detection (45 s)

Point at the warning banner that fired from the distractor.

"We didn't parse any text to get that. Every wrong option in our bank is authored against
a named misconception, so the choice *is* the diagnosis."

Now answer a parallel-resistance question **correctly** but type in the reasoning box:
*"I just added them"*.

"Right answer. Wrong reasoning. Caught anyway — because we also classify what they wrote
against the misconception taxonomy for that concept. Most systems would mark this correct
and move on."

## 3:20 — Live generation (30 s, only with a key in)

Practice → **Generate a new question**.

"This one does not exist yet. It is being written now, for this student's weakest
concept, at the difficulty the model thinks gives them a seventy percent chance — and
every wrong option is tagged to a misconception from our taxonomy, so a question written
three seconds ago still feeds the diagnosis. If the model returns anything malformed we
reject it and fall back to the authored bank, which is also what happens if the wifi
dies."

## 3:50 — Spaced repetition and close (25 s)

My progress → Review queue.

"Everything schedules forward. Intervals scale with model confidence, so a lucky correct
answer on a shaky concept still comes back tomorrow."

Close: "Seven deliverables, but the one that matters is this: we model the student, not
the session. It runs offline with zero API keys, and there are 28 tests behind it."

## If a judge asks

**"Is BKT not quite old?"** Yes — 1994, Corbett and Anderson, and it is still the
baseline deep knowledge tracing is measured against. It is interpretable, it trains on
almost no data, and a teacher can read why it concluded something. Deep KT needs
thousands of learners we do not have on day one.

**"How do you know the mastery numbers are right?"** We don't claim calibration from
real students yet. What we can show is that `seed.py` generates students with a known
hidden ability, runs them through the identical pipeline, and the model recovers the
ranking. That's the honest version of validation at this stage.

**"What if the LLM is down?"** Nothing breaks. Generation and reasoning analysis sit
behind one interface with an offline implementation, and every call falls back to it on
failure. You can watch it happen: clear the key in the sidebar and the product keeps
working on its authored bank.

**"How do you stop the model inventing a bad question?"** We validate every generated
item before it reaches the student: exactly four options, exactly one correct, and any
misconception id the model invents is stripped. A failed item is discarded, not shown.

**"Where does this go next?"** Fit the slip/guess parameters per item from real response
data instead of using difficulty-band defaults, and open the concept graph to teachers
so they can author their own syllabus.
