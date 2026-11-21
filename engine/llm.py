"""Pluggable generation / analysis backend.

The tutor never calls a model directly. It calls this interface, so the system
runs fully offline on an authored bank plus a cue-based reasoning classifier,
and upgrades to a hosted model by setting one environment variable.

    LLMBackend
      |- TemplateBackend   no API key: authored bank + cue matching
      |- AnthropicBackend  set ANTHROPIC_API_KEY to enable
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Protocol

DATA = Path(__file__).resolve().parent.parent / "data"

# Phrases that betray a misconception in a learner's own words. These power
# reasoning analysis without a model; a hosted model supersedes them.
MISCONCEPTION_CUES: dict[str, list[str]] = {
    "M_IMG_ON_SURFACE": ["on the mirror", "on the surface", "image is on the glass", "inside the mirror"],
    "M_EYE_EMITS": ["eye sends", "rays from my eye", "eye emits", "light from the eye"],
    "M_CONCAVE_ALWAYS_REAL": ["concave always", "concave mirror always real", "always real and inverted"],
    "M_F_EQUALS_R": ["f = r", "focal length is the radius", "focal length equals radius", "f equals r"],
    "M_NO_SIGN_CONVENTION": ["all distances positive", "ignored the sign", "took it as positive", "no negative sign"],
    "M_NEG_M_MEANS_SMALL": ["negative means smaller", "minus so smaller", "negative magnification means diminished"],
    "M_REFRACT_IS_REFLECT": ["bounces back", "reflects into the glass", "reflection through the glass"],
    "M_BEND_AWAY_DENSER": ["away from the normal", "bends away when entering water", "denser so away"],
    "M_RI_HAS_UNITS": ["m/s", "metres per second", "unit of refractive index"],
    "M_SPEED_CONSTANT_ALL_MEDIA": ["same speed", "speed does not change", "light always travels at c"],
    "M_CONVEX_LENS_ALWAYS_MAGNIFIES": ["convex always magnif", "convex lens always bigger", "always enlarged"],
    "M_POWER_IN_CM": ["1/20 = 0.05 d", "f in cm", "focal length in centim", "divided by cm"],
    "M_LENS_MOVES_TO_FOCUS": ["lens moves", "moves forward", "lens shifts", "moves back and forth"],
    "M_MYOPIA_CONVEX": ["myopia convex", "near-sighted convex", "convex for myopia", "concave for hypermetropia"],
    "M_SKY_REFLECTS_SEA": ["reflects the sea", "reflection of the ocean", "because of the water"],
    "M_DISPERSION_IS_COLOURING": ["prism adds", "prism colours", "glass creates the colour", "produces the colours"],
    "M_CURRENT_CONSUMED": ["current is used up", "current gets consumed", "less current after", "current reduces after the bulb"],
    "M_CURRENT_SAME_PARALLEL": ["same current in both branches", "current is same in parallel", "equal current in each branch"],
    "M_VOLTAGE_DIVIDES_PARALLEL": ["voltage divides in parallel", "potential divides across branches", "voltage splits in parallel"],
    "M_PARALLEL_ADDS": ["added them", "r1 + r2 in parallel", "sum of the resistances in parallel", "just add"],
    "M_THICK_WIRE_MORE_R": ["thicker so more resistance", "thick wire more resistance", "greater area more resistance"],
    "M_RESISTIVITY_DEPENDS_GEOMETRY": ["resistivity depends on length", "resistivity changes with thickness", "longer so resistivity"],
    "M_HEAT_LINEAR_IN_I": ["heat is proportional to current", "h = irt", "double current double heat"],
    "M_POWER_V_OVER_I": ["p = v/i", "power is v by i", "voltage divided by current"],
    "M_FIELD_LINES_CROSS": ["lines cross", "they intersect", "lines can meet"],
    "M_LINES_NOT_CLOSED": ["start at north and stop", "end at the south pole", "do not continue inside"],
    "M_FLEMING_SWAPPED": ["right hand rule for the motor", "left hand for induced", "right hand for force"],
    "M_INDUCTION_NEEDS_BATTERY": ["needs a battery", "no cell so no current", "requires a source"],
    "M_EARTH_WIRE_CARRIES": ["earth wire carries", "earth completes the circuit", "current flows through earth normally"],
    "M_NUCLEAR_RENEWABLE": ["nuclear is renewable", "uranium is renewable", "nuclear never runs out"],
    "M_SOLAR_ZERO_IMPACT": ["solar has no impact", "completely clean", "zero pollution entirely"],
    "M_TWINKLE_STAR_VARIES": ["star changes brightness", "star is flickering", "stars blink on and off"],
}


GEMINI_HOST = "https://generativelanguage.googleapis.com"
GEMINI_MODELS = ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest")


MISCONCEPTION_LABELS: dict[str, str] = {}


def register_labels(graph) -> None:
    """Give the backends human-readable misconception labels for prompting."""
    MISCONCEPTION_LABELS.update(
        {mid: m.label for mid, m in graph.misconceptions.items()})


class LLMBackend(Protocol):
    name: str

    def classify_reasoning(self, text: str, concept: str,
                           candidates: list[str]) -> list[str]: ...

    def generate_question(self, concept_name: str, difficulty: int,
                          misconceptions: list[dict]) -> dict | None: ...

    def explain(self, concept_name: str, question: str, correct: str,
                misconception: dict | None) -> str: ...


class TemplateBackend:
    """Offline backend. Deterministic, instant, no network."""

    name = "offline (authored bank + cue matching)"

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random(7)

    def classify_reasoning(self, text: str, concept: str,
                           candidates: list[str]) -> list[str]:
        if not text or len(text.strip()) < 4:
            return []
        norm = re.sub(r"\s+", " ", text.lower())
        hits = []
        for mid in candidates:
            for cue in MISCONCEPTION_CUES.get(mid, []):
                if cue in norm:
                    hits.append(mid)
                    break
        return hits

    def generate_question(self, concept_name: str, difficulty: int,
                          misconceptions: list[dict]) -> dict | None:
        return None   # the authored bank is the source of items offline

    def explain(self, concept_name: str, question: str, correct: str,
                misconception: dict | None) -> str:
        if misconception:
            return (f"The answer is {correct}. Your choice points to a common "
                    f"slip: {misconception['label'].lower()}. {misconception['remedy']}")
        return f"The answer is {correct}. Review {concept_name} and try a similar item."


class AnthropicBackend:
    """Hosted backend. Enabled only when ANTHROPIC_API_KEY is present."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-5", fallback: LLMBackend | None = None):
        import anthropic  # imported lazily so the offline path needs no dependency

        self.client = anthropic.Anthropic()
        self.model = model
        self.fallback = fallback or TemplateBackend()

    def _json_call(self, prompt: str, max_tokens: int = 900):
        msg = self.client.messages.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
        return json.loads(raw)

    def classify_reasoning(self, text: str, concept: str,
                           candidates: list[str]) -> list[str]:
        if not text.strip() or not candidates:
            return []
        try:
            options = "\n".join(f"- {c}" for c in candidates)
            out = self._json_call(
                "A Class 10 physics student explained their reasoning below. "
                "Identify which of the listed misconceptions their reasoning "
                "shows. Reply with a JSON array of ids only, [] if none apply. "
                "Do not infer a misconception that is not clearly present.\n\n"
                f"Misconceptions:\n{options}\n\nStudent reasoning: {text!r}",
                max_tokens=200,
            )
            return [m for m in out if m in candidates]
        except Exception:
            return self.fallback.classify_reasoning(text, concept, candidates)

    def generate_question(self, concept_name: str, difficulty: int,
                          misconceptions: list[dict]) -> dict | None:
        try:
            tags = "\n".join(f"- {m['id']}: {m['label']}" for m in misconceptions)
            return self._json_call(
                f"Write one CBSE Class 10 physics multiple-choice question on "
                f"'{concept_name}' at difficulty {difficulty} of 3. Exactly four "
                f"options. Every wrong option must be the answer a student would "
                f"reach holding one of these misconceptions, tagged with its id:\n"
                f"{tags}\n\nReturn JSON: {{\"stem\":str, \"options\":[{{\"text\":str,"
                f"\"correct\":bool,\"misconception\":str|null}}], \"explanation\":str}}"
            )
        except Exception:
            return None

    def explain(self, concept_name: str, question: str, correct: str,
                misconception: dict | None) -> str:
        try:
            extra = f" The student appears to think: {misconception['label']}." if misconception else ""
            msg = self.client.messages.create(
                model=self.model, max_tokens=300,
                messages=[{"role": "user", "content":
                    f"Explain to a Class 10 student, in under 70 words, why the answer "
                    f"to '{question}' is '{correct}'.{extra} Address the mistake directly."}],
            )
            return msg.content[0].text.strip()
        except Exception:
            return self.fallback.explain(concept_name, question, correct, misconception)


class GeminiBackend:
    """Google Gemini over plain HTTPS. No SDK, so nothing extra to install.

    Every method falls back to the offline backend on any failure, so a dead
    network, an expired key or a renamed model degrades the product instead of
    breaking it.
    """

    name = "Gemini"

    def __init__(self, api_key: str, model: str | None = None,
                 fallback: "LLMBackend | None" = None):
        self.api_key = api_key.strip()
        self.model = model or GEMINI_MODELS[0]
        self.fallback = fallback or TemplateBackend()
        self.last_error: str | None = None

    # -- transport -----------------------------------------------------
    def _post(self, model: str, prompt: str, max_tokens: int) -> str:
        import urllib.error
        import urllib.request

        url = f"{GEMINI_HOST}/v1beta/models/{model}:generateContent?key={self.api_key}"
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7},
        }).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.loads(r.read())
        return payload["candidates"][0]["content"]["parts"][0]["text"]

    def _available_model(self) -> str | None:
        """Ask the API which models this key can actually use."""
        import urllib.request

        url = f"{GEMINI_HOST}/v1beta/models?key={self.api_key}"
        with urllib.request.urlopen(url, timeout=20) as r:
            names = [m["name"].split("/")[-1] for m in json.loads(r.read())["models"]]
        for preferred in GEMINI_MODELS:
            if preferred in names:
                return preferred
        return next((n for n in names if "flash" in n and "vision" not in n), None)

    def _call(self, prompt: str, max_tokens: int = 900) -> str:
        """Call the model, re-resolving the model name once if it has moved."""
        try:
            return self._post(self.model, prompt, max_tokens)
        except Exception as first:
            if "404" not in str(first):
                raise
            found = self._available_model()
            if not found:
                raise
            self.model = found
            return self._post(found, prompt, max_tokens)

    @staticmethod
    def _json(raw: str):
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
        start = min((i for i in (raw.find("{"), raw.find("[")) if i != -1), default=-1)
        if start > 0:
            raw = raw[start:]
        return json.loads(raw)

    def check(self) -> tuple[bool, str]:
        """Verify the key works. Returns (ok, message) for the UI to display."""
        try:
            self._call("Reply with the single word: ready", max_tokens=20)
            return True, f"connected, using {self.model}"
        except Exception as e:
            msg = str(e)
            if "400" in msg or "API_KEY_INVALID" in msg:
                return False, "the key was rejected — check it was copied in full"
            if "403" in msg:
                return False, "the key is not authorised for this API"
            if "429" in msg:
                return False, "rate limit reached — wait a minute and retry"
            return False, f"could not reach Gemini ({msg[:80]})"

    # -- capabilities --------------------------------------------------
    def classify_reasoning(self, text: str, concept: str,
                           candidates: list[str]) -> list[str]:
        if not text.strip() or not candidates:
            return []
        try:
            opts = "\n".join(f"- {c}: {MISCONCEPTION_LABELS.get(c, c)}"
                              for c in candidates)
            out = self._json(self._call(
                "A school student explained their reasoning on a physics question. "
                "Decide which of these specific misconceptions their reasoning shows. "
                "Only pick one if it is clearly present; an imprecise but correct "
                "explanation shows none. Reply with a JSON array of ids, or [].\n\n"
                f"Misconceptions:\n{opts}\n\nStudent wrote: {text!r}", 300))
            return [m for m in out if m in candidates]
        except Exception as e:
            self.last_error = str(e)[:120]
            return self.fallback.classify_reasoning(text, concept, candidates)

    def generate_question(self, concept_name: str, difficulty: int,
                          misconceptions: list[dict]) -> dict | None:
        try:
            tags = "\n".join(f"- {m['id']}: {m['label']}" for m in misconceptions)
            out = self._json(self._call(
                f"Write one CBSE Class 10 physics multiple-choice question on "
                f"\"{concept_name}\", difficulty {difficulty} of 3. Exactly four "
                f"options, exactly one correct. Each wrong option must be the answer "
                f"a student would reach while holding one of these misconceptions, "
                f"and must carry that misconception's id:\n{tags}\n\n"
                f"Reply with JSON only:\n"
                f'{{"stem": "...", "options": [{{"text": "...", "correct": true, '
                f'"misconception": null}}], "explanation": "..."}}', 1100))
            opts = out.get("options", [])
            if len(opts) != 4 or sum(bool(o.get("correct")) for o in opts) != 1:
                return None
            valid = {m["id"] for m in misconceptions}
            for o in opts:
                if o.get("misconception") not in valid:
                    o["misconception"] = None
            return out
        except Exception as e:
            self.last_error = str(e)[:120]
            return None

    def explain(self, concept_name: str, question: str, correct: str,
                misconception: dict | None) -> str:
        try:
            extra = (f" They appear to believe: {misconception['label']}. "
                     f"Correct that directly." if misconception else "")
            return self._call(
                f"A Class 10 student answered a question on {concept_name}. "
                f"Question: {question}\nCorrect answer: {correct}.{extra}\n"
                f"Explain in under 60 words, plainly, no preamble.", 250).strip()
        except Exception as e:
            self.last_error = str(e)[:120]
            return self.fallback.explain(concept_name, question, correct, misconception)


def get_backend(gemini_key: str | None = None) -> LLMBackend:
    if gemini_key:
        return GeminiBackend(gemini_key)
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiBackend(os.environ["GEMINI_API_KEY"])
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicBackend()
        except Exception:
            pass
    return TemplateBackend()
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicBackend()
        except Exception:
            pass
    return TemplateBackend()
