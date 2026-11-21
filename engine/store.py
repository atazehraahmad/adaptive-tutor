"""Flat-file persistence. One JSON document per student, no database to set up."""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
STUDENTS = DATA / "students"


def _path(student_id: str) -> Path:
    safe = "".join(ch for ch in student_id if ch.isalnum() or ch in "-_")
    return STUDENTS / f"{safe}.json"


def save(student_id: str, payload: dict) -> None:
    STUDENTS.mkdir(parents=True, exist_ok=True)
    tmp = _path(student_id).with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(_path(student_id))


def load(student_id: str) -> dict | None:
    p = _path(student_id)
    return json.loads(p.read_text()) if p.exists() else None


def list_students() -> list[dict]:
    if not STUDENTS.exists():
        return []
    out = []
    for p in sorted(STUDENTS.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            out.append({"id": d.get("id", p.stem), "name": d.get("name", p.stem),
                        "section": d.get("section", "-"),
                        "answers": len(d.get("history", []))})
        except Exception:
            continue
    return out


def load_all() -> list[dict]:
    if not STUDENTS.exists():
        return []
    out = []
    for p in sorted(STUDENTS.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            continue
    return out
