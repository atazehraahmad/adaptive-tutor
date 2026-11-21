"""Housekeeping for student records.

    python manage.py list                 show every stored student
    python manage.py clear-demo           remove only the synthetic/seeded ones
    python manage.py clear-all            remove everything, real students included
    python manage.py remove <id>          remove one student by id
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import store


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    records = store.load_all()

    if cmd == "list":
        if not records:
            print("no students stored")
            return
        for r in records:
            kind = "demo" if r.get("synthetic") else "real"
            print(f"{r['id']:>8}  {kind:<5}  {r['name']:<22} "
                  f"{len(r.get('history', []))} answers")

    elif cmd in ("clear-demo", "clear-all"):
        gone = 0
        for r in records:
            if cmd == "clear-all" or r.get("synthetic"):
                (store.STUDENTS / f"{r['id']}.json").unlink(missing_ok=True)
                gone += 1
        kept = len(records) - gone
        print(f"removed {gone} student(s); {kept} remaining")
        if cmd == "clear-demo" and kept == 0:
            print("note: the teacher dashboard will now be empty until real "
                  "students answer questions, or you re-run: python seed.py 30")

    elif cmd == "remove" and len(sys.argv) > 2:
        target = sys.argv[2]
        p = store.STUDENTS / f"{target}.json"
        if p.exists():
            p.unlink()
            print(f"removed {target}")
        else:
            print(f"no student with id {target}")

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
