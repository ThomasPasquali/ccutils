#!/usr/bin/env python3
"""Keep the TODO(<ID>) markers in the code and the items of docs/TODO.md in sync.

    python3 tools/check_todos.py          # exit status 1 on any mismatch
    python3 tools/check_todos.py --list   # show every marker with file:line

Rules: every `TODO(P<phase>-<name>)` in the source must have a `### P<phase>-<name>`
heading in docs/TODO.md, and every heading must be referenced by at least one marker
(delete both when the work is done).
"""
import re
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = re.compile(r"TODO\((P\d-[a-z0-9-]+)\)")
HEADING = re.compile(r"^### (P\d-[a-z0-9-]+)\s*$", re.M)
SUFFIXES = {".hpp", ".h", ".cpp", ".py", ".sh", ".yml", ".yaml", ".txt", ".cmake"}


def tracked_files():
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split("\n")
    return [ROOT / f for f in out if f and (ROOT / f).suffix in SUFFIXES]


def main() -> int:
    markers = {}
    for path in tracked_files():
        if path.name == "check_todos.py":
            continue
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            for ident in MARKER.findall(line):
                markers.setdefault(ident, []).append(f"{path.relative_to(ROOT)}:{number}")
    documented = set(HEADING.findall((ROOT / "docs" / "TODO.md").read_text()))

    if "--list" in sys.argv:
        for ident in sorted(markers):
            print(f"{ident:26s} {', '.join(markers[ident])}")
        return 0

    problems = []
    for ident in sorted(set(markers) - documented):
        problems.append(f"{ident}: used in {markers[ident][0]} but has no '### {ident}' in docs/TODO.md")
    for ident in sorted(documented - set(markers)):
        problems.append(f"{ident}: documented in docs/TODO.md but no TODO({ident}) marker in the code")
    for p in problems:
        print(p)
    print(f"{len(markers)} markers, {len(documented)} documented items, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # `| head` must not print a traceback
    sys.exit(main())
