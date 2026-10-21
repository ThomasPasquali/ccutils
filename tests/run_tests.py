#!/usr/bin/env python3
"""Golden-output tests for ccutils.

Each case compiles a small program in one or more configurations ("variants"),
runs it (under mpirun for MPI cases), normalises the non-deterministic timing
numbers and compares stdout byte-for-byte with a stored golden file.

    python3 tests/run_tests.py              # run everything
    python3 tests/run_tests.py -k mpi       # only cases whose id contains "mpi"
    python3 tests/run_tests.py --update     # (re)generate golden files
    python3 tests/run_tests.py -v           # show compiler warnings and diffs

Environment: CXX (default g++), MPICXX (default mpicxx), MPIRUN (default mpirun),
MPIRUN_ARGS (extra launcher arguments; --oversubscribe is added automatically for
Open MPI), CCUTILS_TEST_WERROR=1 to compile with -Werror.
"""
import argparse
import difflib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GOLDEN = HERE / "golden"

VARIANTS = {
    "default": [],
    "pretty": ["-DCCUTILS_JSON_PRETTY_PRINT"],
    "nojson": ["-DCCUTILS_NO_JSON"],
}


@dataclass
class Case:
    name: str
    source: str
    mpi: bool
    variants: dict = field(default_factory=dict)  # variant -> list of process counts ([0] = not MPI)
    known_broken: dict = field(default_factory=dict)  # variant -> reason (skipped, reported)


# TODO(P0-ci): add the matrix of DESIGN.md 5.5 for framed all-print: np 7, empty
#   ranks, 1-byte / no-newline / multi-line / large payloads, pretty JSON (see also P0-golden-gaps).
#   Details: docs/TODO.md  (### P0-ci)
CASES = [
    Case("basic", "programs/golden_basic.cpp", False,
         {"default": [0], "pretty": [0], "nojson": [0]}),
    Case("mpi", "programs/golden_mpi.cpp", True,
         {"default": [1, 2, 4], "pretty": [2], "nojson": [2]}),
    Case("comm_print", "programs/golden_comm_print.cpp", True, {"default": [1]}),
    Case("mpi_timer_macros", "programs/golden_mpi_timer_macros.cpp", True, {"default": [1]}),
    Case("multifile", ["programs/multifile/main.cpp", "programs/multifile/solver.cpp",
                       "programs/multifile/report.cpp"], False, {"default": [0], "nojson": [0]}),
    Case("multifile_mpi", ["programs/multifile_mpi/main.cpp", "programs/multifile_mpi/worker.cpp"],
         True, {"default": [2], "nojson": [2]}),
    Case("emitter_unit", "programs/emitter_unit.cpp", False, {"default": [0]}),
    Case("machine_unit", "programs/machine_unit.cpp", False, {"default": [0]}),
    Case("mpi_section_unit", "programs/mpi_section_unit.cpp", True, {"default": [2]}),
]

# Timing values are the only non-deterministic part of the output.
TIMING = re.compile(r"((?:avg|stddev|min|max|sum|elapsed)_ms)=[-+0-9.eEinfa]+")
ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_SGR.sub("", text)


def mpirun_cmd(nproc: int):
    """Launcher command line; adds --oversubscribe for Open MPI."""
    mpirun = os.environ.get("MPIRUN", "mpirun")
    extra = shlex.split(os.environ.get("MPIRUN_ARGS", ""))
    try:
        probe = subprocess.run([mpirun, "--version"], capture_output=True, text=True,
                               env=dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT="1",
                                        OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1"))
        version = probe.stdout + probe.stderr
    except FileNotFoundError:
        sys.exit(f"error: launcher '{mpirun}' not found (set MPIRUN)")
    if ("Open MPI" in version or "OpenRTE" in version) and "--oversubscribe" not in extra:
        extra.append("--oversubscribe")
    return [mpirun, *extra, "-n", str(nproc)]


# TODO(P0-mpich): run the whole suite on MPICH where `mpirun -n 2` really gives 2 ranks (not possible in the
#   sandbox this was written in; see TODO.md ground rule 8).
#   Details: docs/TODO.md  (### P0-mpich)
# TODO(P0-ci): call this runner from the CI matrix (Open MPI, MPICH; g++, clang++; C++11/17/20).
#   Details: docs/TODO.md  (### P0-ci)
def check_mpi_launcher(nprocs, tmp: Path) -> bool:
    """Make sure mpicxx and mpirun belong together before trusting any output."""
    exe = tmp / "mpi_selfcheck"
    compiler = os.environ.get("MPICXX", "mpicxx")
    comp = subprocess.run([compiler, "-std=c++17", "programs/mpi_selfcheck.cpp", "-o", str(exe)],
                          cwd=HERE, capture_output=True, text=True)
    if comp.returncode != 0:
        print(f"error: cannot compile with '{compiler}':\n{comp.stderr}")
        return False
    env = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    for n in sorted(set(nprocs)):
        run = subprocess.run(mpirun_cmd(n) + [str(exe)], capture_output=True, text=True,
                             timeout=120, env=env)
        if run.stdout.strip() != f"world_size={n}":
            print(f"error: MPI launcher self-check failed for -n {n}: got {run.stdout.strip()!r}.\n"
                  f"       mpicxx ('{compiler}') and mpirun ('{os.environ.get('MPIRUN', 'mpirun')}') "
                  f"probably belong to different MPI installations or the launcher is broken.\n"
                  f"       Set MPICXX / MPIRUN (e.g. mpicxx.openmpi / mpirun.openmpi).\n{run.stderr}")
            return False
    return True


# Banners such as "File x.cpp Line 80" embed __LINE__; editing a test program must not
# invalidate the goldens.
LINENO = re.compile(r"\bLine \d+\b")


def normalise(text: str) -> str:
    return LINENO.sub("Line <n>", TIMING.sub(r"\1=<f>", text))

KNOWN_TYPES = {"section_start", "section_end", "data", "timer", "timer_sample", "timer_agg", "text", "run_start", "warning"}
JSONL_TIMING = re.compile(r'("(?:avg|sd|min|max|sum|value)":)[^,}]+')


def is_jsonl(name: str) -> bool:
    return name.startswith("m.jsonl")


def norm_file(name: str, text: str) -> str:
    text = normalise(text)
    if not is_jsonl(name):
        return text
    out = []
    for line in text.splitlines(True):
        if '"type":"run_start"' in line:
            line = re.sub(r'("host":)"(?:\\.|[^"\\])*"', r'\1"<host>"', line)
            line = re.sub(r'("slurm_job_id":)"(?:\\.|[^"\\])*"', r'\1"<job>"', line)
            line = re.sub(r'("library":)"(?:\\.|[^"\\])*"', r'\1"<mpi>"', line)
        if '"type":"timer' in line:   # timer, timer_sample, timer_agg: timing values only
            line = JSONL_TIMING.sub(r'\1"<f>"', line)
        out.append(line)
    return "".join(out)


def validate_jsonl(text: str) -> list:
    """Schema sanity of a machine stream: one JSON object per line, v == 2, known
    type, and monotonic per-process sequence numbers."""
    problems = []
    last_seq = {}
    for number, line in enumerate(text.splitlines(), 1):
        try:
            record = json.loads(line)
        except ValueError as exc:
            problems.append(f"line {number} is not JSON ({exc})")
            continue
        if record.get("v") != 2 or record.get("type") not in KNOWN_TYPES:
            problems.append(f"line {number}: bad v/type")
        if number == 1 and record.get("type") != "run_start":
            problems.append("first record is not run_start")
        if record.get("type") == "run_start":
            if not all(key in record for key in ("ccutils_version", "format", "host")):
                problems.append(f"line {number}: incomplete run_start")
            mpi = record.get("mpi")
            if mpi is not None and not all(
                    key in mpi for key in ("library", "world_size", "wtime_is_global", "wtick_s")):
                problems.append(f"line {number}: incomplete MPI run_start metadata")
        # A gathered machine file may contain records from several ranks. Sequence
        # numbers are per process, and rank text/timer records can be routed to
        # rank 0's file, so validate monotonicity per source rather than globally.
        source = record.get("rank") if record.get("rank") is not None else 0
        seq = record.get("seq")
        if record.get("type") == "run_start" and number > 1:
            # A gathered/merged file can include the start record from another
            # rank's sidecar; it opens that rank's independent sequence.
            last_seq[source] = seq if isinstance(seq, int) else last_seq.get(source, 0)
        elif not isinstance(seq, int) or seq <= last_seq.get(source, 0):
            problems.append(f"line {number}: non-increasing seq {seq} for source {source}")
        else:
            last_seq[source] = seq
    return problems



def compile_case(case: Case, variant: str, outdir: Path):
    exe = outdir / f"{case.name}.{variant}"
    compiler = os.environ.get("MPICXX", "mpicxx") if case.mpi else os.environ.get("CXX", "g++")
    cmd = [compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Wpedantic",
           f"-I{REPO / 'include'}", *VARIANTS[variant]]
    if case.mpi:
        # Open MPI's C++ compatibility header triggers warnings under -Wpedantic.
        # Treat wrapper-provided include directories as system headers.
        try:
            probe = subprocess.run([compiler, "--showme:incdirs"], capture_output=True,
                                   text=True, timeout=10)
            if probe.returncode == 0:
                for include_dir in shlex.split(probe.stdout):
                    cmd.extend(["-isystem", include_dir])
        except (OSError, subprocess.TimeoutExpired):
            pass
        cmd.append("-DCCUTILS_ENABLE_MPI")
    if os.environ.get("CCUTILS_TEST_WERROR"):
        cmd.append("-Werror")
    sources = case.source if isinstance(case.source, list) else [case.source]
    cmd += [*sources, "-o", str(exe)]
    # Relative source path keeps __FILE__ (printed by some macros) stable.
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    return exe, proc


def run_case(exe: Path, case: Case, nproc: int, workdir: Path):
    env = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
               NO_COLOR="")
    cmd = mpirun_cmd(nproc) + [str(exe)] if case.mpi else [str(exe)]
    return subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=120, env=env)


@dataclass
class Scenario:
    """Re-runs an existing case with extra environment and derives the expected
    output from that case's golden file (so scenarios cannot drift from it)."""
    case: str
    variant: str
    nproc: int
    suffix: str
    env: dict
    expect: object  # callable: base golden text -> dict(stdout, files, stderr | stderr_contains)
    prefill: dict = field(default_factory=dict)
    record: dict = field(default_factory=dict)  # produced file -> golden name (written by --update)

    @property
    def test_id(self):
        return f"{self.case}.{self.variant}" + (f".np{self.nproc}" if self.nproc else "") + f".{self.suffix}"

    @property
    def base_id(self):
        return f"{self.case}.{self.variant}" + (f".np{self.nproc}" if self.nproc else "")


def _mpi_sidecar(base: str) -> dict:
    """np=2 golden split for CCUTILS_HUMAN_OUT=file:out.txt: rank 0 writes out.txt,
    rank 1's timer and routed process-print banners write to out.txt.r1, while the
    block's own stdout line stays on stdout."""
    lines = base.splitlines(True)
    i = next(k for k, l in enumerate(lines) if l.startswith("--------------------- Proc "))
    j = next(k for k, l in enumerate(lines) if 'name="root_last"' in l)
    rest = [l for k, l in enumerate(lines) if k not in (i, i + 1, i + 2, j)]
    return {"stdout": lines[i + 1],
            "files": {"out.txt": "".join(rest), "out.txt.r1": lines[j] + lines[i] + lines[i + 2]}}


def golden_text(name: str) -> str:
    return (GOLDEN / name).read_text()


SCENARIOS = [
    Scenario("basic", "default", 0, "machine_file", {"CCUTILS_MACHINE_OUT": "file:m.jsonl"},
             lambda b: {"stdout": b, "files": {"m.jsonl": golden_text("basic.default.machine.jsonl")}},
             record={"m.jsonl": "basic.default.machine.jsonl"}),
    Scenario("basic", "nojson", 0, "machine_file", {"CCUTILS_MACHINE_OUT": "file:m.jsonl"},
             lambda b: {"stdout": b, "files": {"m.jsonl": golden_text("basic.nojson.machine.jsonl")}},
             record={"m.jsonl": "basic.nojson.machine.jsonl"}),
    Scenario("multifile", "default", 0, "machine_file", {"CCUTILS_MACHINE_OUT": "file:m.jsonl"},
             lambda b: {"stdout": b, "files": {"m.jsonl": golden_text("multifile.default.machine.jsonl")}},
             record={"m.jsonl": "multifile.default.machine.jsonl"}),
    Scenario("multifile_mpi", "default", 2, "machine_file", {"CCUTILS_MACHINE_OUT": "file:m.jsonl"},
             lambda b: {"stdout": b, "files": {"m.jsonl": golden_text("multifile_mpi.default.np2.machine.jsonl")}},
             record={"m.jsonl": "multifile_mpi.default.np2.machine.jsonl"}),
    Scenario("basic", "default", 0, "machine_inline", {"CCUTILS_MACHINE_OUT": "inline"},
             lambda b: {"stdout": None, "files": {}}),
    Scenario("mpi", "default", 2, "machine_file", {"CCUTILS_MACHINE_OUT": "file:m.jsonl"},
             lambda b: {"stdout": b, "files": {
                 "m.jsonl": golden_text("mpi.default.np2.machine.jsonl"),
                 "m.jsonl.r1": golden_text("mpi.default.np2.machine.r1.jsonl")}},
             record={"m.jsonl": "mpi.default.np2.machine.jsonl",
                     "m.jsonl.r1": "mpi.default.np2.machine.r1.jsonl"}),
    Scenario("basic", "default", 0, "human_file", {"CCUTILS_HUMAN_OUT": "file:out.txt"},
             lambda b: {"stdout": "", "files": {"out.txt": b}}),
    Scenario("basic", "default", 0, "human_append", {"CCUTILS_HUMAN_OUT": "append:out.txt"},
             lambda b: {"stdout": "", "files": {"out.txt": "PRE\n" + b}},
             prefill={"out.txt": "PRE\n"}),
    Scenario("basic", "default", 0, "human_none", {"CCUTILS_HUMAN_OUT": "none"},
             lambda b: {"stdout": "", "files": {}}),
    Scenario("basic", "default", 0, "human_stderr", {"CCUTILS_HUMAN_OUT": "stderr"},
             lambda b: {"stdout": "", "files": {}, "stderr": b}),
    Scenario("basic", "default", 0, "color_never_stdout", {"CCUTILS_COLOR": "never"},
             lambda b: {"stdout": strip_ansi(b), "files": {}}),
    Scenario("basic", "default", 0, "color_never_file",
             {"CCUTILS_COLOR": "never", "CCUTILS_HUMAN_OUT": "file:out.txt"},
             lambda b: {"stdout": "", "files": {"out.txt": strip_ansi(b)}}),
    Scenario("basic", "default", 0, "color_never_stderr",
             {"CCUTILS_COLOR": "never", "CCUTILS_HUMAN_OUT": "stderr"},
             lambda b: {"stdout": "", "files": {}, "stderr": strip_ansi(b)}),
    Scenario("basic", "default", 0, "no_color_env", {"CCUTILS_COLOR": "always", "NO_COLOR": "1"},
             lambda b: {"stdout": strip_ansi(b), "files": {}}),
    Scenario("basic", "default", 0, "bad_spec", {"CCUTILS_HUMAN_OUT": "bogus"},
             lambda b: {"stdout": b, "files": {}, "stderr_contains": "CCUTILS_HUMAN_OUT"}),
    Scenario("basic", "default", 0, "format_v2", {"CCUTILS_FORMAT": "v2"},
             lambda b: {"stdout": b, "files": {}, "stderr_contains": "not implemented"}),
    Scenario("mpi", "default", 2, "human_file_sidecar", {"CCUTILS_HUMAN_OUT": "file:out.txt"},
             _mpi_sidecar),
    Scenario("mpi", "default", 2, "flush_wait_us", {"CCUTILS_FLUSH_WAIT_US": "1"},
             lambda b: {"stdout": b, "files": {}}),
    Scenario("comm_print", "default", 1, "human_file",
             {"CCUTILS_HUMAN_OUT": "file:out.txt"},
             lambda b: {"stdout": "", "files": {"out.txt": b}}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="", help="substring filter on test ids")
    ap.add_argument("--update", action="store_true", help="rewrite golden files")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    GOLDEN.mkdir(exist_ok=True)
    failures, ran, warnings = 0, 0, 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "bin").mkdir()
        (tmp / "run").mkdir()
        mpi_nprocs = [n for c in CASES if c.mpi for v, ns in c.variants.items()
                      if v not in c.known_broken
                      and (not args.k or any(args.k in f"{c.name}.{v}.np{n}" for n in ns)) for n in ns]
        if mpi_nprocs and not check_mpi_launcher(mpi_nprocs, tmp / "bin"):
            return 2
        built = {}
        for case in CASES:
            for variant, nprocs in case.variants.items():
                if args.k and not any(args.k in f"{case.name}.{variant}.np{n}" for n in nprocs):
                    continue
                if variant in case.known_broken:
                    print(f"SKIP  {case.name}.{variant}: known bug: {case.known_broken[variant]}")
                    continue
                exe, cproc = compile_case(case, variant, tmp / "bin")
                warnings += cproc.stderr.count("warning:")
                if cproc.returncode != 0:
                    print(f"FAIL  {case.name}.{variant}: compilation failed\n{cproc.stderr}")
                    failures += 1
                    continue
                built[(case.name, variant)] = exe
                if args.verbose and cproc.stderr:
                    print(f"--- compiler output for {case.name}.{variant}:\n{cproc.stderr}")
                for n in nprocs:
                    test_id = f"{case.name}.{variant}" + (f".np{n}" if n else "")
                    ran += 1
                    workdir = tmp / "run" / test_id
                    workdir.mkdir()
                    try:
                        rproc = run_case(exe, case, n, workdir)
                    except subprocess.TimeoutExpired:
                        print(f"FAIL  {test_id}: timeout")
                        failures += 1
                        continue
                    if rproc.returncode != 0:
                        print(f"FAIL  {test_id}: exit code {rproc.returncode}\n{rproc.stderr}")
                        failures += 1
                        continue
                    if rproc.stderr.strip():
                        print(f"FAIL  {test_id}: unexpected stderr output:\n{rproc.stderr}")
                        failures += 1
                        continue
                    leftovers = sorted(p.name for p in workdir.iterdir())
                    if leftovers:
                        print(f"FAIL  {test_id}: left files behind: {leftovers}")
                        failures += 1
                        continue
                    actual = normalise(rproc.stdout)
                    gfile = GOLDEN / f"{test_id}.txt"
                    if args.update:
                        gfile.write_text(actual)
                        print(f"WROTE {gfile.relative_to(REPO)}")
                        continue
                    if not gfile.exists():
                        print(f"FAIL  {test_id}: missing golden file (run with --update)")
                        failures += 1
                        continue
                    expected = gfile.read_text()
                    if actual == expected:
                        print(f"ok    {test_id}")
                    else:
                        failures += 1
                        print(f"FAIL  {test_id}: output differs from {gfile.relative_to(REPO)}")
                        if args.verbose:
                            diff = difflib.unified_diff(
                                expected.splitlines(True), actual.splitlines(True),
                                "golden", "actual")
                            sys.stdout.writelines(diff)

        if True:
            cases_by_name = {c.name: c for c in CASES}
            for sc in SCENARIOS:
                if args.k and args.k not in sc.test_id:
                    continue
                if args.update and not sc.record:
                    continue
                case = cases_by_name[sc.case]
                if (sc.case, sc.variant) not in built:
                    exe, cproc = compile_case(case, sc.variant, tmp / "bin")
                    if cproc.returncode != 0:
                        print(f"FAIL  {sc.test_id}: compilation failed\n{cproc.stderr}")
                        failures += 1
                        continue
                    built[(sc.case, sc.variant)] = exe
                exe = built[(sc.case, sc.variant)]
                ran += 1
                workdir = tmp / "run" / sc.test_id
                workdir.mkdir()
                for name, content in sc.prefill.items():
                    (workdir / name).write_text(content)
                env = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
                env["NO_COLOR"] = ""
                env.update(sc.env)
                cmd = (mpirun_cmd(sc.nproc) + [str(exe)]) if case.mpi else [str(exe)]
                try:
                    r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=120, env=env)
                except subprocess.TimeoutExpired:
                    print(f"FAIL  {sc.test_id}: timeout")
                    failures += 1
                    continue
                if args.update:
                    for produced, gname in sc.record.items():
                        (GOLDEN / gname).write_text(norm_file(produced, (workdir / produced).read_text()))
                        print(f"WROTE tests/golden/{gname}")
                    continue
                exp = sc.expect((GOLDEN / f"{sc.base_id}.txt").read_text())
                problems = []
                if r.returncode != 0:
                    problems.append(f"exit code {r.returncode}")
                if exp["stdout"] is None:
                    # inline machine records are mixed into the human stream: the
                    # "@ccutils " lines must form a valid machine stream, and removing
                    # them must give back the plain human output
                    lines = r.stdout.splitlines(True)
                    inline = "".join(l[len("@ccutils "):] for l in lines if l.startswith("@ccutils "))
                    rest = "".join(l for l in lines if not l.startswith("@ccutils "))
                    problems += validate_jsonl(inline)
                    if normalise(rest) != (GOLDEN / f"{sc.base_id}.txt").read_text():
                        problems.append("human output changed by inline records")
                    if not inline:
                        problems.append("no inline records")
                elif normalise(r.stdout) != exp["stdout"]:
                    problems.append("stdout differs")
                if "stderr" in exp:
                    if normalise(r.stderr) != exp["stderr"]:
                        problems.append("stderr differs")
                elif "stderr_contains" in exp:
                    if exp["stderr_contains"] not in r.stderr:
                        problems.append(f"stderr lacks {exp['stderr_contains']!r}")
                elif r.stderr.strip():
                    problems.append("unexpected stderr")
                present = sorted(p.name for p in workdir.iterdir())
                if present != sorted(exp["files"]):
                    problems.append(f"files {present} != {sorted(exp['files'])}")
                else:
                    for name, content in exp["files"].items():
                        actual_text = (workdir / name).read_text()
                        if norm_file(name, actual_text) != content:
                            problems.append(f"file {name} differs")
                        if is_jsonl(name):
                            problems += [f"{name}: {m}" for m in validate_jsonl(actual_text)]
                if problems:
                    failures += 1
                    print(f"FAIL  {sc.test_id}: " + "; ".join(problems))
                    if args.verbose:
                        print(r.stderr[:500])
                else:
                    print(f"ok    {sc.test_id}")

    print(f"\n{ran} run, {failures} failed, {warnings} compiler warnings")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
