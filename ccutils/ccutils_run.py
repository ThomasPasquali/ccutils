"""ccutils parser v2: one model for every output format.

    import ccutils
    run = ccutils.load("run.jsonl")     # JSON Lines, "@ccutils " inline lines, or a legacy log
    run.tree()                          # what is in here
    run["solve/spmv"].timers["kernel"].avg_ms
    run["solve"].data["residual"]
    run.timers_table()                  # list of dicts (or run.timers_df() with pandas)

The format is detected automatically. Every front end produces records of the
schema described in docs/FORMAT.md and the model is built from those records, so the
same queries work whatever the input was. Sidecar files ``<file>.r<N>`` written by
other MPI ranks are merged automatically.
"""
from __future__ import annotations

import difflib
import base64
import glob
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

try:  # package use
    from . import ccutils_parser as _legacy
except ImportError:  # script / flat use
    import ccutils_parser as _legacy

SCHEMA_VERSION = 2
KNOWN_TYPES = {"run_start", "section_start", "section_end", "data", "json",
               "data_batch", "mpi_json_batch", "mpi_json_rows", "timer",
               "timer_sample", "timer_samples", "timer_agg", "text", "warning"}
_RANK_JSON_BLOCKS = {"rank_data", "ccutils_rank_json", "ccutils_subsection_json"}


class FormatError(ValueError):
    """The input is not in a format this parser understands."""


# --------------------------------------------------------------------------- model

@dataclass
class TimerSummary:
    """Statistics of one timer report. Times are in `unit` (milliseconds)."""
    name: str
    group: str = ""
    scope: str = "rank"            # "rank" (one process) or "comm" (pooled over a communicator)
    rank: Optional[int] = None
    unit: str = "ms"
    clock: str = ""
    n: int = 0
    avg: Optional[float] = None
    sd: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    sum: Optional[float] = None
    comm: Optional[Dict[str, Any]] = None      # {"name", "size", "root"} for scope "comm"
    source: str = ""
    seq: Optional[int] = None
    per_rank: Any = None
    rank_stats: Any = None
    imbalance: Optional[float] = None
    slowest_rank: Optional[int] = None
    ranks_reporting: Optional[int] = None
    errors: Optional[int] = None

    def _ms(self, value):
        if value is None or self.unit == "ms":
            return value
        raise ValueError(f"timer {self.name!r} is in unit {self.unit!r}, not milliseconds")

    avg_ms = property(lambda self: self._ms(self.avg))
    sd_ms = property(lambda self: self._ms(self.sd))
    min_ms = property(lambda self: self._ms(self.min))
    max_ms = property(lambda self: self._ms(self.max))
    sum_ms = property(lambda self: self._ms(self.sum))


@dataclass
class Timer:
    """Everything reported under one timer name inside a section."""
    name: str
    summaries: List[TimerSummary] = field(default_factory=list)
    samples: List[Dict[str, Any]] = field(default_factory=list)   # timer_sample records
    aggregates: List[Dict[str, Any]] = field(default_factory=list)  # timer_agg records

    @property
    def last(self) -> Optional[TimerSummary]:
        """The most recent report (a timer may be printed several times)."""
        return self.summaries[-1] if self.summaries else None

    def _need(self) -> TimerSummary:
        if not self.summaries:
            raise AttributeError(f"timer {self.name!r} has no summary report")
        return self.summaries[-1]

    n = property(lambda self: self._need().n)
    avg = property(lambda self: self._need().avg)
    sd = property(lambda self: self._need().sd)
    min = property(lambda self: self._need().min)
    max = property(lambda self: self._need().max)
    sum = property(lambda self: self._need().sum)
    avg_ms = property(lambda self: self._need().avg_ms)
    sd_ms = property(lambda self: self._need().sd_ms)
    min_ms = property(lambda self: self._need().min_ms)
    max_ms = property(lambda self: self._need().max_ms)
    sum_ms = property(lambda self: self._need().sum_ms)

    @property
    def comm(self) -> Optional[TimerSummary]:
        """Most recent report pooled over a communicator (MPI timers), if any."""
        pooled = [s for s in self.summaries if s.scope == "comm"]
        return pooled[-1] if pooled else None

    def by_rank(self) -> Dict[Optional[int], TimerSummary]:
        """Most recent per-process report for every rank (None = rank unknown)."""
        out: Dict[Optional[int], TimerSummary] = {}
        for s in self.summaries:
            if s.scope == "rank":
                out[s.rank] = s
        return out

    def per_rank(self, kind: str = "sum"):
        """Return per-rank timer values as a pandas Series.

        `kind` is one of `n`, `avg`, `sd`, `min`, `max`, or `sum`.
        Requires pandas.
        """
        if kind not in ("n", "avg", "sd", "min", "max", "sum"):
            raise ValueError("kind must be n, avg, sd, min, max, or sum")
        ranked = self.by_rank()
        values = {rank: getattr(summary, kind) for rank, summary in ranked.items()}
        if not values and self.comm and isinstance(self.comm.rank_stats, list):
            values = {int(item["rank"]): item[kind] for item in self.comm.rank_stats
                      if kind in item}
        try:
            import pandas
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("pandas is required for Timer.per_rank()") from exc
        return pandas.Series(values, name=f"{self.name}.{kind}").sort_index()


@dataclass
class RankText:
    """Text emitted by one rank in an all-ranks print call."""
    lines: List[str] = field(default_factory=list)
    trailing_newline: bool = True
    _parts: Dict[int, bytes] = field(default_factory=dict, repr=False)
    _last_seen: bool = field(default=False, repr=False)
    _last_part: Optional[int] = field(default=None, repr=False)

    @property
    def text(self) -> str:
        return "\n".join(self.lines) + ("\n" if self.trailing_newline and self.lines else "")

    def json(self) -> List[Any]:
        """JSON values in this text: the whole text if it is one value (pretty-printed
        or not), otherwise every line that is one."""
        try:
            return [json.loads(self.text)]
        except ValueError:
            pass
        found = []
        for line in self.lines:
            try:
                found.append(json.loads(line))
            except ValueError:
                continue
        return found


@dataclass
class AllPrint:
    """One call of an all-ranks print: the text of every rank that printed something."""
    name: Optional[str]
    call: Optional[int]
    comm: Optional[str] = None
    ranks: Dict[int, RankText] = field(default_factory=dict)

    @property
    def jsons(self) -> Dict[int, Any]:
        """First JSON value printed by each rank that emitted valid JSON."""
        return {rank: values[0] for rank, output in sorted(self.ranks.items())
                if (values := output.json())}

    @property
    def json_values(self) -> Dict[int, List[Any]]:
        """All JSON values printed by each rank that emitted valid JSON."""
        return {rank: values for rank, output in sorted(self.ranks.items())
                if (values := output.json())}

    def grep(self, pattern: str, flags: int = 0) -> Dict[int, List[str]]:
        """Return matching output lines grouped by rank; `pattern` is a regex."""
        rx = re.compile(pattern, flags)
        return {rank: [line for line in text.lines if rx.search(line)]
                for rank, text in self.ranks.items()}

    def diff_ranks(self, left: int, right: int) -> str:
        """Return a unified text diff between two ranks."""
        a = self.ranks.get(left, RankText()).lines
        b = self.ranks.get(right, RankText()).lines
        return "\n".join(difflib.unified_diff(a, b, fromfile=f"rank {left}",
                                               tofile=f"rank {right}", lineterm=""))


class PrintCalls(dict):
    def calls(self, name: str) -> List[AllPrint]:
        """Return all calls of the named MPI print, in input order."""
        return list(self.get(name, []))


class Section:
    """Parsed section with global/rank data, timers, prints, and child sections."""

    def __init__(self, name: str, path: str, parent: Optional["Section"] = None):
        self.name = name
        self.path = path
        self.parent = parent
        self.title = ""
        self.children: List["Section"] = []
        self.started = False
        self.complete = False
        self.ok: Optional[bool] = None
        self.implicit = False                       # created only because a record referred to it
        self.data: Dict[str, Any] = {}              # global values (last write wins)
        self.rank_data: Dict[Optional[int], Dict[str, Any]] = {}
        self.timers: Dict[str, Timer] = {}
        self.prints: PrintCalls = PrintCalls()

    @property
    def global_data(self) -> Dict[str, Any]:
        """Global JSON values (alias for ``data``)."""
        return self.data

    def child(self, name: str) -> Optional["Section"]:
        """Return the direct child named `name`, or None."""
        return next((c for c in self.children if c.name == name), None)

    def __getitem__(self, path: str) -> "Section":
        """Look up a descendant by slash-separated path relative to this section."""
        node: Section = self
        for part in path.split("/"):
            found = node.child(part)
            if found is None:
                raise KeyError(f"{self.path or '<root>'}: no section {path!r} "
                               f"(children: {[c.name for c in node.children]})")
            node = found
        return node

    def walk(self) -> Iterator["Section"]:
        """Iterate over this section and all descendants, depth first."""
        yield self
        for c in self.children:
            yield from c.walk()

    def per_rank(self, key: str) -> Dict[Optional[int], Any]:
        """Value of a rank-local `key` for every rank that has it."""
        return {r: d[key] for r, d in self.rank_data.items() if key in d}

    def __repr__(self) -> str:
        return f"Section({self.path or '<outside>'!r}, children={len(self.children)}, " \
               f"data={list(self.data)}, timers={list(self.timers)})"


class Run:
    """The whole output of one program run."""

    def __init__(self, format: str = "jsonl", sources: Optional[List[str]] = None):
        """Create an empty run model. Usually use `load()` or `loads()` instead."""
        self.format = format
        self.sources: List[str] = list(sources or [])
        self.warnings: List[str] = []
        self.lossy: List[str] = []
        self.record_count = 0
        self.unknown_types: Dict[str, int] = {}
        self.roots: List[Section] = []
        self.outside = Section("", "")               # records that belong to no section
        self._by_path: Dict[str, List[Section]] = {}
        self._open: Dict[str, Section] = {}
        self._seq: Dict[str, int] = {}
        self._warned_version = False

    # ---- queries
    def __getitem__(self, path: str) -> Section:
        """Return the first section at `path`; raise KeyError if absent."""
        found = self._by_path.get(path)
        if not found:
            close = difflib.get_close_matches(path, list(self._by_path), n=3)
            hint = f" Did you mean {close}?" if close else ""
            raise KeyError(f"no section {path!r}.{hint}")
        return found[0]

    def __contains__(self, path: str) -> bool:
        """Return whether a section exists at `path`."""
        return path in self._by_path

    def all(self, path: str) -> List[Section]:
        """Return every section at `path`, including repeated sections."""
        return list(self._by_path.get(path, []))

    def sections(self) -> Iterator[Section]:
        """Iterate over all sections in document order."""
        for root in self.roots:
            yield from root.walk()

    def find(self, name: str) -> List[Section]:
        """Return all sections named `name`, at any depth."""
        return [s for s in self.sections() if s.name == name]

    # ---- tables
    def timers_table(self) -> List[Dict[str, Any]]:
        """One row per timer report: section, name, group, scope, rank, n, avg, ... (ms)."""
        rows = []
        for sec in [self.outside, *self.sections()]:
            for timer in sec.timers.values():
                for s in timer.summaries:
                    rows.append({
                        "section": sec.path or None, "name": s.name, "group": s.group,
                        "scope": s.scope, "rank": s.rank, "n": s.n, "avg": s.avg, "sd": s.sd,
                        "min": s.min, "max": s.max, "sum": s.sum, "unit": s.unit,
                        "clock": s.clock,
                        "comm_size": (s.comm or {}).get("size"),
                        "comm_root": (s.comm or {}).get("root"),
                        "per_rank": s.per_rank, "imbalance": s.imbalance,
                        "slowest_rank": s.slowest_rank,
                        "ranks_reporting": s.ranks_reporting, "errors": s.errors,
                    })
        return rows

    def data_table(self) -> List[Dict[str, Any]]:
        """One row per data value: section, scope, rank, key, value."""
        rows = []
        for sec in [self.outside, *self.sections()]:
            for key, value in sec.data.items():
                rows.append({"section": sec.path or None, "scope": "global", "rank": None,
                             "key": key, "value": value})
            for rank, values in sec.rank_data.items():
                for key, value in values.items():
                    rows.append({"section": sec.path or None, "scope": "rank", "rank": rank,
                                 "key": key, "value": value})
        return rows

    def timers_df(self, level: Optional[str] = None):
        """Return timer reports as a pandas DataFrame, optionally filtered by scope."""
        if level is not None and level not in ("comm", "rank"):
            raise ValueError("level must be 'comm' or 'rank'")
        rows = self.timers_table()
        return _dataframe(rows if level is None else [r for r in rows if r["scope"] == level])

    def data_df(self):
        """Return global and rank-local data as a pandas DataFrame."""
        return _dataframe(self.data_table())

    # ---- inspection
    def tree(self, file=None) -> str:
        """Return a text outline and print it to `file` (stdout by default).

        Pass `file=False` to return the outline without printing.
        """
        lines = [f"{', '.join(self.sources) or '<string>'}  "
                 f"({self.format}, {self.record_count} records, {len(self.warnings)} warnings)"]

        def describe(sec: Section, prefix: str):
            items = []
            if sec.data:
                items.append("data: " + ", ".join(sec.data))
            for rank in sorted(sec.rank_data, key=lambda r: (r is None, r)):
                items.append(f"rank {rank} data: " + ", ".join(sec.rank_data[rank]))
            if sec.timers:
                items.append("timers: " + ", ".join(sec.timers))
            if sec.prints:
                items.append("prints: " + ", ".join(sec.prints))
            entries = items + sec.children
            for i, entry in enumerate(entries):
                last = i == len(entries) - 1
                branch, cont = ("└─ ", "   ") if last else ("├─ ", "│  ")
                if isinstance(entry, str):
                    lines.append(prefix + branch + entry)
                else:
                    lines.append(prefix + branch + _section_label(entry))
                    describe(entry, prefix + cont)

        if self.outside.timers or self.outside.data:
            lines.append("(outside sections)")
            describe(self.outside, "")
        for root in self.roots:
            lines.append(_section_label(root))
            describe(root, "")
        text = "\n".join(lines)
        if file is not False:
            print(text, file=file)
        return text

    def check(self) -> List[str]:
        """Return warnings and structural problems found while parsing the run."""
        problems = list(self.warnings)
        for sec in self.sections():
            if sec.started and not sec.complete:
                problems.append(f"section '{sec.path}' was never closed (truncated log?)")
            if sec.ok is False:
                problems.append(f"section '{sec.path}' ended abnormally")
        for name, count in sorted(self.unknown_types.items()):
            problems.append(f"{count} record(s) of unknown type {name!r} ignored")
        return problems

    # ---- building (used by the front ends)
    def _warn(self, message: str) -> None:
        self.warnings.append(message)

    def _implicit(self, path: str) -> Section:
        parent = self._target(path.rpartition("/")[0]) if "/" in path else None
        sec = self._new_section(path, parent)
        sec.implicit = True
        self._warn(f"section '{path}' referenced before it was opened")
        return sec

    def _new_section(self, path: str, parent: Optional[Section]) -> Section:
        sec = Section(path.rsplit("/", 1)[-1], path, parent)
        (parent.children if parent is not None else self.roots).append(sec)
        self._by_path.setdefault(path, []).append(sec)
        return sec

    def _target(self, path: Optional[str]) -> Section:
        if not path:
            return self.outside
        sec = self._open.get(path)
        if sec is None:
            existing = self._by_path.get(path)
            sec = existing[-1] if existing else self._implicit(path)
        return sec

    def add_record(self, rec: Dict[str, Any], source: str = "") -> None:
        if not isinstance(rec, dict):
            return
        self.record_count += 1
        if isinstance(rec.get("v"), int) and rec["v"] > SCHEMA_VERSION and not self._warned_version:
            self._warned_version = True
            self._warn(f"records of schema version {rec['v']} are newer than this parser "
                       f"({SCHEMA_VERSION}); unknown fields are ignored")
        seq = rec.get("seq")
        if isinstance(seq, int) and source:
            rank = rec.get("rank")
            if rank is None:
                match = re.search(r"\.r(\d+)$", source)
                rank = int(match.group(1)) if match else 0
            seq_source = f"{source}#rank={rank}"
            previous = self._seq.get(seq_source)
            if previous is not None and (seq <= previous or (rank == 0 and seq != previous + 1)):
                expected = "a larger value" if rank != 0 else str(previous + 1)
                self._warn(f"{seq_source}: record seq {seq}, expected {expected} (lost lines?)")
            self._seq[seq_source] = seq
        kind = rec.get("type")
        path = rec.get("section") or None
        if kind == "section_start":
            parent = self._target(path.rpartition("/")[0]) if path and "/" in path else None
            sec = self._new_section(path, parent)
            sec.title = rec.get("title", "")
            sec.started = True
            self._open[path] = sec
        elif kind == "section_end":
            sec = self._open.pop(path, None) if path else None
            if sec is None:
                self._warn(f"section_end for '{path}' without a matching section_start")
            else:
                sec.complete = True
                sec.ok = rec.get("ok", True)
        elif kind == "data":
            self._add_data(self._target(path), rec)
        elif kind == "data_batch":
            self._add_data_batch(self._target(path), rec)
        elif kind == "json":
            value = rec.get("value")
            if self._add_rank_data_print(path, rec.get("print"), rec.get("rank"), value):
                return
            trailing_newline = rec.get("trailing_newline", True)
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            if trailing_newline:
                text += "\n"
            self._add_text(self._target(path), {
                "print": rec.get("print"), "call": rec.get("call"),
                "comm": rec.get("comm"), "rank": rec.get("rank"),
                "part": 0, "last": True, "trailing_newline": trailing_newline,
                "text": text,
            })
        elif kind in ("mpi_json_batch", "mpi_json_rows"):
            schema = rec.get("schema") if kind == "mpi_json_batch" else None

            def expand_json_schema(node, values):
                if not isinstance(node, list) or len(node) != 2:
                    raise ValueError("invalid schema node")
                kind_name, body = node
                if kind_name == "object" and isinstance(body, list):
                    if not isinstance(values, list) or len(values) != len(body):
                        raise ValueError("object row does not match its schema")
                    result = {}
                    for field, value in zip(body, values):
                        if not isinstance(field, list) or len(field) != 2 or not isinstance(field[0], str):
                            raise ValueError("invalid object field")
                        result[field[0]] = expand_json_schema(field[1], value)
                    return result
                if kind_name == "array" and isinstance(values, list):
                    return values if body is None else [expand_json_schema(body, item) for item in values]
                if kind_name == "array_raw" and body is None and isinstance(values, list):
                    return values
                if kind_name == "value" and isinstance(body, str):
                    return values
                raise ValueError("unknown schema node")

            for row in rec.get("rows", []):
                if kind == "mpi_json_rows":
                    obj = row.get("value")
                    if "value" not in row:
                        self._warn("mpi_json_rows row has no JSON value")
                        continue
                else:
                    values = row.get("values", [])
                    if not isinstance(values, list):
                        self._warn("mpi_json_batch row values are not an array")
                        continue
                    try:
                        obj = expand_json_schema(schema, values)
                    except (TypeError, ValueError):
                        self._warn("mpi_json_batch row does not match its schema")
                        continue
                if self._add_rank_data_print(
                        path, row.get("print", rec.get("print")), row.get("rank"), obj):
                    continue
                trailing_newline = row.get("trailing_newline", rec.get("trailing_newline", True))
                text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                if trailing_newline:
                    text += "\n"
                self._add_text(self._target(path), {
                    "print": row.get("print", rec.get("print")),
                    "call": row.get("call", rec.get("call")),
                    "comm": row.get("comm", rec.get("comm")), "rank": row.get("rank"),
                    "part": 0, "last": True, "trailing_newline": trailing_newline,
                    "text": text,
                })
        elif kind == "timer":
            self._add_timer(self._target(path), rec, source)
        elif kind == "timer_sample":
            self._timer(self._target(path), rec["name"]).samples.append(
                {k: rec.get(k) for k in ("group", "rank", "index", "value", "last", "unit")})
        elif kind == "timer_samples":
            self._timer(self._target(path), rec["name"]).samples.append(
                {"rank": rec.get("rank"), "part": rec.get("part"),
                 "values": rec.get("values", []), "last": rec.get("last"),
                 "unit": rec.get("unit", "ms")})
        elif kind == "timer_agg":
            self._timer(self._target(path), rec["name"]).aggregates.append(
                {k: rec.get(k) for k in ("rank", "op", "value", "unit")})
        elif kind == "text":
            self._add_text(self._target(path), rec)
        elif kind == "run_start":
            pass
        elif kind == "warning":
            self._warn(str(rec.get("message", "machine-stream warning")))
        else:
            name = str(kind)
            self.unknown_types[name] = self.unknown_types.get(name, 0) + 1

    def _add_data(self, sec: Section, rec: Dict[str, Any]) -> None:
        if rec.get("scope") == "rank":
            sec.rank_data.setdefault(rec.get("rank"), {})[rec["key"]] = rec.get("value")
        else:
            sec.data[rec["key"]] = rec.get("value")

    def _add_rank_data_print(self, path: Optional[str], print_name: Optional[str],
                             rank: Optional[int], value: Any) -> bool:
        """Normalize the internal section-close MPI JSON print into rank data."""
        if print_name != "rank_data" or not isinstance(value, dict):
            return False
        self._target(path).rank_data.setdefault(rank, {}).update(value)
        return True

    def _add_data_batch(self, sec: Section, rec: Dict[str, Any]) -> None:
        values = rec.get("values")
        if not isinstance(values, dict):
            self._warn("data_batch record has no object in 'values'")
            return
        if rec.get("scope") == "rank":
            sec.rank_data.setdefault(rec.get("rank"), {}).update(values)
        else:
            sec.data.update(values)

    @staticmethod
    def _timer(sec: Section, name: str) -> Timer:
        return sec.timers.setdefault(name, Timer(name))

    def _add_timer(self, sec: Section, rec: Dict[str, Any], source: str) -> None:
        summary = TimerSummary(
            name=rec["name"], group=rec.get("group", ""), scope=rec.get("scope", "rank"),
            rank=rec.get("rank"), unit=rec.get("unit", "ms"), clock=rec.get("clock", ""),
            n=rec.get("n", 0), avg=rec.get("avg"), sd=rec.get("sd"), min=rec.get("min"),
            max=rec.get("max"), sum=rec.get("sum"), comm=rec.get("comm"),
            per_rank=rec.get("per_rank"), imbalance=rec.get("imbalance"),
            rank_stats=rec.get("rank_stats"),
            slowest_rank=rec.get("slowest_rank"), ranks_reporting=rec.get("ranks_reporting"),
            errors=rec.get("errors"),
            source=source, seq=rec.get("seq"))
        self._timer(sec, summary.name).summaries.append(summary)

    def _add_text(self, sec: Section, rec: Dict[str, Any]) -> None:
        name = rec.get("print") or ""
        calls = sec.prints.setdefault(name, [])
        call = rec.get("call")
        comm = rec.get("comm")
        block = next((b for b in calls if b.call == call and b.comm == comm), None)
        if block is None:
            block = AllPrint(rec.get("print"), call, comm)
            calls.append(block)
        rank = rec.get("rank")
        text = block.ranks.setdefault(rank if rank is not None else -1, RankText())
        part = rec.get("part")
        if part is not None:
            if rec.get("encoding") == "base64":
                payload = base64.b64decode(rec.get("data", ""))
            elif "text" in rec:
                payload = str(rec.get("text", "")).encode("utf-8", "surrogateescape")
            else:
                payload = "\n".join(rec.get("lines", [])).encode("utf-8", "surrogateescape")
                if rec.get("lines") and not rec.get("last"):
                    payload += b"\n"
            text._parts[int(part)] = payload
            text.trailing_newline = rec.get("trailing_newline", True)
            text._last_seen = bool(rec.get("last")) or text._last_seen
            if rec.get("last"):
                text._last_part = int(part)
            raw = b"".join(text._parts[k] for k in sorted(text._parts))
            decoded = raw.decode("utf-8", errors="surrogateescape")
            text.lines = decoded.split("\n")
            if decoded.endswith("\n"):
                text.lines.pop()
            return
        if rec.get("encoding") == "base64":
            decoded = base64.b64decode(rec.get("data", "")).decode("utf-8", errors="surrogateescape")
            lines = decoded.split("\n")
            if decoded.endswith("\n"):
                lines.pop()
            text.lines.extend(lines)
        else:
            text.lines.extend(rec.get("lines", []))
        text.trailing_newline = rec.get("trailing_newline", True)


def _section_label(sec: Section) -> str:
    label = sec.name or "(outside sections)"
    if sec.title:
        label += f"   {sec.title}"
    if sec.started and not sec.complete:
        label += "   [never closed]"
    return label


def _dataframe(rows: List[Dict[str, Any]]):
    try:
        import pandas
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("pandas is required for *_df(); use *_table() for plain dicts") from exc
    return pandas.DataFrame(rows)


# ------------------------------------------------------------------- front ends

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _jsonl_records(lines: Iterable[str], run: Run, label: str) -> Iterator[Dict[str, Any]]:
    for number, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            run._warn(f"{label}:{number}: not valid JSON, line ignored")
            continue
        if isinstance(rec, dict):
            yield rec


def _inline_lines(text: str) -> List[str]:
    out = []
    for line in text.splitlines():
        line = _ANSI.sub("", line)
        if line.startswith("@ccutils "):
            out.append(line[len("@ccutils "):])
    return out


def _human_v2_records(text: str) -> Iterator[Dict[str, Any]]:
    """Read the line-oriented v2 human grammar into the common record model."""
    section = None
    print_name = None
    call = 0
    timer_location = None
    timer_columns = None
    json_schema = None
    seq = 0

    def expand_human_json(template: Any, values: Any) -> Any:
        if isinstance(template, dict):
            if not isinstance(values, list) or len(values) != len(template):
                raise ValueError("JSON object values do not match schema")
            return {key: expand_human_json(shape, value)
                    for (key, shape), value in zip(template.items(), values)}
        if isinstance(template, list):
            if len(template) == 1 and template[0] == "<values>":
                if not isinstance(values, list):
                    raise ValueError("raw JSON array values are not an array")
                return values
            if not template:
                if not isinstance(values, list):
                    raise ValueError("JSON array values are not an array")
                return values
            if len(template) != 1 or not isinstance(values, list):
                raise ValueError("JSON array values do not match schema")
            return [expand_human_json(template[0], value) for value in values]
        if template in ("<string>", "<number>", "<boolean>") or template is None:
            return values
        return template

    def record(kind: str, **fields):
        nonlocal seq
        seq += 1
        return {"v": SCHEMA_VERSION, "type": kind, "section": section,
                "scope": fields.pop("scope", "global"), "rank": fields.pop("rank", None),
                "seq": seq, **fields}

    for raw in text.split("\n"):
        line = _ANSI.sub("", raw).rstrip("\r")
        start = re.match(r"^== (.+?) :: (.*?) ==\s*$", line)
        end = re.match(r"^== end (.+?) ==\s*$", line)
        if start:
            timer_location = timer_columns = None
            section = start.group(1)
            yield record("section_start", title=start.group(2))
            continue
        if end:
            timer_location = timer_columns = None
            section = end.group(1)
            yield record("section_end", ok=True)
            section = end.group(1).rpartition("/")[0] or None
            continue
        named = re.match(r"^-- all-ranks(?:: (.*?))? --\s*$", line)
        if named:
            print_name = named.group(1) or "all-ranks"
            call += 1
            json_schema = None
            continue
        if re.match(r"^-- end all-ranks:", line):
            print_name = None
            json_schema = None
            continue
        schema_line = re.match(r"^schema:\s*(.+?)\s*$", line)
        if schema_line and print_name:
            try:
                json_schema = json.loads(schema_line.group(1))
            except ValueError:
                json_schema = None
            continue
        timer = re.match(r"^timers \[([^]]+)\]  \((.*?)\)\s*$", line)
        if timer:
            timer_location = timer.group(2)
            timer_columns = None
            continue
        if timer_location is not None:
            cells = line.split()
            if not cells:
                # Timer rows are separated by blank lines; v2 prints the table
                # header only once for consecutive reports at the same location.
                continue
            if cells[0] == "name":
                timer_columns = cells
                continue
            if timer_columns and len(cells) >= 7 and cells[1].isdigit():
                try:
                    values = dict(zip(timer_columns, cells))
                    timer_name, n = values["name"], int(values["n"])
                    vals = [float(values[key]) for key in ("avg", "sd", "min", "max", "sum")]
                except (KeyError, ValueError):
                    timer_location = None
                    timer_columns = None
                    continue
                comm_match = re.match(r"comm (.*), (\d+) ranks", timer_location)
                rank_match = re.match(r"rank (\d+)", timer_location)
                comm = ({"name": comm_match.group(1), "size": int(comm_match.group(2)),
                         "root": None} if comm_match else None)
                yield record("timer", scope="comm" if comm_match else "rank",
                             rank=int(rank_match.group(1)) if rank_match else None,
                             name=timer_name, group="MPI Timer" if comm_match else "Timer",
                             unit="ms", clock="mpi_wtime" if comm_match else "steady",
                             n=n, avg=vals[0], sd=vals[1], min=vals[2], max=vals[3],
                             sum=vals[4], comm=comm)
                continue
            timer_location = timer_columns = None
        rank_line = re.match(r"^\[r(\d+)\] (.*)$", line)
        if rank_line:
            rank = int(rank_line.group(1))
            if json_schema is not None:
                try:
                    compact = json.loads(rank_line.group(2))
                    value = expand_human_json(json_schema, compact)
                except (TypeError, ValueError):
                    value = None
                if value is not None:
                    yield record("json", scope="rank", rank=rank, print=print_name,
                                 call=call, value=value, trailing_newline=True)
                    continue
            yield record("text", scope="rank", rank=rank, print=print_name,
                         call=call if print_name else None, part=0, last=True,
                         lines=[rank_line.group(2)], trailing_newline=True)
            continue
        match = re.match(r"^([^=]+?)\s*=\s*(.+?)\s*$", line)
        if match:
            try:
                value = json.loads(match.group(2))
            except ValueError:
                continue
            yield record("data", key=match.group(1).strip(), value=value)


def detect_format(text: str) -> str:
    """Identify `text` as jsonl, inline, human-v2, or legacy output."""
    """Detect JSONL, inline, human-v2, or legacy output."""
    seen = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        seen += 1
        if stripped.startswith('{"v":'):
            return "jsonl"
        if _ANSI.sub("", line).startswith("@ccutils "):
            return "inline"
        if re.match(r"^\s*== .+ :: .* ==\s*$", _ANSI.sub("", line)):
            return "human-v2"
        if re.match(r"^\s*(timers \[|-- all-ranks)", _ANSI.sub("", line)):
            return "human-v2"
        if seen > 2000:
            break
    return "legacy"


def _legacy_timer_records(rec, line, path: Optional[str]) -> List[Dict[str, Any]]:
    rank = line.rank if line is not None else None
    kind = rec.record_type
    if kind == "summary":
        group = rec.get("group", "")
        comm = group == "MPI Timer"
        return [{"type": "timer", "section": path, "scope": "comm" if comm else "rank",
                 "rank": None if comm else rank, "name": rec.get("name"), "group": group,
                 "unit": "ms", "clock": "mpi_wtime" if comm else "steady", "n": rec.get("n", 0),
                 "avg": rec.get("avg_ms"), "sd": rec.get("stddev_ms"), "min": rec.get("min_ms"),
                 "max": rec.get("max_ms"), "sum": rec.get("sum_ms")}]
    if kind == "run":
        return [{"type": "timer_sample", "section": path, "scope": "rank", "rank": rank,
                 "name": rec.get("name"), "group": "Timer", "unit": "ms", "index": rec.get("index"),
                 "value": rec.get("elapsed_ms"), "last": False}]
    if kind == "sample":
        return [{"type": "timer_sample", "section": path, "scope": "rank", "rank": rank,
                 "name": rec.get("name"), "group": rec.get("group", ""), "unit": "ms",
                 "index": None, "value": rec.get("elapsed_ms"), "last": True}]
    if kind == "aggregate":
        return [{"type": "timer_agg", "section": path, "scope": "rank", "rank": rank,
                 "name": rec.get("name"), "op": rec.get("operation"), "unit": "ms",
                 "value": rec.get("elapsed_ms")}]
    return []


def _legacy_records(text: str) -> Iterator[Dict[str, Any]]:
    """Convert a legacy human-readable log into schema-v2 records."""
    doc = _legacy.parse(text)
    by_number = {l.no: l for l in doc.lines}
    timers: Dict[str, List[Any]] = {}
    for rec in doc.timer_records:
        line = by_number.get(rec.line)
        timers.setdefault((line.section if line else "") or "", []).append((rec, line))
    calls: Dict[str, int] = {}

    def data_records(path, values, scope, rank):
        if isinstance(values, dict):
            for key, value in values.items():
                yield {"type": "data", "section": path, "scope": scope, "rank": rank,
                       "key": key, "value": value}

    def content(path: Optional[str], json_blocks, prints):
        for _, block in json_blocks.items():
            yield from data_records(path, block, "global", None)
        for name, printed in prints.items():
            base = name.split("#")[0]
            if base in _RANK_JSON_BLOCKS:             # rank-local JSON: data, not text
                for rank, out in sorted(printed.ranks.items()):
                    for value in out.json():
                        yield from data_records(path, value, "rank", rank)
                continue
            calls[base] = calls.get(base, 0) + 1      # one call per block, not per rank
            for rank, out in sorted(printed.ranks.items()):
                yield {"type": "text", "section": path, "scope": "rank", "rank": rank,
                       "print": base, "call": calls[base],
                       "lines": [l.text for l in out.lines], "trailing_newline": True}
        for rec, line in timers.pop(path or "", []):
            yield from _legacy_timer_records(rec, line, path)

    def visit(sec):
        yield {"type": "section_start", "section": sec.path, "title": sec.title, "scope": "global"}
        yield from content(sec.path, sec.json_blocks, sec.mpi_all_prints)
        for child in sec.children:
            yield from visit(child)
        yield {"type": "section_end", "section": sec.path, "ok": sec.complete, "scope": "global"}

    yield from content(None, doc.json_blocks, doc.mpi_all_prints)
    for sec in doc.sections.values():
        yield from visit(sec)
    for leftovers in list(timers.values()):          # timers of sections the parser did not keep
        for rec, line in leftovers:
            yield from _legacy_timer_records(rec, line, None)
    for message in doc.warnings:
        yield {"type": "__warning__", "message": message}


# ------------------------------------------------------------------------ API

def loads(text: str, format: Optional[str] = None, name: str = "<string>") -> Run:
    """Parse output text into a Run, detecting its format unless `format` is set.

    Supported formats: `jsonl`, `inline`, `human-v2`, and `legacy`.
    `name` labels the source in the returned model. Raises FormatError if unsupported.
    """
    fmt = format or detect_format(text)
    if fmt not in ("jsonl", "inline", "human-v2", "legacy"):
        raise FormatError(f"unknown format {fmt!r}")
    run = Run(fmt, [name] if name else [])
    if fmt == "human-v2":
        structured = _inline_lines(text)
        if structured:
            for rec in _jsonl_records(structured, run, name):
                run.add_record(rec, name)
        else:
            run.lossy.append("human-v2 log: timer values are limited to displayed precision")
            for rec in _human_v2_records(text):
                run.add_record(rec, name)
    elif fmt == "legacy":
        run.lossy.append("legacy log: timer values have 6 decimals, and CPU timers printed "
                         "outside an all-print block carry no rank")
        for rec in _legacy_records(text):
            if rec.get("type") == "__warning__":
                run._warn(rec["message"])
            else:
                run.add_record(rec)
    else:
        lines = text.splitlines() if fmt == "jsonl" else _inline_lines(text)
        for rec in _jsonl_records(lines, run, name):
            run.add_record(rec, name)
    for sec in [run.outside, *run.sections()]:
        for calls in sec.prints.values():
            for block in calls:
                for rank, ranked in block.ranks.items():
                    if ranked._parts:
                        if ranked._last_part is not None:
                            missing = sorted(set(range(ranked._last_part + 1)) - set(ranked._parts))
                            if missing:
                                run._warn(f"text print {block.name!r} call {block.call} rank {rank}: missing parts {missing}")
                        if not ranked._last_seen:
                            run._warn(f"text print {block.name!r} call {block.call} rank {rank}: missing last part (truncated log?)")
    return run


def load(path: Union[str, Path], format: Optional[str] = None, sidecars: bool = True) -> Run:
    """Parse a log file into a Run, merging MPI `<path>.r<N>` sidecars by default.

    Set `sidecars=False` to read only `path`; `format` overrides auto-detection.
    """
    path = Path(path)
    text = path.read_text(errors="replace")
    run = loads(text, format, str(path))
    embedded_human_records = run.format == "human-v2" and bool(_inline_lines(text))
    if sidecars and (run.format in ("jsonl", "inline") or embedded_human_records):
        found = sorted(glob.glob(glob.escape(str(path)) + ".r[0-9]*"),
                       key=lambda p: int(re.search(r"\.r(\d+)$", p).group(1)))
        for side in found:
            if not re.search(r"\.r\d+$", side):
                continue
            run.sources.append(side)
            side_text = Path(side).read_text(errors="replace")
            if run.format == "jsonl":
                lines = side_text.splitlines()
            else:
                lines = _inline_lines(side_text)
            for rec in _jsonl_records(lines, run, side):
                run.add_record(rec, side)
    return run
