# Reading ccutils output from Python

```python
import ccutils

run = ccutils.load("run.jsonl")     # machine stream, "@ccutils " inline lines, or a human log
run.tree()                          # what is in here
run["solve/spmv"].timers["kernel"].avg_ms
run["solve"].data["residual"]
run.timers_table()                  # list of dicts; run.timers_df() gives a DataFrame (pandas)
```

The format is detected automatically, and **the same queries work on every format**: the
machine stream (`CCUTILS_MACHINE_OUT=file:...`), the inline stream (`=inline`, human log and
records in one file) and a plain human-readable log. A human log is parsed more slowly and
with less precision (see below), but you get the same `Run` object.

## The model

```
Run                              run.format, run.sources, run.warnings, run.lossy
 ├─ run["a/b"], run.all("a/b"), run.find("b"), run.sections()
 └─ Section                      .name .path .title .complete .children .parent
     ├─ .data                    global values, {key: value}
     ├─ .rank_data / .per_rank(key)   rank-local values
     ├─ .timers                  {name: Timer}
     └─ .prints                  all-ranks prints: {name: [AllPrint]}, .calls(name)
                                  AllPrint.grep()/diff_ranks(); .ranks[r].lines/.text/.json()

Timer                            .last .n .avg .sd .min .max .sum (and *_ms) of the latest report
 ├─ .summaries                   every report of that name (a timer may be printed several times)
 ├─ .comm                        the report pooled over a communicator (MPI timers), with .comm["size"], ["root"]
 ├─ .by_rank()                   per-process reports {rank: TimerSummary}
 ├─ .per_rank(kind)               pandas Series for rank reports (kind: n, avg, sd, min, max, sum)
 └─ .samples / .aggregates       CCUTILS_TIMER_PRINT_ALL / _LAST and the SUM macros
```

Real example (2 MPI ranks, a worker file adding to a section created in `main`):

```
tests/fixtures/multifile_mpi.inline.txt  (inline, 6 records, 0 warnings)
run   Whole run
├─ data: iterations
├─ timers: halo
└─ work   Work
```

* `run["a/b"]` is the first section with that path; a section re-opened in a loop appears
  several times: use `run.all("a/b")`. A wrong path raises `KeyError` with suggestions.
* `timers_table()` includes pooled and per-rank MPI statistics, imbalance, slowest rank,
  reporting count, and errors. `timers_df(level="comm"|"rank")` filters by scope;
  with no level, it returns all reports. `data_table()` rows: `section, scope, rank, key, value`.

## MPI: files from several ranks

Records written by a rank other than 0 go to a sidecar file `<file>.r<N>` (see OUTPUT.md).
`load("run.jsonl")` merges `run.jsonl.r*` automatically (`sidecars=False` to skip them);
`run.sources` lists the files. Records that belong to no section end up in `run.outside`.

## Checking a log

`run.check()` returns problems found while reading: sections that were never closed (job
killed), lines that were lost (gaps in `seq`), invalid JSON lines, records of a newer schema.
Unknown record types and fields are ignored, so newer logs still load.

## Python functions

```python
ccutils.detect_format(text)                       # identify jsonl, inline, human-v2, or legacy
ccutils.loads(text, format=None, name="<string>") # parse text; format is optional
ccutils.load(path, format=None, sidecars=True)     # parse a file; merge MPI .r<N> files by default
```

`load` and `loads` return a `Run`; unsupported input raises `ccutils.FormatError`.
`Run` provides:

| Function | Result |
| --- | --- |
| `run[path]`, `path in run` | First section at a slash-separated path; membership check |
| `run.all(path)`, `run.find(name)` | Repeated sections at a path; sections matching a name |
| `run.sections()` | Iterate through all sections |
| `run.tree(file=None)` | Return and print a tree; pass `file=False` to return without printing |
| `run.check()` | List structural warnings |
| `run.data_table()`, `run.timers_table()` | Lists of dictionaries for global/rank data and timer reports |
| `run.data_df()`, `run.timers_df(level=None)` | Optional pandas DataFrames; timer `level` may be `"rank"` or `"comm"` |

Each `Section` has `.data`, `.rank_data`, `.timers`, and `.prints` dictionaries.
Use `section.per_rank(key)` to collect a rank-local value. A `Timer` exposes its
latest summary (`.last`, `.avg_ms`, `.min_ms`, and related fields), all `.summaries`,
`.samples`, `.aggregates`, `.by_rank()`, and `.per_rank(kind)` (requires pandas).
For MPI print calls, use `section.prints.calls(name)`. `call.jsons` directly maps each
rank to its first parsed JSON value; `call.json_values` maps ranks to lists when a rank
prints multiple JSON values. Each `call.ranks[rank]` also provides `.text`, `.lines`, and
`.json()`. `AllPrint.grep(pattern)` searches rank lines; `diff_ranks(left, right)`
compares two ranks.

```python
jsons = run["graph_structure"].prints.calls("ccutils_rank_json")[0].jsons
print(jsons[0])  # parsed JSON from rank 0
```

The compatibility API is available from `ccutils.ccutils_parser`: `parse(text, **options)`
and `parse_file(path, **options)` return the older `Document`; `parse_timer_records(text)`
reads standalone timer lines. `describe_structure(document)` returns metadata, and
`pretty_print_structure(document, verbosity=1)` prints it. New code should use `load` or
`loads` above.

## Human logs: what is different

`run.lossy` says what a human log cannot give you. For the legacy format: timer values have
6 decimals; CPU timers printed outside an all-print block carry no rank; MPI timer reports
carry no communicator size or root. The machine stream has none of these limits.
Legacy output has no escaping for rank text: a line that exactly matches a structural marker
can be interpreted as that marker. Use JSONL or inline output when rank text may contain
marker-like lines.

## Command line

```bash
ccutils tree   run.jsonl
ccutils timers run.jsonl --csv > timers.csv
ccutils data   run.jsonl --section solve --key residual
ccutils check  run.jsonl        # exit status 1 if there are problems
```

The older functions (`parse`, `parse_file`, `MPIOutputParser`, ...) remain importable from
`ccutils.ccutils_parser` as compatibility APIs. The old top-level `parser` package and
`python -m parser` entry point are retained as deprecated aliases where that name is
available; Python versions through 3.9 include a standard-library module named `parser`.
