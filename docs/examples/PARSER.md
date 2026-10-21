# Parsing CCUTILS output

Use the `ccutils` package for new parser code. It detects machine JSONL, inline records,
human-v2 logs, and legacy logs and returns the same `Run` model. Machine output preserves
more detail than human output; enable it with `CCUTILS_MACHINE_OUT=file:run.jsonl` when
exact timer values, sequence checks, or lossless rank text matter.

Install the Python package from the checkout with `python3 -m pip install .`, or add the
repository root to `PYTHONPATH` while developing:

```python
import ccutils

run = ccutils.load("run.jsonl")
print(run.tree())

solve = run["solve"]
print(solve.data["residual"])
print(solve.timers["kernel"].avg_ms)
print(run.timers_table())
```

`Run` supports path lookup, repeated sections with `run.all(path)`, name search with
`run.find(name)`, rank-local values through `Section.per_rank(key)`, and MPI text through
`Section.prints`. `run.lossy` identifies details that cannot be recovered from human logs.
Pandas is optional; install `ccutils[pandas]` to use `timers_df()` and other DataFrame
helpers.

The command line can inspect, export, and validate logs:

```sh
ccutils tree run.jsonl
ccutils timers run.jsonl --csv > timers.csv
ccutils data run.jsonl --section solve --key residual
ccutils check run.jsonl
```

The older `Document`, `MPIOutputParser`, and `parse_*` functions remain available from
`ccutils.ccutils_parser` for existing applications. The top-level `parser` package is a
deprecated compatibility alias and can conflict with Python's standard library on older
Python versions. See the full [parser guide](../PARSER_V2.md).
