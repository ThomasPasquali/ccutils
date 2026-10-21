# Migrating to CCUTILS 1.0 output

CCUTILS 1.0 selects the v2 human renderer by default. The Python `ccutils` parser reads
v2 human logs, legacy human logs, inline streams and JSON Lines machine streams into the
same `Run` model. Existing C++ macros keep their names and signatures.

## Keep legacy output temporarily

Select the old human format at runtime:

```sh
CCUTILS_FORMAT=legacy ./app
```

To make legacy the default for a build, define `CCUTILS_FORMAT_LEGACY` for the consuming
target. A runtime `CCUTILS_FORMAT` value or `ccutils::Config::format` setting can still
select either supported format. The legacy renderer preserves the previous output layout
for scripts that depend on it. Migrate those scripts to the v2 layout or parse the log with
`ccutils.load()` before removing this compatibility setting.

## Human output changes

The v2 renderer uses full section paths and explicit end markers, key/value lines, timer
tables, and `[rN]` prefixes for gathered MPI text. Rank text and section structure are
unambiguous in v2. Colors default to `auto`, so ANSI sequences are emitted only to terminal
sinks; `NO_COLOR` disables them. Set `CCUTILS_COLOR=always` or `never` to override this
behavior, or use `CCUTILS_COLOR_LEVEL` to choose the compile-time palette.

See [the human and machine format reference](FORMAT.md) for the record schema and output
examples. For file destinations and rank sidecars, see [output configuration](OUTPUT.md).

## Python parser changes

New code should use the `ccutils` package:

```python
import ccutils

run = ccutils.load("run.jsonl")
print(run.tree())
print(run["solve"].data)
```

`load()` detects JSONL, inline, human-v2 and legacy logs. `run.lossy` identifies information
that a human log cannot retain; use JSONL when you need full precision, sequence checks, or
exact rank text. Pandas remains optional and is needed only for the DataFrame helpers.

The older `ccutils.ccutils_parser` functions and classes remain available for compatibility.
New code should use `load`, `Run`, and `Section`; the old top-level `parser` package is a
deprecated compatibility alias and may conflict with Python's standard-library module.

## MPI timer naming

Use `ccutils::MpiTimer` for the MPI timer class. The global `TimerStats` name remains as a
deprecated alias for source compatibility. `ccutils_timers::TimerStats` is a separate type
for CPU timer statistics.
