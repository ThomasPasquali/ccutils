# ccutils design

This is the design the implementation follows. Open work is in [TODO.md](TODO.md); the
sections below are the original plan, unchanged except where the status section
says otherwise.

## Implementation status and deviations from the plan

What exists (all with tests, see `tests/README.md`):

| Design section | Implemented in |
|----------------|----------------|
| §2 sinks, env vars, §2.5 writer rules / sidecars | `include/ccutils/emitter.hpp`, `docs/OUTPUT.md` |
| §3.1 machine stream (partly) | `include/ccutils/machine.hpp`, `docs/FORMAT.md` |
| §4.5 MPI timers (v2 statistics, rank details and raw samples) | `include/ccutils/mpi/mpi_timers.hpp` |
| §6 parser (JSONL / inline / human-v2 / legacy front ends, model, CLI) | `ccutils/ccutils_run.py`, `docs/PARSER_V2.md` |
| §8 tests: golden harness, launcher self-check, scenarios, parser tests | `tests/` |

Deviations and additions (decisions made while implementing):

1. **Sections, timers and JSON objects are looked up by name** (new requirement:
   "create a section/timer in `main` and update it in another file's function"). A
   process-wide registry (`registry.hpp`, `sections.hpp`) replaces the per-scope variables
   the macros used; see `docs/MULTIFILE.md`. `ccutils::Section` provides RAII scope
   management; `ccutils::section("name")` returns a `SectionRef` handle.
2. **The legacy macros keep their trailing `;`.** §1 proposed removing it; the docs call the
   macros without a semicolon, so that would break users. Multi-statement macros were made
   single statements instead.
3. **The emitter has no MPI dependency** (ODR, TODO.md ground rule 4). The rank reaches it
   through `set_world_rank()` (called by the MPI macros and `MpiTimer`), else from launcher
   environment variables, else 0. `ccutils::mpi_rank()` / `mpi_size()` replace the
   `ccutils_inmacro_*` locals inside the library macros, so `CCUTILS_MPI_INIT` is optional.
4. **`data` records are written when a section ends**, because that is when the legacy code
   prints a section's JSON (§3.1 does not say when).
5. **CPU-timer values and statistics use `double`**; machine records serialize them with
   `real()` and statistics use compensated summation.
6. **Legacy MPI timer records carry pooled statistics** plus `comm` {name, size, root};
   v2 adds per-rank distributions and records, imbalance, slowest rank, reporting and misuse
   counts, warm-up exclusion and opt-in raw samples.
7. `timer_xrank` (early draft) does not exist; its content is the `per_rank` part of the
   comm-scope `timer` record.
8. **Existing all-print forms still use `MPI_COMM_WORLD`**; explicit `_COMM` variants accept another communicator.
9. New Python code imports `ccutils`; `parser` is retained as a deprecated shim. The wheel
   includes both packages and the `ccutils` CLI entry point.
10. `CCUTILS_COLOR_LEVEL` already existed (compile-time); the runtime `CCUTILS_COLOR` of §2.2
    is P1-color.
11. `CCUTILS_MPI_GLOBAL_JSON_PUT` / `_LOCAL_JSON_PUT` are now no-ops under `CCUTILS_NO_JSON`
    (they did not exist there).
12. Bugs found in the existing code that the plan did not list, all fixed with a test:
    `CCUTILS_MPI_SECTION_DEF/END` did not compile with JSON enabled; the stale
    `examples/mpi/mpi_example.cpp`.

---

# ccutils improvement plan


Scope: timers, sections, JSON, MPI all-print, and the Python parser.
Files reviewed: `macros.hpp`, `timers.hpp`, `formats.h`, `mpi/mpi_macros.hpp`, `parser/ccutils_parser.py`.
Also reviewed: `mpi/mpi_timers.hpp` (the MPI timer class, called `TimerStats` in the global namespace).

---

## 0. Goals and hard constraints

### Goals
1. **Readable output** for humans (aligned tables, clear nesting, per-rank layout).
2. **Easy-to-parse output** for tools (one schema, one record per line, explicit context).
3. **Configurable destinations**: stdout by default, optionally a file, optionally two separate streams (human + machine).
4. **A parser with one obvious way to get data** (path lookup, DataFrames, a `tree()` that shows what exists).
5. **The parser still works on human logs** (slower and possibly less precise, but same data model).

### Hard constraints (must not regress)
| # | Constraint | Why it matters |
|---|------------|----------------|
| C1 | `CCUTILS_MPI_ALL_PRINT*` gathers rank-local H/M frames to the communicator root, which is the only sink writer, then synchronizes with `MPI_Barrier`. | No shared FS, no interleaving, only standard collectives; large frame sets use bounded Gatherv rounds. |
| C2 | Existing macro names and signatures keep working, including the `fp` variable inside `PRINTS_CODE_BLOCK`. | User code must compile unchanged. |
| C3 | Old output stays **byte-identical** under `CCUTILS_FORMAT=legacy` (the default until the new format is stable). | Old logs, scripts and the old parser keep working. |
| C4 | No new collective operations inside existing macros unless opt-in. | Adding a hidden collective can deadlock a program that calls a macro from a subset of ranks. |
| C5 | `CCUTILS_NO_JSON` and non-MPI builds keep compiling. | Library is header-only and optional-dependency based. |
| C6 | Flush/ordering behaviour (`fflush`, `CCUTILS_FLUSH_WAIT`) is preserved by default. | These are probably there for launcher stdout ordering; do not remove blindly. |
| C8 | `TimerStats::print_stats(label, comm, root)` stays a **collective over `comm`**, called by all its members, with the same meaning of `root`; it must keep working on sub-communicators and with `root != 0`. In `legacy` format the **old reduction code and formulas are kept verbatim**. | Existing programs call it that way, and legacy numbers must not change. |
| C7 | **Per-rank text may be multi-line** (blank lines, leading whitespace, no trailing newline, text that looks like a marker). It is treated as an opaque byte string: never trimmed, re-wrapped or interpreted, and it must be recoverable exactly from both streams. | The all-print block exists precisely to carry arbitrary multi-line rank output. See 3.4. |

---

## 1. What is wrong today (concrete findings)

### Output
* **Five marker dialects** in one stream: `=+=+=+= name :: title =+=+=+=`, `=-=-= name =-=-=`, `-+-+-+- name -+-+-+-`, `[[Rank N]]`, and `CCUTILS_TIMER v=1 key=value`. The parser needs one regex per dialect.
* **JSON blocks have generic names** (`ccutils_json`, `ccutils_global_json`, `ccutils_rank_json`, `ccutils_subsection_json`), so the name carries no information. The parser dedupes them as `name#2`, `name#3`.
* **Timers are not attached to sections.** They are free lines (`group=Timer name=...`); the parser keeps them in a flat `doc.timer_records`.
* **Context is positional.** To know which section/rank a line belongs to, the parser replays a begin/end state machine over the whole file (including bracket matching for pretty-printed JSON).
* **Precision:** CPU timer samples and statistics use `double`; statistics use compensated summation.
* Colors are always emitted, even when stdout is a file or pipe.

### Parser
Data lives in five places: `section.json_blocks`, `section.mpi_all_prints[n].ranks[r].json()`, `doc.timer_records`, `Line.data`, and `json_data` ("first block, whatever it is"). There is no single way to answer "give me `residual` for section `solve`".

Multi-line text specifically:
* An all-print block's **name is not part of the call identity**: calling `CCUTILS_MPI_ALL_PRINT_NAMED(x, ...)` in a loop produces blocks the parser can only disambiguate as `x`, `x#2`, `x#3` by file order.
* Rank text is delimited only by `[[Rank N]]` / `[[END Rank N]]`. A rank whose text contains a line identical to a marker (or a section banner) corrupts the parse; legacy has no escaping.
* The parser splits input with `str.splitlines()`, which also splits on `\r`, `\f`, `\v`, `\x1c-\x1e`, `\x85`, `\u2028`, `\u2029`. Rank text containing any of them is silently re-split. Splitting must be on `\n` only.
* Trailing-newline information is lost (the macro appends `\n` if missing), so the exact text cannot be reconstructed.

### Timers (`timers.hpp` in MPI programs and `mpi/mpi_timers.hpp`)

How the MPI timer works today: each rank records local `MPI_Wtime` intervals; `print_stats(label, comm, root)` does **five** `MPI_Reduce` calls (count, min, max, sum, sum of squares) to `root`, and the root prints one `CCUTILS_TIMER ... group="MPI Timer"` line with `printf`.

* **All samples of all ranks are pooled into one distribution** (`n`, `avg`, `sd`, `min`, `max`, `sum`). That answers "how much total work", but not the question that matters most for MPI phases: *which rank is slowest and by how much*. The time that bounds a phase is the **maximum over ranks of the per-rank total**, and the pooled `sd` mixes within-rank noise with between-rank imbalance.
* **Five collectives per timer.** With 20 timers that is 100 reductions at every report. One gather of a small fixed-size struct per rank carries the same information (and more).
* **Numerics:** the variance is computed from `sum` and `sum of squares`, which cancels catastrophically when `sd << avg` (the code clamps negative variance to 0, which is a symptom). Use a centered two-pass/Chan merge.
* **Output bypasses everything else:** the root writes straight to `stdout` (so `root != 0` prints from that rank), the record has no section, no communicator info and only the string `"MPI Timer"` to tell it apart from a CPU timer.
* **`std::logic_error` on misuse** (`start` while running, `stop` while stopped) is thrown on one rank only; if that rank then skips the collective `print_stats`, the others hang.
* **CPU timers in an MPI program** (`timers.hpp`): every rank prints its own `CCUTILS_TIMER` line directly, with **no rank field**. Lines from different ranks interleave and the parser cannot attribute them unless the code was wrapped in an all-print block.
* **Two clocks:** CPU timers use `steady_clock` with `double` milliseconds; MPI timers use `MPI_Wtime` with `double`. Nothing records the clock source or `MPI_Wtick`.
* **Feature gaps vs. CPU timers:** no exclude-first-N (warm-up), no per-sample output, no sum/aggregate helpers.
* **Naming/hygiene:** the global class `TimerStats` has the same name as the unrelated `ccutils_timers::TimerStats` struct (ambiguous after `using namespace ccutils_timers`); `CCUTILS_MPI_TIMER_START/STOP` end with a stray `;`; `CCUTILS_MPI_TIMER_CLOSE` expands to several statements without `do { } while (0)` (breaks in an unbraced `if`); the include guard `__CCUTILS_MPI_TIMERS__` is a reserved identifier; the header includes `<ccutils/...>` while the other headers use relative paths; `<iostream>` is unused.

### Small bugs/risks found while reading (fix in Phase 0)
* `CCUTILS_MPI_COMMUNICATOR_PRINT`: `char s[50]` with `sprintf("temp_%s_%d_%d.txt")` can overflow (communicator names go up to `MPI_MAX_OBJECT_NAME`), and it shells out with `system("cat ...")`/`rm`.
* Identifiers with a double underscore or leading `__` + uppercase (`__TIMER_PTR`, `__CCUTILS_FORMATS__`, `__timer_vals_*`, `__section_json_*`) are reserved by the C++ standard. Rename (keep old names as aliases if needed).
* `CCUTILS_MPI_STATUS_CHECK` is followed by the `#endif` of the include guard, so `CCUTILS_MPI_CLEANUP` is outside the guard.
* Several macros end with `;` inside the definition (`... while (0);`), which gives `;;` and breaks `if/else` without braces.
* `CCUTILS_MPI_SECTION_DEF` allocates with `new` on rank 0; an early `return`/exception between DEF and END leaks it and skips the closing collective.
* `CCUTILS_FLUSH_WAIT(200000)` sleeps 200 ms after **every** `PRINT_ONCE`. Keep the default (C6) but make it configurable.
* `CCUTILS_MPI_BUFFERED_PRINT_*` depends on a writable, rank-visible working directory. Not part of all-print; document the requirement.

---

## 2. Output architecture: two logical streams, configurable sinks

### 2.1 Concepts
* **Human stream**: pretty, aligned, optionally colored. Meant to be read.
* **Machine stream**: JSON Lines (JSONL), one record per line. Meant to be parsed.
* **Sink**: where a stream goes: `stdout`, `stderr`, a file, or nothing.
* Each stream has **its own sink**. Writing both to stdout is allowed (see `inline` below).

### 2.2 Configuration (environment variables, with C++ equivalents)

| Variable | Values | Default |
|----------|--------|---------|
| `CCUTILS_FORMAT` | `legacy` \| `v2` | `legacy` (switches to `v2` in a later major release) |
| `CCUTILS_HUMAN_OUT` | `stdout` \| `stderr` \| `none` \| `file:<path>` \| `append:<path>` | `stdout` |
| `CCUTILS_MACHINE_OUT` | same as above, or `inline` | `none` |
| `CCUTILS_COLOR` | `auto` \| `always` \| `never` (`NO_COLOR` is honoured) | `auto` (color only if the sink is a TTY) |
| `CCUTILS_HUMAN_PRECISION` | digits for timer tables | `3` |
| `CCUTILS_FLUSH_WAIT_US` | microseconds slept after `PRINT_ONCE` | `200000` (unchanged) |
| `CCUTILS_TIMER_RANKS` | detail of MPI timer reports: `summary` (one comm-scope record with per-rank min/avg/max), `all` (plus one record per rank), `top:K` (plus the K slowest ranks), `none` | `summary` |
| `CCUTILS_TIMER_MISUSE` | `throw` (today's `std::logic_error`) \| `record` (count it, report it, never throw) | `throw` |

Path templates: `%j` = `SLURM_JOB_ID` (or pid if unset), `%p` = pid, `%r` = rank.
* A path **without** `%r` is opened **only by rank 0**, which is the only writer (consistent with C1).
* A path **with** `%r` makes every rank write its own file (no gather). This is opt-in, useful for huge outputs, and documented as "non-collective".

C++ equivalent (takes precedence over env if called before first output):

```cpp
ccutils::Config cfg;
cfg.format      = ccutils::Format::V2;
cfg.human_out   = ccutils::Sink::stdout_();
cfg.machine_out = ccutils::Sink::file("run-%j.jsonl");
ccutils::init(cfg);        // optional; lazy init from env otherwise
```

### 2.3 Typical setups

```bash
# 1) Default: everything human-readable on stdout (what you have today, just prettier in v2)
mpirun -n 4 ./app

# 2) Human on stdout, machine-readable in a file  (the "two separate streams" case)
CCUTILS_FORMAT=v2 CCUTILS_MACHINE_OUT=file:run-%j.jsonl mpirun -n 4 ./app

# 3) Only machine output, silent terminal
CCUTILS_FORMAT=v2 CCUTILS_HUMAN_OUT=none CCUTILS_MACHINE_OUT=stdout ./app > run.jsonl

# 4) Human log to a file, nothing on the terminal; parse the human log later
CCUTILS_FORMAT=v2 CCUTILS_HUMAN_OUT=file:run.log ./app
python -m ccutils show run.log           # parser autodetects the human format

# 5) One stream, both audiences ("inline"): machine records are prefixed lines
CCUTILS_FORMAT=v2 CCUTILS_MACHINE_OUT=inline ./app > run.out
grep -v '^@ccutils ' run.out             # human view
grep    '^@ccutils ' run.out | cut -c10- # machine view (JSONL)
```

`inline` is the safest option for batch systems where only the job's stdout is kept (Slurm `--output`): one file, losslessly parseable, and still readable.

### 2.4 Internal design (C++)

One process-wide `Emitter` owns the sinks. Everything else goes through it.

```cpp
namespace ccutils {
struct Sink { enum Kind { None, Stdout, Stderr, File, Append, Inline } kind; std::string path; /*...*/ };

class Emitter {
public:
  static Emitter& get();                 // lazy singleton, reads env on first use
  void configure(const Config&);
  void human  (std::string_view text);   // only called on the writer rank
  void machine(std::string_view line);   // same
  bool human_enabled()   const;
  bool machine_enabled() const;
};
}
```

Rules:
* In an MPI program the Emitter only writes on rank 0 (or, with `%r`, on every rank to its own file). The rank is obtained with `MPI_Initialized` + `MPI_Comm_rank`, never assumed.
* `ccutils::init()` is optional and non-collective. Files are opened lazily on first write, by the writer rank only.
* Everything flushes at the same points as today (C6).

### 2.5 Which rank may write (collective vs. non-collective emits)

The "rank 0 is the only writer" rule (C1) holds for everything that is **gathered**. Some output is produced by a different rank by design (an MPI timer report on a communicator root, a CPU timer printed outside a section), so the rule needs to say what happens there. The key point is that **no emit may require point-to-point communication or an extra collective**; those are the things that deadlock.

| Emit kind | Writer | `stdout` / `stderr` sink | file sink |
|-----------|--------|--------------------------|-----------|
| Gathered (`ALL_PRINT`, `Section::end()`, batched timers) | world rank 0 | as today | the single file, opened by rank 0 only |
| `*_ONCE` prints, global data | world rank 0 | as today | the single file |
| Non-collective, per-process emit (CPU timer printed outside a section, misuse/error records) | the calling rank | one complete line per call, written with a single `fwrite` + `fflush` (atomic for lines up to `PIPE_BUF` on pipes/files, so lines never tear) | **sidecar file** `<path>.r<world_rank>` |
| MPI timer report on a communicator root | the comm root (as today) | direct, as today | main file if the root is world rank 0, otherwise sidecar `<path>.r<world_rank>` |

Why a sidecar: several processes appending to one shared file is not safe on every filesystem (NFS, Lustre), and a rank that is not part of a gather cannot hand its text to world rank 0 without communication. A sidecar costs nothing at runtime. The Python `load("run.jsonl")` auto-merges `run.jsonl.r*` (`sidecars=False` to disable) and lists the files in `run.sources`; records carry `seq` and `section` so order within a writer is preserved.

Because every v2 record is exactly one physical line (3.1), interleaving of non-collective emits on a shared `stdout` can reorder lines but never corrupt a record. This is a major reason to keep records single-line.

---

## 3. Formats

### 3.1 Machine format v2 (JSONL)

One JSON object per line. Common envelope:

| Field | Meaning |
|-------|---------|
| `v` | schema version (`2`) |
| `type` | record type (below) |
| `section` | full path, e.g. `"solve/spmv"`; `null` outside sections |
| `scope` | `"global"` (one value for the run), `"comm"` (computed over a communicator, e.g. an MPI timer report) or `"rank"` |
| `rank` | integer for `scope:"rank"`, otherwise `null` |
| `seq` | per-writer monotonically increasing counter (ordering aid) |

Record types:

| `type` | Extra fields |
|--------|--------------|
| `run_start` | first record in each process stream; optional: `ccutils_version`, `format`, `mpi` (`library`, `world_size`, `wtime_is_global`, `wtick_s`), `slurm_job_id`, `host`. Lets a parser interpret timers (clock resolution, whether timestamps are comparable across ranks). MPI metadata is supplied by the MPI headers so the core stays MPI-free. |
| `section_start` | `title` |
| `section_end` | `ok` (false if closed by exception/unwind) |
| `data` | `key`, `value` (any JSON) |
| `timer` | `name`, `unit` (`"ms"`), `clock` (`"steady"` or `"mpi_wtime"`), `n`, `avg`, `sd`, `min`, `max`, `sum`, `errors`. **Rank scope** (CPU timer, or one rank of an MPI timer): statistics of that rank's samples, with `rank` set. **Comm scope** (MPI timer report): the same fields pooled over all samples of all ranks, plus `comm` (`{"name","size","root"}`, `root` is a world rank), `ranks_reporting`, `per_rank` (distribution of the per-rank `sum` and `avg`: `min`, `avg`, `max`, `argmin`, `argmax`), `imbalance` (`per_rank.sum.max / per_rank.sum.avg`, 1.0 = perfectly balanced) and `slowest_rank`. See 4.5. |
| `timer_sample` | `name`, `index`, `value` |
| `timer_agg` | `name`, `op` (`sum`/`sum_avg`/`sum_last`), `value` |
| `timer_samples` (opt-in) | `name`, `rank`, `values` (array of ms): raw samples of one rank, chunked with `part`/`last` like `text`. Replaces `print_all_times` for MPI timers. |
| `text` | free-form, possibly multi-line output: `print` (all-print name or `null`), `call` (all-print call counter), `lines` (array of strings), `trailing_newline` (bool), `part`/`last` (chunking), optional `encoding:"base64"` + `data`. See 3.4. |

Example:

```jsonl
{"v":2,"type":"section_start","section":"solve","title":"Solve phase","scope":"global","rank":null,"seq":1}
{"v":2,"type":"data","section":"solve","scope":"global","rank":null,"key":"residual","value":1.0e-9,"seq":2}
{"v":2,"type":"data","section":"solve","scope":"rank","rank":0,"key":"bytes_sent","value":1048576,"seq":3}
{"v":2,"type":"data","section":"solve","scope":"rank","rank":1,"key":"bytes_sent","value":1049600,"seq":4}
{"v":2,"type":"section_start","section":"solve/spmv","title":"Sparse mat-vec","scope":"global","rank":null,"seq":5}
{"v":2,"type":"timer","section":"solve/spmv","scope":"rank","rank":0,"name":"kernel","unit":"ms","n":10,"avg":1.2,"sd":0.1,"min":1.1,"max":1.4,"sum":12.0,"seq":6}
{"v":2,"type":"section_end","section":"solve/spmv","ok":true,"scope":"global","rank":null,"seq":7}
{"v":2,"type":"section_end","section":"solve","ok":true,"scope":"global","rank":null,"seq":8}
```

Key properties: context is **in every record** (no state machine needed), the machine renderer adds no ANSI, every record is exactly one physical line (multi-line text is carried as a JSON array of lines, never as raw newlines), and unknown fields/types are ignored by readers (forward compatible).

### 3.2 Human format v2 ("readable, but parseable by construction")

The human format is designed so that a parser can recover the same model from it. It follows a small **documented grammar**:

```
== solve :: Solve phase ==============================================
  residual           = 1.0e-9
  bytes_sent (rank)  : r0=1048576  r1=1049600  r2=1048832  r3=1050112

  == solve/spmv :: Sparse mat-vec ====================================
    timers [ms]  (rank 0)
      name      n     avg      sd     min     max     sum
      kernel   10   1.200   0.100   1.100   1.400  12.000
      halo     10   0.310   0.020   0.290   0.350   3.100
  == end solve/spmv ==================================================
== end solve ==========================================================
```

Grammar rules (these are what the parser relies on, so they are part of the spec):
1. Structure lines are `== <path> :: <title> ==...` and `== end <path> ==...`. The **full path** is always printed, so indentation is cosmetic only.
2. Scalar data: `key = <json>`. Per-rank scalar data: `key (rank) : r<i>=<json> ...`.
3. Non-scalar JSON: `key = <json>`; may wrap over several lines (width-limited pretty print, as `CCUTILS_JSON_PRETTY_PRINT` already does). The parser reuses the existing bracket-matching logic.
4. Timer tables start with `timers [<unit>]  (rank N | comm <name>, <size> ranks)`, followed by a header row (column names are read from it) and rows until a blank line. A `|` in the header and rows separates the pooled columns from the per-rank columns of an MPI timer (example in 4.5); a `slowest` cell is `r<world_rank>`.
5. Per-rank free text and all-print blocks use a line prefix `[rN] ` on **every** line, and rank-0/section text uses the gutter `| ` on every line (details and escaping rules in 3.4). Inside a text block only prefixed lines are content, so only unprefixed lines can be structure.

**Known trade-off (documented, not hidden):** human tables round numbers (`CCUTILS_HUMAN_PRECISION`, default 3). A timer parsed from a human log has display precision; the parser flags it (`run.lossy`). For exact values use the machine stream or `inline` mode.

### 3.3 Legacy format

Exactly today's output, selected by `CCUTILS_FORMAT=legacy` (default at first). The current regex parser logic is kept as the "legacy front end" (see 6.2). Legacy has no escaping, so a rank text line identical to a legacy marker can still confuse it; this is a documented limitation of the old format, not something v2 inherits.

### 3.4 Multi-line text (all-print blocks, `*_ONCE` prints, section text)

**Principle (C7):** a rank's text is an opaque byte string. ccutils never trims, re-wraps, re-indents or interprets it. Both streams must reproduce it exactly, with one documented normalization: the human stream always ends the last line with `\n`, as the macro does today.

**Rules shared by all v2 formats**
1. **Lines are split on `\n` only.** `\r` and every other Unicode separator are content. Parsers must not use language-level helpers such as Python's `splitlines()`.
2. **Blank lines and whitespace are preserved**: empty lines in the middle or at the end, leading spaces, tabs.
3. **Trailing newline is recorded.** `lines = text.split('\n')`; if the text ends with `\n`, the last empty element is dropped and `trailing_newline` is `true`. Examples: `"a\n\n"` -> `["a",""]`, `true`; `"a"` -> `["a"]`, `false`.
4. **Zero bytes means "no output"** and the rank is omitted (as today). Whitespace-only output is output.
5. **Binary-unsafe content:** if the text is not valid UTF-8 (or contains NUL), the machine record uses `"encoding":"base64"` with a `data` field instead of `lines`. Valid UTF-8, including control characters, is JSON-escaped.
6. **Bounded record size:** long text is split into chunks (default at most 1000 lines or 64 KiB per record) with `part` (0-based) and `last`. The parser concatenates chunks in `part` order. JSONL lines therefore stay small even for huge outputs.
7. **Call identity:** every all-print record carries `print` (the name) and `call` (a per-process counter of all-print invocations). The macro is collective, so the counter agrees on all ranks. Repeated names in a loop are distinguished by `call`, not by file order.

**Machine stream example.** The user block on rank 1 prints:

```
iter 1
  residual: 3.2e-3

iter 2 (converged)
```

which becomes a single physical line:

```jsonl
{"v":2,"type":"text","section":"solve","scope":"rank","rank":1,"print":"convergence","call":4,"lines":["iter 1","  residual: 3.2e-3","","iter 2 (converged)"],"trailing_newline":true,"part":0,"last":true,"seq":21}
```

**Human stream example** (same block, ranks 0 and 1):

```
-- all-ranks: convergence #4 --------------------------------------
[r0] iter 1
[r0]   residual: 1.1e-3
[r0] 
[r0] iter 2 (converged)
[r1] iter 1
[r1]   residual: 3.2e-3
[r1] 
[r1] iter 2 (converged)
-- end all-ranks: convergence #4 ----------------------------------
```

Human-stream rules:
* **Every** line gets the prefix, blank lines included, so a blank line can never be mistaken for the end of a block. The prefix is exactly `[rN] `; parsers also accept `[rN]` with no trailing space (editors strip it).
* Text is never wrapped or truncated; original indentation follows the prefix unchanged.
* **Marker-lookalikes are harmless:** inside a block only prefixed lines are content and only unprefixed lines can be structure. A rank line `== end solve ==` is printed as `[r1] == end solve ==` and is ignored by the structure scanner. A line already starting with `[r2] ` becomes `[r1] [r2] ...`; the parser strips exactly one prefix.
* Rank-0 and section text (`CCUTILS_MPI_PRINT_ONCE`, `CCUTILS_MPI_PRINTF_ONCE`, plain section text) gets the gutter `| ` on every line, so a text line such as `residual = 3` is never misread as a data line.
* In `inline` mode these prefixes also guarantee that rank text can never begin a line with `@ccutils `. The parser additionally requires `@ccutils {"v":` and valid JSON, and ignores anything else.
* Output printed by the application outside ccutils macros is not prefixed; the parser keeps it as best-effort section text.

**Multi-line JSON inside rank text.** Users may print pretty-printed JSON through `fp` (as `CCUTILS_JSON_PRETTY_PRINT` does). It stays ordinary text in both streams. The parser exposes `RankText.json()`, which finds single-line and multi-line values (the existing bracket-matching code is reused). The recommended path remains `CCUTILS_MPI_JSON_PRINT` / `put_local`, which emit typed `data` records and avoid text extraction altogether.

**Transport interaction.** Frames are **length-delimited**, so the gather is indifferent to newlines, blank lines and NUL bytes: it counts bytes only. Line splitting, prefixing and newline normalization happen in the local render step on each rank. Note that the rendered machine blob can be larger than the raw text (JSON escaping; chunk overhead), so the `INT_MAX` guard is evaluated on the **rendered** blob. This raises the priority of the optional chunked gather (Phase 4).

---

## 4. C++ API

### 4.1 Context stack
A section stack (per process; MPI ranks are processes) gives every emitted record its `section` automatically. Timers, `put` calls and prints inside a section inherit it.

### 4.2 RAII `ccutils::Section` (macros stay as thin wrappers)

```cpp
#include <ccutils/ccutils.hpp>   // new umbrella header (old headers still work)

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  CCUTILS_MPI_INIT;                           // unchanged

  ccutils::Section solve("solve", "Solve phase");     // == CCUTILS_MPI_SECTION_DEF

  {
    ccutils::Section spmv("spmv", "Sparse mat-vec");  // nested -> path "solve/spmv"

    for (int i = 0; i < 10; ++i) {
      auto t = spmv.time("kernel");           // scoped timer (ms), stops at end of scope
      run_kernel();
    }
    spmv.timer_start("halo");  exchange();  spmv.timer_stop("halo");

    spmv.put_local("bytes_sent", local_bytes);   // rank-local value
  }                                              // spmv ends: prints its timers, collective end

  solve.put("residual", residual);               // global value (stored on rank 0)
  solve.end();                                   // explicit end (collective in MPI builds)
}
```

Equivalent old-style code keeps working and maps onto the same machinery:

| Old macro | Maps to |
|-----------|---------|
| `CCUTILS_MPI_SECTION_DEF(name, title)` / `_END(name)` | `Section` construct / `end()` |
| `CCUTILS_MPI_SUBSECTION_DEF(sec, sub, title)` / `_END` | nested `Section` |
| `CCUTILS_MPI_GLOBAL_JSON_PUT(name, k, v)` | `section.put(k, v)` |
| `CCUTILS_MPI_LOCAL_JSON_PUT` / `CCUTILS_SECTION_JSON_PUT` | `section.put_local(k, v)` |
| `CCUTILS_SECTION_TIMER_*` / `CCUTILS_CPU_TIMER_*` | same storage, now registered in the section |
| `CCUTILS_MPI_JSON_DEF/PUT/PRINT(name)` | named rank-local data block, emitted as `data` records |
| `CCUTILS_MPI_TIMER_DEF/START/STOP/PRINT/PRINT_ROOT/INIT/CLOSE` | `ccutils::MpiTimer` with the same collective `print_stats(label, comm, root)` semantics (4.5) |

### 4.3 MPI-safety rules for the RAII class (important)
* `Section::end()` in an MPI build is **collective** (it triggers the gather) and is documented as such, exactly like `CCUTILS_MPI_SECTION_END` is today.
* The **destructor never runs collectives while unwinding** (`std::uncaught_exceptions() > 0`) or after `MPI_Finalize`. In those cases it writes a local, non-collective `section_end` record with `"ok":false`. Otherwise one rank throwing would hang the others in a gather.
* No hidden collectives are added to existing macros (C4). New behaviour is opt-in (for example reporting all MPI timers registered in a section with one gather at section end, see 4.5):

```cpp
solve.end({.mpi_timers = ccutils::TimerReport::Batch});   // one gather for all MPI timers registered in this section (default: Off)
```

### 4.4 Timer details
* Store samples as `double` (not `float`), keep `std::chrono::steady_clock`, unit stays milliseconds, and the unit is written into every record.
* Replace the `float` accumulation in `compute_stats` with `double` (Welford for variance).
* Keep `CCUTILS_TIMER_PRINT*` names; in v2 they emit a `timer` record + human row.
* **Aggregation across timers** (`CCUTILS_TIMER_SUM*`) stays as is; the print variants emit `timer_agg`.
* **CPU timers in MPI builds always carry `rank`** (`null` in non-MPI builds). Inside a section their records are *deferred* and emitted at the collective `Section::end()` through the same gather frames as other rank-local output, so they appear once per rank, in rank order, never interleaved. Outside any section a print is immediate and non-collective: one complete line, written by the calling rank according to 2.5.

### 4.5 MPI timers (`mpi_timers.hpp`)

**API.** The class is renamed `ccutils::MpiTimer`. The old global `TimerStats` stays as a plain alias (`using TimerStats = ccutils::MpiTimer;`), without a deprecation warning until 1.0, so `-Werror` builds keep compiling. The old macros keep their signatures.

```cpp
#include <ccutils/mpi/mpi_timers.hpp>

ccutils::MpiTimer halo("halo", comm);       // comm defaults to MPI_COMM_WORLD
for (int it = 0; it < 100; ++it) {
  auto s = halo.time();                     // RAII; or halo.start() / halo.stop()
  exchange_halos();
}
halo.report({.exclude_first_n = 5});        // COLLECTIVE over comm (C8)

// Old style, unchanged:
CCUTILS_MPI_TIMER_INIT(t);   /* ... */   CCUTILS_MPI_TIMER_CLOSE(t, "halo", comm);
```

**Report algorithm (v2).** Same participants and same `root` semantics as today (C8); only the internals change.

1. Each rank computes a fixed-size local summary: `{world_rank, n, sum, mean, M2, min, max, errors}`, where `M2` (sum of squared deviations) is computed around the **local mean in two passes**, so there is no cancellation.
2. **One `MPI_Gather`** of that struct (64 bytes per rank) to `root` replaces the five `MPI_Reduce` calls.
3. The root merges the summaries with the parallel (Chan) formula for the pooled `n/avg/sd/min/max/sum`, and computes the distribution of the per-rank `sum` and `avg` (`min`, `avg`, `max`, `argmin`, `argmax`), `imbalance` and `slowest_rank`.
4. The root emits one comm-scope `timer` record, and with `CCUTILS_TIMER_RANKS=all|top:K` also rank-scope records.

For very large communicators `report({.method = Method::Reduce})` uses two packed reductions plus `MPI_MAXLOC` instead of the gather (no per-rank table, but the pooled statistics, min/max and slowest rank are still available).

**Batching at section end.** Timers registered with the section (`auto& t = solve.mpi_timer("halo");`) can be reported together with `TimerReport::Batch`: one gather of an `n_timers x 8` matrix instead of `5 x n_timers` collectives. The batch needs every member of the communicator to register the same names in the same order. Before the gather a tiny `MPI_Allreduce` (min/max of timer count and a 64-bit hash of the names) checks this; on mismatch ccutils skips the batch, writes a `section_end` with `"ok":false` plus a warning record, and **never enters a gather with mismatched sizes**. Immediate `report()` calls stay immediate and unbatched.

**Misuse and errors.** Starting a running timer or stopping a stopped one throws `std::logic_error` by default (compatible). `CCUTILS_TIMER_MISUSE=record` counts it instead; the count travels in the same gather and is reported as `errors`. Documented hazard: a rank that throws and then skips `report()` leaves the others waiting in the gather.

**Where the output goes.** Through the Emitter (2.5): a root that is world rank 0 writes the main file; any other root prints directly to `stdout`/`stderr` as today, or writes the sidecar `<path>.r<rank>` when a file sink is configured. Records add `comm` (`name` from `MPI_Comm_get_name`, `size`, `root` as a **world** rank) so sub-communicator timers are identifiable. `rank` fields always mean the `MPI_COMM_WORLD` rank.

**Clocks and units.** MPI timers keep `MPI_Wtime` (seconds, converted to ms); CPU timers keep `steady_clock`; **both use `double`** (the `float` of `timers.hpp` goes away). Every record states `clock`, and `run_start` records `MPI_Wtick()` and `MPI_WTIME_IS_GLOBAL`.

**Parity with CPU timers.** `exclude_first_n` (warm-up), `report({.samples = true})` for raw `timer_samples` (opt-in; uses `MPI_Gatherv` with the same `INT_MAX` guards as all-print), and sums/aggregates over several MPI timers.

**Machine record example** (8 ranks, 100 samples each, rank 5 slowest):

```jsonl
{"v":2,"type":"timer","section":"solve","scope":"comm","rank":null,"name":"halo","unit":"ms","clock":"mpi_wtime","comm":{"name":"WORLD","size":8,"root":0},"ranks_reporting":8,"n":800,"avg":0.31,"sd":0.02,"min":0.29,"max":0.35,"sum":248.0,"per_rank":{"sum":{"min":24.1,"avg":31.0,"max":34.0,"argmin":2,"argmax":5},"avg":{"min":0.241,"avg":0.31,"max":0.34,"argmin":2,"argmax":5}},"imbalance":1.097,"slowest_rank":5,"errors":0,"seq":14}
```

With `CCUTILS_TIMER_RANKS=top:1` a rank-scope record follows:

```jsonl
{"v":2,"type":"timer","section":"solve","scope":"rank","rank":5,"name":"halo","unit":"ms","clock":"mpi_wtime","n":100,"avg":0.34,"sd":0.03,"min":0.30,"max":0.35,"sum":34.0,"errors":0,"seq":15}
```

**Human rendering of the same report** (column names in the header row are what the parser reads; `|` only separates the pooled part from the per-rank part):

```
  timers [ms]  (comm WORLD, 8 ranks)
    name   n    avg     sd     min     max      sum | rsum_min rsum_avg rsum_max slowest  imb
    halo   800  0.310   0.020  0.290   0.350  248.000 |   24.100   31.000   34.000  r5     1.10
```

**Legacy.** In `legacy` format `print_stats` runs the old body verbatim (five reductions, same formulas, root `printf` of the `group="MPI Timer"` line) so existing logs and numbers are unchanged (C3, C8).

---

## 5. MPI all-print: keep the transport, change only the payload

### 5.1 Today (to preserve)
```
each rank: open_memstream -> user block writes to fp -> buffer
MPI_Gather(sizes) -> rank 0 computes displs -> MPI_Gatherv(bytes) -> rank 0 prints -> free -> MPI_Barrier
```
Guards: per-rank size and total size must fit `INT` (abort otherwise).

### 5.2 Proposed split into two layers

```
 [render layer]   per rank, local, parallel      -> "frames"
 [transport]      UNCHANGED gather algorithm     -> rank 0 gets one blob per rank
 [write layer]    rank 0 only                    -> Emitter sinks
```

**Frames.** The per-rank buffer becomes a sequence of tiny frames instead of raw text:

```
frame := kind(1 byte: 'H' human | 'M' machine) | length(uint32) | payload
```

* Each rank renders its own human text (with the `[rN] ` prefix) and its own machine records (with `"rank":N` filled in) **locally**, only for sinks that are enabled.
* Rank 0 never parses JSON: it demultiplexes frames and copies `H` payloads to the human sink and `M` payloads to the machine sink, **in rank order**.
* It is still **one** `MPI_Gather` + **one** `MPI_Gatherv`, same guards, same barrier. The only difference is what the bytes contain.
* The user block still writes to `fp` (C2) and may print any number of lines. Its text is rendered once per sink: as `[rN] `-prefixed lines in an `H` frame, and as one or more `text` records (see 3.4) in an `M` frame, so nothing is lost in machine mode.
* Frame payloads are **length-delimited, not newline-delimited**, so multi-line text, blank lines and NUL bytes pass through the gather untouched.

Pseudo-code of the new macro body (the structure mirrors today's):

```cpp
#define CCUTILS_MPI_ALL_PRINT_IMPL(NAME, BLOCK) do {                         \
  /* STEP 1: render locally (unchanged: memstream + user block on fp) */     \
  std::string raw = ccutils::detail::capture([&](FILE* fp){ BLOCK; });       \
  std::string blob = ccutils::detail::make_frames(NAME, rank, raw);          \
  /* STEP 2-3: UNCHANGED gather of sizes and bytes to rank 0 */              \
  ccutils::detail::gatherv_to_root(MPI_COMM_WORLD, blob, /*out*/ per_rank);  \
  /* STEP 4: rank 0 only */                                                  \
  if (rank == 0) ccutils::detail::write_frames(NAME, per_rank);              \
  MPI_Barrier(MPI_COMM_WORLD);                                               \
} while (0)
```

`gatherv_to_root` is the **existing** code moved into a plain function (no behaviour change) so it can be unit-tested in isolation.

### 5.3 Human layout of an all-print block

```
-- all-ranks: halo_check ------------------------------------------
[r0] sent=1024 recv=1024
[r1] sent=1024 recv=2048   <- mismatch
[r2] sent=1024 recv=1024
[r3] (no output)
-- end all-ranks: halo_check --------------------------------------
```
Per-rank prefix lines are `grep`-friendly (`grep '^\[r1\]'`), and the begin/end markers keep the block parseable. Ranks with no output are omitted (as today) or optionally shown as `(no output)` with `CCUTILS_SHOW_EMPTY_RANKS=1`.

For ranks that print several lines, **every** line carries the `[rN] ` prefix (blank lines included) and text is never wrapped; see 3.4 for the full example and the escaping rules.

### 5.4 Behaviour that must stay identical
* Collective semantics: all ranks of `MPI_COMM_WORLD` must call it; ends with a barrier.
* Rank 0 is the single writer to stdout/file (no shared-file concurrent writes).
* `INT_MAX` guards and `MPI_Abort` on failure remain. (Optional, later and opt-in: chunked gather for outputs above 2 GiB.)
* `open_memstream` fallback message stays; if/when portability to platforms without it is needed, add a `tmpfile`/`std::ostringstream` fallback behind a feature check, not as a default change.
* Flush sequence (`fflush(stdout)` etc.) and `CCUTILS_FLUSH_WAIT` defaults stay (C6).
* Multi-line rank text keeps today's semantics: rank order, ranks with zero bytes omitted, a missing final newline is added in the human stream, and the whole block of a rank is printed contiguously (never interleaved with another rank's lines).
* In `legacy` format the **old macro body is used verbatim** (C3).

### 5.5 Compatibility test matrix (CI, Phase 0 onward)

| Axis | Values |
|------|--------|
| MPI implementation | MPICH, Open MPI (Docker images in CI); optionally Intel MPI / Cray MPICH / MVAPICH on a cluster |
| Launcher | `mpirun`/`mpiexec`, `srun` |
| stdout target | TTY, pipe, regular file, `srun --output` |
| Ranks | 1, 2, 4, 7, 16 |
| Per-rank payload | empty, 1 byte, no trailing newline, a few KB, ~100 MB; multi-line with blank lines (middle and end), CRLF, tabs, embedded NUL / invalid UTF-8, marker-lookalike lines, lines starting with `[r1] ` or `@ccutils `, pretty-printed JSON; different line counts per rank |
| Rank participation | all ranks; some ranks produce empty output |
| Timer communicators | `MPI_COMM_WORLD`; `MPI_Comm_split` sub-communicators; a communicator that does not contain world rank 0; `root != 0`; a communicator of size 1; ranks with zero samples; a timer still running at report |
| Build | with/without `CCUTILS_NO_JSON`, with/without `CCUTILS_JSON_PRETTY_PRINT`, `-Wall -Wextra -Werror`, C++11/14/17/20 |

Test oracle: **golden files**. In `legacy` mode the output must be byte-identical to what the current version prints (after normalizing timings). In `v2` mode the machine stream is validated against a JSON schema and the human stream against the grammar by the parser.

---

## 6. Parser v2

### 6.1 One model, many front ends

```
 text/log file
     |
     v
 detect_format()  ->  front end (jsonl | inline | human-v2 | legacy)
     |                      each yields a stream of Events (same schema as 3.1)
     v
 ModelBuilder  ->  Run (tree of Sections)  ->  tables / queries / CLI
```

* **JSONL front end**: trivial (`json.loads` per line).
* **Inline front end**: only lines starting with `@ccutils `, then as JSONL.
* **Human-v2 front end**: implements the grammar of 3.2 (slower, display precision, flagged in `run.lossy`).
* **Legacy front end**: the current `MPIOutputParser` logic, adapted to yield Events.
* **Autodetect**: first matching ccutils line decides (`{"v":2` -> jsonl, `@ccutils ` -> inline, `== ... ==` with `end` markers -> human-v2, old banners -> legacy). Can be forced with `format="..."`.

Consequence for the requirement "parser must work on human logs too": the data model and every query below behave the same regardless of the input format; only precision/speed differ.

### 6.2 Data model

```python
Run
 ├─ format: "jsonl" | "inline" | "human-v2" | "legacy"
 ├─ lossy: list[str]            # e.g. ["timers rounded to 3 decimals"]
 ├─ warnings: list[str]
 └─ root sections ──► Section
        ├─ name, title, path ("solve/spmv"), complete
        ├─ data:    dict            # global values  (scope == "global")
        ├─ ranks:   dict[int, RankData]   # rank-local values and text
        ├─ prints:  dict[str, AllPrint]   # all-print blocks; ranks[r].lines / .text / .json()
        ├─ timers:  dict[str, Timer]      # per name; Timer = comm stats and/or per-rank stats (see 6.3c)
        ├─ children: list[Section]
        └─ text: list[str]            # once/section text, lines preserved verbatim
```

### 6.3 Usage examples

```python
import ccutils

run = ccutils.load("run-1234.jsonl")       # or "slurm-1234.out", or "run.log"
run.tree()
```
```
run-1234.jsonl  (jsonl, 4 ranks, 0 warnings)
solve                         Solve phase
├─ data:    residual
├─ rank:    bytes_sent
└─ spmv                       Sparse mat-vec
   └─ timers: kernel, halo
```

```python
s = run["solve/spmv"]                    # path lookup (KeyError lists close matches)
s.timers["kernel"].avg_ms                # aggregated over ranks (mean of rank means)
s.timers["kernel"].rank(3).max_ms        # one rank
run["solve"].data["residual"]            # 1e-09
run["solve"].per_rank("bytes_sent")      # pandas Series indexed by rank
run.find("spmv")                         # all sections named spmv at any depth
```

Tables for analysis:

```python
df = run.timers_df()
#   section      name    rank   n   avg_ms  sd_ms  min_ms  max_ms  sum_ms
#   solve/spmv   kernel  0      10  1.2     0.1    1.1     1.4     12.0
#   solve/spmv   kernel  1      10  1.3     0.1    1.2     1.5     13.0
df.groupby(["section", "name"]).avg_ms.agg(["min", "mean", "max"])

run.data_df()                            # flattened via json_normalize
#   section  scope  rank  key         value
#   solve    global  NaN  residual    1e-09
#   solve    rank    0    bytes_sent  1048576
```

Same code on a human log:

```python
run = ccutils.load("run.log")            # autodetected: human-v2
run.lossy                                # ['timers rounded to 3 decimals']
run["solve/spmv"].timers["kernel"].avg_ms
```

### 6.3b Multi-line all-print text

```python
p = run["solve"].prints["convergence"]     # first call; works on jsonl, inline, human-v2
p.call                                     # 4
p.ranks[1].lines                           # ['iter 1', '  residual: 3.2e-3', '', 'iter 2 (converged)']
p.ranks[1].text                            # same, joined with '\n' (+ final '\n' if trailing_newline)
p.text_by_rank()                           # {0: '...', 1: '...'}
p.grep(r"converged")                       # [(rank, line_no, line), ...]
p.diff_ranks()                             # unified diff of each rank's text against rank 0
p.ranks[1].json()                          # JSON values inside the text, incl. multi-line pretty-printed ones

run["solve"].prints.calls("convergence")   # every call of that name, in order (loop case), each with .call
```

Parser rules for text (all front ends):
* Split on `\n` only; never `str.splitlines()`. A trailing `\r` is kept as content in the machine stream and stripped only from structure lines of CRLF-converted human logs (`newline="auto"`).
* Human front end: inside an all-ranks block, strip exactly one `[rN] ` (or `[rN]`) prefix from each line; only unprefixed lines are structure. Rank-0/section text is recognized by the `| ` gutter.
* Chunked machine records (`part`/`last`) are reassembled in `part` order; a missing `last` is reported as a warning (log truncated).
* Human-parsed text is exact except for the final-newline normalization; `run.lossy` notes it when `trailing_newline` cannot be recovered.

### 6.3c Timers: pooled, per-rank and imbalance

A `Timer` can have two views, and the parser fills whichever the log contains:

```
Timer
 ├─ name, unit, clock
 ├─ comm:  CommStats | None     # scope "comm": pooled n/avg/sd/min/max/sum, comm (name, size, root),
 │                              #   per_rank distribution, imbalance, slowest_rank, errors
 └─ ranks: dict[int, Stats]     # scope "rank": CPU timers, or MPI timers with CCUTILS_TIMER_RANKS=all|top:K
```

```python
t = run["solve"].timers["halo"]
t.comm.size                      # 8
t.avg_ms, t.sd_ms                # always pooled over ranks, weighted by n (from comm stats or rank stats)
t.imbalance, t.slowest_rank      # 1.097, 5
t.per_rank("sum")                # Series by rank if rank records exist, else the min/avg/max summary only

run.timers_df(level="comm")      # one row per (section, name, comm): pooled stats, imbalance, slowest_rank
run.timers_df(level="rank")      # one row per (section, name, rank)
run.timers_df(level="comm").query("imbalance > 1.2")     # which timers are badly balanced?
```

Mapping from older logs (so the same code works on them):

| Legacy / human input | Parser result |
|----------------------|---------------|
| `type=summary group="MPI Timer"` | `Timer.comm` with the pooled fields only (`imbalance`, `per_rank`, `comm.size` are `None`) |
| `type=summary group="Timer"` | `Timer.ranks[r]`; `r` comes from the enclosing `[[Rank N]]` block, otherwise `None` (legacy CPU timer lines carry no rank) |
| human table of 4.5 | `Timer.comm` including `imbalance`/`slowest_rank` (values rounded, flagged in `run.lossy`) |
| sidecar files `run.jsonl.r*` | merged automatically, listed in `run.sources` |

### 6.4 CLI

```bash
python -m ccutils tree   run.jsonl
python -m ccutils show   run.jsonl solve/spmv       # human rendering of one section
python -m ccutils timers run.jsonl --csv > timers.csv
python -m ccutils data   run.jsonl --section solve --key residual
python -m ccutils check  run.log                    # structural problems (unclosed sections, duplicate ranks)
```

### 6.5 Backwards compatibility of the Python API
* `parse`, `parse_file`, `parse_ccutils*`, `MPIOutputParser`, `Document`, `Section.json_blocks`, etc. remain importable from `ccutils_parser.py` (thin adapter on top of the legacy front end) and are marked deprecated, not removed.
* New package layout: `ccutils/` (`load`, `Run`, `Section`, front ends, CLI) + the old module as a shim.

---

## 7. Migration and versioning

| Release | Default `CCUTILS_FORMAT` | Notes |
|---------|--------------------------|-------|
| 0.x (now) | `legacy` | v2 available via env/config. Old parser untouched. |
| 0.y | `legacy` | v2 declared stable; parser v2 autodetects all formats. |
| 1.0 | `v2` | `legacy` still selectable via `CCUTILS_FORMAT=legacy` / `-DCCUTILS_FORMAT_LEGACY`. |

* Every machine record carries `v`; the parser rejects higher major versions with a clear message and ignores unknown fields.
* The human grammar and JSONL schema live in `docs/FORMAT.md` and are the single source of truth. Both the C++ renderers and the Python front ends are tested against it.

---

## 8. Testing strategy

1. **Golden tests (legacy):** run small programs under MPI, normalize timing numbers, compare byte-for-byte with output from the current version.
2. **Schema tests (v2 machine):** validate every record with a JSON Schema; random fuzz of keys/values (quotes, newlines, unicode, NaN/inf, huge ints).
3. **Round-trip tests:** program -> (human log, machine log) -> parser. Assert `load(human)` and `load(machine)` produce equal models, modulo `lossy`.
4. **MPI protocol tests:** unit-test `gatherv_to_root` alone (empty ranks, 1 byte, no trailing newline, large payload); run the full matrix of 5.5 in CI.
5. **Failure tests:** exception inside a section on one rank (no hang, `ok:false` record), early return, rank with zero output, unclosed section at `MPI_Abort`.
6. **Parser tests:** truncated logs (job killed mid-section), interleaved application output, ANSI-colored logs, CRLF, files with only some formats.
8. **MPI timer tests:**
   * *Numerics:* compare pooled `n/avg/sd/min/max/sum` and the per-rank distribution with numpy on synthetic samples, including the cancellation case (average 1e4 ms, sd 1e-3 ms) that breaks the sum-of-squares formula.
   * *Communicators:* `MPI_COMM_WORLD`, `MPI_Comm_split` sub-communicators, a communicator without world rank 0, `root != 0`, size 1.
   * *Edge cases:* ranks with zero samples, a timer still running at report, misuse with `throw` and with `record`.
   * *Batching:* matching registration, and mismatched names/counts (must warn and skip the batch, never hang).
   * *Legacy:* output of `print_stats` byte-identical to the current implementation (numbers included).
   * *Writers:* non-zero root with a file sink writes the sidecar; the parser merges it; no torn lines when many ranks print to one pipe.
7. **Multi-line text tests:** for every case in the payload row of 5.5, check that `load(human)` and `load(machine)` give identical `lines` per rank (and `trailing_newline` for machine). Include: blank first/middle/last lines, no final newline, `\r`, `\f`, `\u2028` inside a line, a rank line equal to a section end marker, a line starting with `[r2] `, a line starting with `@ccutils `, 10^5-line output (chunking), invalid UTF-8 (base64), the same all-print name called in a loop, and ranks with different line counts.

---

## 9. Roadmap

### Phase 0 - Safety net (no behaviour change)
* CI with MPICH + Open MPI, `-Wall -Wextra`, sanitizers where possible.
* Golden-output tests of today's output.
* Fix the small bugs from section 1 (buffer overflow, include guard, reserved identifiers, trailing `;`).
* `mpi_timers.hpp` hygiene: stray `;` in `CCUTILS_MPI_TIMER_START/STOP`, `do { } while (0)` around `CCUTILS_MPI_TIMER_CLOSE`, include guard name, include style, unused `<iostream>`. Rename the class to `ccutils::MpiTimer` and keep `TimerStats` as an alias. No behaviour change.
* Golden test for `print_stats` output on `WORLD`, a split communicator and `root != 0`.
* **Done when:** CI green, golden files committed, no output diff.

### Phase 1 - Sinks and configuration
* `Emitter`, `Sink`, `Config`, env parsing, path templates, TTY-aware color.
* `legacy` mode routes through the Emitter but prints the same bytes.
* Writer rules of 2.5: world-rank-0 file, direct `stdout`/`stderr` for non-collective emits, sidecar files for other writers.
* **Done when:** `CCUTILS_HUMAN_OUT=file:...` works in legacy mode; golden tests unchanged.

### Phase 2 - Context, sections, machine stream
* Section stack, `ccutils::Section`, `put`/`put_local`, timers registered per section, `double` stats.
* JSONL renderer + schema (`FORMAT.md`) + schema validation tests.
* Old macros re-implemented on top, still emitting legacy text when `FORMAT=legacy`.
* CPU timers: `rank` field in MPI builds, deferred emission inside sections (4.4), `double` milliseconds, `clock` field; `run_start` record with `MPI_Wtick` / `MPI_WTIME_IS_GLOBAL`.
* **Done when:** a v2 run produces a valid JSONL log with correct `section`/`rank` context.

### Phase 3 - Human renderer v2
* Tables, rank prefixes, indentation, collapse of identical per-rank values, `inline` mode.
* **Done when:** renderer output matches the grammar in 3.2.

### Phase 4 - MPI integration (highest risk)
* Move gather code into `gatherv_to_root` (unchanged algorithm), add frames, rank-0 demux.
* `ccutils::MpiTimer` (4.5): single-gather report with stable statistics, per-rank distribution, `imbalance`, `slowest_rank`, `CCUTILS_TIMER_RANKS`, `exclude_first_n`, misuse counting; batched report at section end (opt-in, with the name/count consistency check); legacy `print_stats` body kept verbatim.
* Optional: `%r` per-rank files, chunked gather, `Method::Reduce` for very large communicators, raw `timer_samples` (all opt-in).
* **Done when:** full compatibility matrix (5.5) passes; legacy mode still byte-identical.

### Phase 5 - Parser v2
* Package layout, front ends (jsonl, inline, human-v2, legacy), model builder, `tree()`, DataFrames, CLI, deprecation shim.
* **Done when:** round-trip tests pass for all four formats.

### Phase 6 - Docs and switch
* `docs/FORMAT.md`, README with the examples above, migration guide, default flipped to `v2` for 1.0.

---

## 10. Open questions for you

1. Should rank-local data in the **human** stream default to a rank x key table, or stay one `[rN]` line per value? (Table is nicer; lines are easier to grep.)
2. Is a hard dependency on pandas acceptable in the parser, or should DataFrame support be optional (`pip install ccutils[pandas]`) with plain lists/dicts as fallback? (I recommend optional.)
3. Do you need communicators other than `MPI_COMM_WORLD` for all-print? The current macro is hard-wired to it; a `comm` parameter could be added without changing existing calls.
4. Which MPI/launcher combinations do you actually run on? That decides how much of the matrix in 5.5 must run on real clusters rather than Docker.
5. How large can a single rank's all-print text get in practice? Since the machine blob is larger than the raw text (JSON escaping), the `INT_MAX` abort may trigger earlier than today. If outputs above ~1 GiB per rank are plausible, the chunked gather should move from optional to default in v2.
6. Is the `| ` gutter for rank-0/section text acceptable visually, or would you prefer a different marker (for example `[--]`)? It exists so multi-line once-text can never be mistaken for `key = value` lines.
7. Which per-rank quantity should define `imbalance`? My default is the per-rank **total** (`sum`), because it bounds the phase; the alternative is the per-rank mean sample duration. Both distributions are in the record, so this only decides the headline number.
8. Is the **sidecar file** rule (2.5) acceptable for non-rank-0 writers with a file sink, or would you rather restrict file sinks to world rank 0 and let timers on communicators that exclude it fall back to `stdout`?
9. Is `CCUTILS_TIMER_RANKS=summary` the right default (one comm-scope record with the per-rank min/avg/max), or do you usually want per-rank rows as well?
