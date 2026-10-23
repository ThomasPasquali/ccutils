# Core API

Include `<ccutils/ccutils.hpp>` for core utilities, or include one header such as `<ccutils/macros.hpp>` or `<ccutils/timers.hpp>`. MPI and CUDA headers must be included explicitly. Headers do not initialize MPI or CUDA state for the application.

## Sections and JSON

```cpp
const std::string phase = "setup";
const std::string hardware = phase + "/hardware";
CCUTILS_SECTION_PATH_DEF(phase, "Setup")
CCUTILS_SECTION_PATH_JSON_PUT(phase, "device_count", 2)
CCUTILS_SECTION_PATH_DEF(hardware, "Hardware")
CCUTILS_SECTION_PATH_JSON_PUT(hardware, "threads", 8)
CCUTILS_SECTION_PATH_JSON_SUB_PUT(hardware, "device", "name", "GPU")
CCUTILS_SECTION_PATH_END(hardware)
CCUTILS_SECTION_PATH_END(phase)
```

Sections form a tree from slash-separated path strings. Open parents before children and
close children first. A path can be opened only once per process. Reopening a closed path,
opening a child without its parent, or closing a parent while a child is open prints a
specific `ccutils section error` and terminates the process. Paths and titles can be
ordinary strings, including formatted strings assembled at runtime. A section owns its
global JSON, MPI rank-local JSON, and timers explicitly obtained through
`ccutils::section(path).timer(name)` or (with MPI headers) `.mpi_timer(name)`; nonempty
values are emitted when that section closes. MPI timer names and close order must match on
all ranks because each timer is reported collectively.
JSON keys and values use nlohmann/json assignment rules. Define `CCUTILS_NO_JSON` to
disable JSON support.

The older `CCUTILS_SUBSECTION_*` macro spellings are transitional wrappers around child
paths. New code should use `CCUTILS_SECTION_PATH_*` for every node in the tree; in Python,
read the resulting values through `run[path].data` and `run[path].rank_data`.

For JSON output outside sections, declare a named object and print it when ready:

```cpp
CCUTILS_JSON_DEF(run_metadata);
CCUTILS_JSON_PUT(run_metadata, "build", "release");
CCUTILS_JSON_SUB_PUT(run_metadata, "host", "cores", 16);
CCUTILS_JSON_PRINT(run_metadata);
```

The print macro emits a delimited JSON block named after the object and produces no output for an empty object. Names are C++ identifiers; JSON keys and values follow nlohmann/json assignment rules.

JSON output is compact by default. Define `CCUTILS_JSON_PRETTY_PRINT` to format long objects across lines while keeping short values inline. Arrays are kept compact when they fit; longer arrays are wrapped into lines containing multiple values. Set `CCUTILS_JSON_LINE_WIDTH` to change the target width (default `100` columns), for example `-DCCUTILS_JSON_PRETTY_PRINT -DCCUTILS_JSON_LINE_WIDTH=120`. This applies to core and MPI JSON output macros.

`CCUTILS_PRINT_SPLIT(title)` and `CCUTILS_DEBUG_PRINT(format, ...)` write to stdout. `CCUTILS_ASSERT(condition, format, ...)` writes a diagnostic to stderr and terminates with `EXIT_FAILURE` when the condition is false.

## Color levels

Set `CCUTILS_COLOR_LEVEL` as a compiler definition (or before including CCUTILS headers) to select the ANSI palette:

| Level | Output |
| --- | --- |
| `0` | No ANSI escape codes |
| `1` | Basic 8 foreground colors |
| `2` | Bright 16 foreground colors |
| `3` | Bright foreground colors and colored backgrounds on CCUTILS output markers |

For example, in CMake use `target_compile_definitions(myapp PRIVATE CCUTILS_COLOR_LEVEL=0)`, or with Make add `-DCCUTILS_COLOR_LEVEL=1` to compiler flags. Level 2 is the default. Existing `CCUTILS_NO_COLOR` definitions remain supported and force level 0.

## CPU timers

```cpp
#include <ccutils/timers.hpp>
CCUTILS_CPU_TIMER_DEF(work)
CCUTILS_CPU_TIMER_START(work)
do_work();
CCUTILS_CPU_TIMER_STOP(work)
CCUTILS_TIMER_PRINT(work)
```

Elapsed time is milliseconds from `std::chrono::steady_clock`. `CCUTILS_TIMER_VALUES(name)` exposes the underlying `std::vector<double>`. `CCUTILS_TIMER_STATS(name)` returns `ccutils_timers::TimerStats` (`n`, `avg`, sample `stddev`, `min`, `max`, `sum`), with compensated summation for `sum`. Printing variants include `CCUTILS_TIMER_PRINT_LAST`, `CCUTILS_TIMER_PRINT_ALL`, `CCUTILS_TIMER_PRINT_EXCLUDING_FIRST_N`, and `CCUTILS_TIMER_SUM`, `_AVG`, and `_LAST` (up to ten timer names).

Timer print macros produce no output when their timer has no samples; aggregate print macros produce no output when all input timers are empty.

The default v2 human output displays timers as tables in milliseconds. Set
`CCUTILS_FORMAT=legacy` to emit the previous versioned key/value lines prefixed by
`CCUTILS_TIMER`. For example, with legacy format and colors disabled:

```text
CCUTILS_TIMER v=1 type=summary group="Timer" name="work" n=3 avg_ms=1.250000 stddev_ms=0.100000 min_ms=1.150000 max_ms=1.350000 sum_ms=3.750000
```

Other output (including ordinary `printf`) can be freely interleaved. Legacy record types
are `summary`, `sample`, `run`, and `aggregate`. Print macros emit no output when the timer
has no samples. Run indices start at zero. The `v=1` field identifies this legacy record
layout. For structured records independent of the human format, enable
`CCUTILS_MACHINE_OUT`; see [FORMAT.md](FORMAT.md).

## RAII sections

`ccutils::Section` opens a section when constructed and closes it when it leaves
scope. Use `put()` to add data. Call `end()` to close it explicitly. During exception
unwinding, its destructor omits the human end banner and records `ok:false`.

```cpp
{
  ccutils::Section solve("solve", "Solve phase");
  solve.put("iterations", 12);
}
```

With MPI, `ccutils::MpiSection::end()` is collective and all ranks must call it. Its
destructor only closes local state and never gathers; call `end()` when rank-local data
must be gathered and printed.


## Other core headers

- `colors.h`: ANSI escape string constants (`RED`, `BRIGHT_CYAN`, `RESET`, and background variants).
- `formats.h`: output format string constants used by CCUTILS.
- `easyargs.hpp`: standalone EasyArgs command-line parser API; see its header comments and [upstream guide](https://github.com/gouwsxander/easy-args).
- Python: import `ccutils` and use `ccutils.load()` for a `Run` model across JSONL, inline,
  human-v2, and legacy logs. See [PARSER_V2.md](PARSER_V2.md). The older `Document` parser
  API remains available from `ccutils.ccutils_parser` for compatibility.

## Optional modules

MPI: include `<ccutils/mpi/mpi_macros.hpp>` and/or `<ccutils/mpi/mpi_timers.hpp>` after MPI headers. Link `ccutils::ccutils_mpi`. The application must call `MPI_Init` before CCUTILS MPI operations and ensure all ranks enter collective operations.

MPI section close is collective: every rank in the communicator must close the same path
in the same order. Close synchronizes, gathers nonempty rank-local JSON, and writes the
result on rank zero. Use `CCUTILS_MPI_SECTION_PATH_DEF/END` with
`CCUTILS_MPI_SECTION_PATH_GLOBAL_JSON_PUT` and
`CCUTILS_MPI_SECTION_PATH_RANK_JSON_PUT`. Unnamed all-ranks text uses the stable name
`all_ranks`; prefer `CCUTILS_MPI_ALL_PRINT_NAMED` or its communicator variant to give
each print a meaningful name.

CUDA: include `<ccutils/cuda/cuda_macros.hpp>`, `<ccutils/cuda/cuda_timers.hpp>`, or `<ccutils/cuda/cuda_utils.hpp>` from CUDA-aware code. Link `ccutils::ccutils_cuda`. The application owns device selection, streams, and CUDA resource lifetimes.
