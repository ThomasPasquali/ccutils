# Core API

Include `<ccutils/ccutils.hpp>` for core utilities, or include one header such as `<ccutils/macros.hpp>` or `<ccutils/timers.hpp>`. MPI and CUDA headers must be included explicitly. Headers do not initialize MPI or CUDA state for the application.

## Sections and JSON

```cpp
CCUTILS_SECTION_DEF(setup, "Setup")
CCUTILS_SECTION_JSON_PUT(setup, "device_count", 2)
CCUTILS_SECTION_JSON_SUB_PUT(setup, "config", "mode", "fast")

CCUTILS_SUBSECTION_DEF(setup, hardware, "Hardware")
CCUTILS_SUBSECTION_JSON_PUT(setup, hardware, "threads", 8)
CCUTILS_SUBSECTION_JSON_SUB_PUT(setup, hardware, "device", "name", "GPU")
CCUTILS_SUBSECTION_END(setup, hardware)

CCUTILS_SECTION_END(setup)
```

Each section and subsection name is a C++ identifier and must be unique among its siblings. Subsections are nested section markers, so the parser attaches them as child sections. A nonempty section or subsection emits a `ccutils_json` block containing a JSON object. JSON keys and values use nlohmann/json assignment rules. Define `CCUTILS_NO_JSON` to disable JSON support; JSON storage/put/print macros then become no-ops, while section and subsection markers remain available.

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

Elapsed time is milliseconds from `std::chrono::steady_clock`. `CCUTILS_TIMER_VALUES(name)` exposes the underlying `std::vector<float>`. `CCUTILS_TIMER_STATS(name)` returns `ccutils_timers::TimerStats` (`n`, `avg`, sample `stddev`, `min`, `max`, `sum`). Printing variants include `CCUTILS_TIMER_PRINT_LAST`, `CCUTILS_TIMER_PRINT_ALL`, `CCUTILS_TIMER_PRINT_EXCLUDING_FIRST_N`, and `CCUTILS_TIMER_SUM`, `_AVG`, and `_LAST` (up to ten timer names).

Timer print macros produce no output when their timer has no samples; aggregate print macros produce no output when all input timers are empty.

Timer output uses versioned key/value lines prefixed by `CCUTILS_TIMER`. For example, with colors disabled:

```text
CCUTILS_TIMER v=1 type=summary group="Timer" name="work" n=3 avg_ms=1.250000 stddev_ms=0.100000 min_ms=1.150000 max_ms=1.350000 sum_ms=3.750000
```

Summary fields have fixed names and units. Labels are JSON-quoted strings, so quotes, spaces, and line breaks do not break record boundaries. With colors enabled, timer and section markers are ANSI-colored; the CCUTILS parser strips those codes before parsing. Other output (including ordinary `printf`) can be freely interleaved; consumers can select timer records by their prefix. Record types are `summary`, `sample`, `run`, and `aggregate`. Print macros emit no record when the timer has no samples. Run indices start at zero. The `v=1` field identifies this record layout.

## Other core headers

- `colors.h`: ANSI escape string constants (`RED`, `BRIGHT_CYAN`, `RESET`, and background variants).
- `formats.h`: output format string constants used by CCUTILS.
- `easyargs.hpp`: standalone EasyArgs command-line parser API; see its header comments and [upstream guide](https://github.com/gouwsxander/easy-args).
- `ccutils_parser.py`: installed Python module; import `parse_file` or `parse` for a complete `Document` (sections, JSON, MPI output, and timer records), or `parse_timer_records` to extract just timer lines. Compatibility helpers `parse_ccutils_output` and `parse_ccutils_output_file` return the section mapping.

## Optional modules

MPI: include `<ccutils/mpi/mpi_macros.hpp>` and/or `<ccutils/mpi/mpi_timers.hpp>` after MPI headers. Link `ccutils::ccutils_mpi`. The application must call `MPI_Init` before CCUTILS MPI operations and ensure all ranks enter collective operations.

MPI subsections use `CCUTILS_MPI_SUBSECTION_DEF/END` and the matching `CCUTILS_MPI_SUBSECTION_JSON_PUT/SUB_PUT` macros. Markers are printed once; nonempty subsection JSON is gathered and printed with rank markers. For standalone rank-local JSON, use `CCUTILS_MPI_JSON_DEF/PUT/SUB_PUT/PRINT`; every rank must call `CCUTILS_MPI_JSON_PRINT` in the same order, and each nonempty object's JSON is labeled by rank. With `CCUTILS_NO_JSON`, storage and JSON output are disabled while subsection markers remain.

CUDA: include `<ccutils/cuda/cuda_macros.hpp>`, `<ccutils/cuda/cuda_timers.hpp>`, or `<ccutils/cuda/cuda_utils.hpp>` from CUDA-aware code. Link `ccutils::ccutils_cuda`. The application owns device selection, streams, and CUDA resource lifetimes.
