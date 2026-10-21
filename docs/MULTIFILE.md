# Using ccutils across files and functions

Sections, timers and JSON objects are looked up **by name**, so a function in another
file can update what `main()` created. Nothing has to be passed around and no variable
has to be shared. Include the same ccutils headers in every file that uses them.

```cpp
// main.cpp
#include <ccutils/ccutils.hpp>
void solve(int n);                                  // solver.cpp

int main() {
  CCUTILS_SECTION_DEF(run, "Whole run");
  CCUTILS_CPU_TIMER_DEF(kernel);
  solve(4);
  CCUTILS_TIMER_PRINT(kernel);                      // includes the samples taken in solve()
  CCUTILS_SECTION_END(run);                         // prints everything solve() added
}

// solver.cpp
#include <ccutils/ccutils.hpp>
void solve(int n) {
  CCUTILS_CPU_TIMER_USE(kernel);                    // main's timer, by name
  CCUTILS_SUBSECTION_DEF(run, solve, "Solve");      // nested in main's section
  for (int i = 0; i < n; ++i) {
    CCUTILS_CPU_TIMER_START(kernel);
    /* ... */
    CCUTILS_CPU_TIMER_STOP(kernel);
  }
  CCUTILS_SECTION_JSON_PUT(run, "iterations", n);   // main's section, by name
  CCUTILS_SUBSECTION_END(run, solve);
}
```

## What works from any function

| You want to | Macro | Object API |
|-------------|-------|------------|
| add data to an open section | `CCUTILS_SECTION_JSON_PUT(name, key, value)`, `..._SUB_PUT` | `ccutils::section("name").put(key, value)` |
| add data to the innermost open section | | `ccutils::current_section().put(key, value)` |
| open / close a subsection | `CCUTILS_SUBSECTION_DEF/END(section, subsection, ...)` | |
| add data to a subsection | `CCUTILS_SUBSECTION_JSON_PUT(section, subsection, key, value)` | `ccutils::section("subsection").put(...)` |
| use a timer defined elsewhere | `CCUTILS_CPU_TIMER_USE(name)`, then the usual `START` / `STOP` / `PRINT` / `SUM` macros | `ccutils::TimerRef t = ccutils::timer("name");` `t.start() t.stop() t.print() t.count() t.stats()` |
| time the enclosing scope | `CCUTILS_TIMER_SCOPE(name)` | `ccutils::ScopedTimer s("name");` |
| use a JSON object defined elsewhere | `CCUTILS_JSON_USE(name)` | `ccutils::named_json("name")` |

MPI equivalents: `CCUTILS_MPI_TIMER_USE(name)` / `ccutils::mpi_timer("name")`,
`CCUTILS_MPI_JSON_USE(name)`, `CCUTILS_MPI_GLOBAL_JSON_PUT` and
`CCUTILS_MPI_LOCAL_JSON_PUT` (by section name), `CCUTILS_MPI_SUBSECTION_*`.

## MPI

`CCUTILS_MPI_INIT` is optional. It declares `ccutils_inmacro_myid` / `ccutils_inmacro_ntask`
for *your* code in that scope; the ccutils macros no longer need them, so they work in
any function. Elsewhere use `ccutils::mpi_rank()` and `ccutils::mpi_size()`.

```cpp
// worker.cpp - no CCUTILS_MPI_INIT, no shared variable
void do_work() {
  CCUTILS_MPI_TIMER_USE(halo);                                      // main's MPI timer
  CCUTILS_MPI_SUBSECTION_DEF(run, work, "Work");
  CCUTILS_MPI_GLOBAL_JSON_PUT(run, "iterations", 3);                // main's section
  CCUTILS_MPI_LOCAL_JSON_PUT(run, "rank", ccutils::mpi_rank());
  CCUTILS_MPI_ALL_PRINT_NAMED(worker, fprintf(fp, "rank %d\n", ccutils::mpi_rank()));
  CCUTILS_MPI_SUBSECTION_END(run, work);
}
```

The collective rules are unchanged: `CCUTILS_MPI_ALL_PRINT*`, the `*_SECTION_END` /
`*_SUBSECTION_END` macros and `CCUTILS_MPI_TIMER_PRINT*` must be reached by every rank
of the communicator, whichever file they are in.

## Rules

* **Names are program-wide.** If several open sections (or live timers) share a name, the
  innermost / most recently defined one is used.
* **A section must be open** when you update it. Otherwise the data is ignored and ccutils
  prints one warning on stderr. Data added after the section has ended is never printed.
* **A timer lives as long as its scope.** `CCUTILS_CPU_TIMER_DEF` registers the variables it
  declares; they are unregistered when the scope ends. `USE`/`ccutils::timer()` on a name
  that no live timer has creates a persistent timer instead of failing.
* `CCUTILS_CPU_TIMER_IMPORT` (extern, for a timer defined at file scope) still works.
* **Use ccutils from one thread.** Registry lookup and output sinks use locks, but section
  JSON values and timer state do not. Human banners and their bodies are also separate writes,
  so concurrent output can interleave. The locks do not make concurrent section, timer, or
  output calls safe.
* With `CCUTILS_NO_JSON` the JSON macros compile to nothing, including in other files.
