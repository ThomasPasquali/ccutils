# Sections across files

Sections are a user-defined tree addressed by slash-separated paths. A child path must
extend the currently open parent path. Paths may be constants or assembled at runtime.

```cpp
const std::string root = "solver";
const std::string step = root + "/iteration_" + std::to_string(iteration);
CCUTILS_SECTION_PATH_DEF(root, "Solver")
CCUTILS_SECTION_PATH_DEF(step, "Solver iteration")

ccutils::section(step).put("residual", residual);
ccutils::TimerRef factor = ccutils::section(step).timer("factorization");
factor.start();
factorization();
factor.stop();

CCUTILS_SECTION_PATH_END(step)
CCUTILS_SECTION_PATH_END(root)
```

Closing a section emits its registered timer summaries and nonempty JSON. For MPI timers,
use `ccutils::section(path).mpi_timer(name)`; every rank must register the same timer names
and sequence, and section close reports each timer collectively. Open parents
before children and close children first. Each full path may be opened once during a
process. Reopening a path, opening a child without its parent, or closing a parent while a
child remains open prints a specific diagnostic and terminates the program.

The tree owns each node's global JSON and timers. With MPI-enabled section APIs, each node
also has rank-local JSON. Close is collective over the relevant communicator: ranks close
the same path in the same order, synchronize, and rank 0 writes the gathered values. A
non-collective close is not provided for MPI sections.

## Naming and JSON

Use the `CCUTILS_SECTION_PATH_*` macros when the path is a string expression. JSON is
available through `ccutils::section(path).put(key, value)` or the corresponding path macro.
MPI path sections provide `CCUTILS_MPI_SECTION_PATH_GLOBAL_JSON_PUT` and
`CCUTILS_MPI_SECTION_PATH_RANK_JSON_PUT`. The parser exposes global values as
`section.data` (or `section.global_data`) and rank values as `section.rank_data` /
`section.per_rank(key)`; scripts do
not need to know internal print or JSON block names.

Legacy section/subsection macro spellings remain transitional wrappers. New code should
model every node as a section path. In Python, read global JSON from `run[path].data` and
rank-local JSON from `run[path].rank_data` or `run[path].per_rank(key)`. For all-ranks text,
provide an explicit name with `CCUTILS_MPI_ALL_PRINT_NAMED` or
`CCUTILS_MPI_ALL_PRINT_NAMED_COMM`; the unnamed forms use the stable fallback name
`all_ranks`.
