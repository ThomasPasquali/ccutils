# MPI utilities

Configure CCUTILS with `-DCCUTILS_ENABLE_MPI=ON`, then link `ccutils::ccutils_mpi` in
CMake. In Make projects use an MPI compiler wrapper and define `CCUTILS_ENABLE_MPI`.
The 1.0 default human output uses path-based section markers, timer tables, and rank-prefixed
all-print text. Set `CCUTILS_FORMAT=legacy` to retain the earlier layout. To save records
for analysis, set `CCUTILS_MACHINE_OUT=file:run.jsonl` before launching the program.

```cpp
#include <mpi.h>
#include <ccutils/mpi/mpi_macros.hpp>
#include <ccutils/mpi/mpi_timers.hpp>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    CCUTILS_MPI_INIT
    CCUTILS_MPI_PRINTF_ONCE("world size: %d\n", ccutils_inmacro_ntask);
    CCUTILS_MPI_ALL_PRINT_NAMED(ranks, fprintf(fp, "hello from rank %d\n", ccutils_inmacro_myid));

    const std::string run = "run";
    const std::string measurements = run + "/measurements";
    CCUTILS_MPI_SECTION_PATH_DEF(run, "Run")
    CCUTILS_MPI_SECTION_PATH_DEF(measurements, "Measurements")
    CCUTILS_MPI_SECTION_PATH_RANK_JSON_PUT(measurements, "rank_value", ccutils_inmacro_myid)
    ccutils::MpiTimer& step_timer = ccutils::section(measurements).mpi_timer("step");
    step_timer.start();
    /* collective or local work */
    step_timer.stop();
    CCUTILS_MPI_SECTION_PATH_END(measurements)
    CCUTILS_MPI_SECTION_PATH_END(run)

    CCUTILS_MPI_JSON_DEF(metadata);
    CCUTILS_MPI_JSON_PUT(metadata, "rank", ccutils_inmacro_myid);
    CCUTILS_MPI_JSON_PRINT(metadata);

    CCUTILS_MPI_TIMER_DEF(work)
    CCUTILS_MPI_TIMER_START(work)
    /* collective or local work */
    CCUTILS_MPI_TIMER_STOP(work)
    CCUTILS_MPI_TIMER_PRINT(work, "work", MPI_COMM_WORLD)
    MPI_Finalize();
}
```

All processes in a communicator must call collective operations in the same order. The
all-print block writes to the provided `fp`; `CCUTILS_MPI_ALL_PRINT` gathers rank buffers
and writes them on the communicator root. `CCUTILS_MPI_ALL_PRINT_COMM` accepts an explicit
communicator. `CCUTILS_MPI_ONCE` and `CCUTILS_MPI_PRINTF_ONCE` require
`CCUTILS_MPI_INIT` first. Timer reporting is collective over its communicator and must be
called by every member; an empty timer emits no summary.

MPI section markers and global values are written by rank zero. Rank-local values are
preserved with their rank in the machine stream and shown with rank labels in human output.
`CCUTILS_MPI_JSON_PRINT` gathers standalone rank-local JSON and must be called by every
rank in matching order. See the [MPI macro reference](../API.md#optional-modules),
[output settings](../OUTPUT.md), and [machine record format](../FORMAT.md).
