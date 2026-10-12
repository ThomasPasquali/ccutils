# MPI utilities

Configure CCUTILS with `-DCCUTILS_ENABLE_MPI=ON`, then link `ccutils::ccutils_mpi` in CMake. In Make projects use an MPI compiler wrapper and define `CCUTILS_ENABLE_MPI`.

```cpp
#include <mpi.h>
#include <ccutils/mpi/mpi_macros.hpp>
#include <ccutils/mpi/mpi_timers.hpp>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    CCUTILS_MPI_INIT
    CCUTILS_MPI_PRINTF_ONCE("world size: %d\n", ccutils_inmacro_ntask);
    CCUTILS_MPI_ALL_PRINT_NAMED(ranks, fprintf(fp, "hello from rank %d\n", ccutils_inmacro_myid));

    CCUTILS_MPI_SECTION_DEF(run, "Run")
    CCUTILS_MPI_SUBSECTION_DEF(run, measurements, "Measurements")
    CCUTILS_MPI_SUBSECTION_JSON_PUT(run, measurements, "rank_value", ccutils_inmacro_myid)
    CCUTILS_MPI_SUBSECTION_END(run, measurements)
    CCUTILS_MPI_SECTION_END(run)

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

All processes must call collective operations in the same order. The print block writes to the provided `fp`; `CCUTILS_MPI_ALL_PRINT` gathers nonempty rank buffers and writes them on rank zero. Ranks with no buffered content, and named blocks with no content on any rank, produce no markers. `CCUTILS_MPI_ONCE` and `CCUTILS_MPI_PRINTF_ONCE` require `CCUTILS_MPI_INIT` first. Timer aggregation also requires every rank in the communicator to call print together; it emits nothing if there are no samples across the communicator.

MPI subsection markers are printed once, while subsection JSON is gathered and labeled by rank. `CCUTILS_MPI_JSON_PRINT` does the same for standalone rank-local JSON objects and must be called by every rank in matching order.
