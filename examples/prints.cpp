#include <ccutils/timers.hpp>
#include <ccutils/macros.hpp>
#include <cstdio>

#ifdef WITH_MPI
    #include <mpi.h>
    #include <ccutils/mpi/mpi_timers.hpp>
    #include <ccutils/mpi/mpi_macros.hpp>
#endif

int main(int argc, char **argv) {
#ifdef WITH_MPI
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
#endif

    CCUTILS_SECTION_DEF(section1, "an output section")
    CCUTILS_SECTION_JSON_PUT(section1, "myval", "value")
    CCUTILS_SECTION_JSON_SUB_PUT(section1, "topkey", "innerkey", "nested value")
    CCUTILS_SECTION_END(section1)
    printf("\n\n");

#ifdef WITH_MPI
    }
    CCUTILS_MPI_INIT
    CCUTILS_MPI_TIMER_DEF(mpi_timer)
    for (size_t i = 0; i < 3; i++) {
        CCUTILS_MPI_TIMER_START(mpi_timer)
        usleep(250);
        CCUTILS_MPI_TIMER_STOP(mpi_timer)
    }
    float second_time = CCUTILS_TIMER_VALUES(mpi_timer)[1];
    
    CCUTILS_MPI_SECTION_DEF(mpi_section, "this works with MPI as well")
    CCUTILS_MPI_LOCAL_JSON_PUT(mpi_section, "value_stored_per_rank", second_time)
    CCUTILS_MPI_GLOBAL_JSON_PUT(mpi_section, "value_stored_once", "one value")
    CCUTILS_MPI_GLOBAL_JSON_PUT(mpi_section, "rank_0_timer_values", CCUTILS_TIMER_VALUES(mpi_timer))
    CCUTILS_MPI_SECTION_END(mpi_section)
    MPI_Finalize();
#endif

    return 0;
}