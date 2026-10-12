#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <ccutils/mpi/mpi_timers.hpp>
#include <ccutils/mpi/mpi_macros.hpp>

// Example function to simulate CPU work
void simulate_work(int iterations) {
    volatile double sum = 0.0;
    for (int i = 0; i < iterations; ++i) {
        sum += i * 0.000001;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    CCUTILS_MPI_INIT

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    CCUTILS_MPI_PRINTF_ONCE("MPI Test Program starting with %d ranks.\n", nprocs);

    // --- Define timers ---
    CCUTILS_MPI_TIMER_DEF(work1);
    CCUTILS_MPI_TIMER_DEF(work2);

    // --- Timer 1: repeated deterministic work ---
    for (int run = 0; run < 5; run++) { // 5 runs with no-warmup
        CCUTILS_MPI_TIMER_START(work1);

        simulate_work(1000000);

        CCUTILS_MPI_TIMER_STOP(work1);

        CCUTILS_MPI_PRINTF_ONCE("[Run %d] rank 0 finished iteration\n", run);
    }

    // --- Close timers (print + destroy) ---
    CCUTILS_MPI_TIMER_CLOSE(work1);

    // --- Print macros ---
    CCUTILS_MPI_ALL_PRINT(
        fprintf(fp, "MPI_ALL_PRINT test message from rank %d\n", rank);
    );

    CCUTILS_MPI_PROCESS_PRINT(MPI_COMM_WORLD, 1, 
        printf("Only rank 1 prints this message.\n");
    );

    if (rank == 0 || rank == 1) {
        CCUTILS_MPI_BUFFERED_PRINT_ADD(
            fprintf(fp, "First message from rank %d\n", rank);
        )

        CCUTILS_MPI_BUFFERED_PRINT_ADD(
            fprintf(fp, "Second message from rank %d\n", rank);
        )
    }
    CCUTILS_MPI_BUFFERED_PRINT_ALL_FLUSH

    // Remove any leftover tmp file (not needed here)
    CCUTILS_MPI_CLEANUP

    MPI_Finalize();
    return 0;
}
