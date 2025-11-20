#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <ccutils/mpi/mpi_timers.h>
#include <ccutils/mpi/mpi_macros.h>

// Example function to simulate CPU work
void simulate_work(int iterations) {
    volatile double sum = 0.0;
    for (int i = 0; i < iterations; ++i) {
        sum += i * 0.000001;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    MPI_PRINT_ONCE("MPI Test Program starting with %d ranks.\n", nprocs);

    // --- Define timers ---
    MPI_TIMER_DEF(work1);
    MPI_TIMER_DEF(work2);

    // --- Timer 1: repeated deterministic work ---
    for (int run = 0; run < 5; run++) { // 5 runs with no-warmup
        MPI_TIMER_START(work1);

        simulate_work(1000000);

        MPI_TIMER_STOP(work1);

        MPI_PRINT_ONCE("[Run %d] rank 0 finished iteration\n", run);
    }

    // --- Close timers (print + destroy) ---
    MPI_TIMER_CLOSE(work1);

    // --- Test other macros ---
    MPI_ALL_PRINT({
        printf("[Rank %d] MPI_ALL_PRINT test message\n", rank);
    });

    MPI_PROCESS_PRINT(MPI_COMM_WORLD, 1, {
        printf("Only rank 1 prints this message.\n");
    });

    MPI_Finalize();
    return 0;
}
