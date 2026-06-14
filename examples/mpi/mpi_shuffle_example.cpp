#include <mpi.h>
#include <cstdio>

#include <ccutils/mpi/mpi_shuffling.h>
#include <ccutils/mpi/mpi_macros.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    MPI_PRINT_ONCE("MpiShufflingHelper example: %d ranks, 4 indices per rank.\n", nprocs);

    // --- Create shuffler: 4 indices per process ---
    MpiShufflingHelper shuff(MPI_COMM_WORLD, 4);

    // Print the full distributed shuffle
    shuff.print();

    MPI_Finalize();
    return 0;
}
