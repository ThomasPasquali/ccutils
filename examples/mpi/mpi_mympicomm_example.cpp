#include <mpi.h>
#include <cstdio>

#include <ccutils/mpi/mpi_mympicomm.h>
#include <ccutils/mpi/mpi_macros.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // --- Test MyMpiComm wrapping MPI_COMM_WORLD ---
    CcutilsMpiComm world;
    world.init(MPI_COMM_WORLD);
    world.print_info("world", stdout);

    // --- Test MyMpiComm wrapping MPI_COMM_NULL ---
    CcutilsMpiComm null_comm;
    null_comm.init(MPI_COMM_NULL);
    MPI_PRINT_ONCE("null_comm: rank=%d size=%d (expected -1 -1)\n",
                   null_comm.rank, null_comm.size);

    // --- Split into two sub-communicators and wrap each ---
    int color = world.rank % 2;
    MPI_Comm sub;
    MPI_Comm_split(MPI_COMM_WORLD, color, world.rank, &sub);

    CcutilsMpiComm subcomm;
    subcomm.init(sub);
    subcomm.print_info(color == 0 ? "even" : "odd", stdout);

    MPI_Comm_free(&sub);
    MPI_Finalize();
    return 0;
}
