#include <mpi.h>
#include <cstdio>

#include <ccutils/mpi/mpi_vcollauxbuff.h>
#include <ccutils/mpi/mpi_macros.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    CcutilsMpiComm comm;
    comm.init(MPI_COMM_WORLD);

    // --- ALL2ALLV: each rank sends a different amount to every other rank ---
    MPI_PRINT_ONCE("\n=== ALL2ALLV ===\n");
    {
        CcutilsVcollectiveAuxiliaryBuffers aux(comm, 0);
        int send_sizes[comm.size];
        for (int i = 0; i < comm.size; i++)
            send_sizes[i] = comm.rank + i + 1;
        aux.alltoallv_init(send_sizes);
        aux.print();
    }

    // --- ALLGATHERV: each rank sends rank+1 elements ---
    MPI_PRINT_ONCE("\n=== ALLGATHERV ===\n");
    {
        CcutilsVcollectiveAuxiliaryBuffers aux(comm, 0);
        aux.allgatherv_init(comm.rank + 1);
        aux.print();
    }

    // --- GATHERV: each rank sends rank+1 elements to root ---
    MPI_PRINT_ONCE("\n=== GATHERV (root=0) ===\n");
    {
        CcutilsVcollectiveAuxiliaryBuffers aux(comm, 0);
        aux.gatherv_init(comm.rank + 1);
        aux.print();
    }

    // --- SCATTERV: each rank receives rank+1 elements from root ---
    MPI_PRINT_ONCE("\n=== SCATTERV (root=0) ===\n");
    {
        CcutilsVcollectiveAuxiliaryBuffers aux(comm, 0);
        aux.scatterv_init(comm.rank + 1);
        aux.print();
    }

    MPI_Finalize();
    return 0;
}
