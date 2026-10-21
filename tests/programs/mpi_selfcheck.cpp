// Prints the MPI world size seen by rank 0. Used by the test runner to verify
// that the configured mpicxx/mpirun pair really launches N cooperating ranks
// (a mismatched pair silently starts N independent singleton worlds).
#include <mpi.h>
#include <cstdio>
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0) std::printf("world_size=%d\n", size);
  MPI_Finalize();
  return 0;
}
