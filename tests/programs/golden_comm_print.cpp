// CCUTILS_MPI_COMMUNICATOR_PRINT with a long communicator name. The macro
// builds a temporary file name from the name, which used to overflow a
// fixed 50-byte buffer. Run with a single rank so the output is race-free.
#include <mpi.h>
#include <cstdio>
#include <ccutils/ccutils.hpp>
#include <ccutils/mpi/mpi_macros.hpp>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  CCUTILS_MPI_INIT;

  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  MPI_Comm_set_name(comm, "communicator_with_a_deliberately_long_descriptive_name_01");
  CCUTILS_MPI_COMMUNICATOR_PRINT(comm, fprintf(fp, "world rank %d speaking\n", ccutils_inmacro_myid));
  MPI_Comm_free(&comm);

  CCUTILS_MPI_CLEANUP;
  MPI_Finalize();
  return 0;
}
