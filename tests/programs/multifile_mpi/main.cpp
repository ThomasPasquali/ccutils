// MPI multi-file test: main() owns the section, an MPI timer and a JSON object;
// worker.cpp updates them without CCUTILS_MPI_INIT and without any shared variable.
#include <mpi.h>
#include <ccutils/ccutils.hpp>
#include <ccutils/mpi/mpi_macros.hpp>
#include <ccutils/mpi/mpi_timers.hpp>
#include "worker.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  CCUTILS_MPI_SECTION_DEF(run, "Whole run");
  CCUTILS_MPI_TIMER_DEF(halo);          // started and stopped in worker.cpp
  CCUTILS_MPI_JSON_DEF(stats);          // filled in worker.cpp

  do_work(3);

  CCUTILS_MPI_JSON_PRINT(stats);
  CCUTILS_MPI_TIMER_PRINT(halo, "halo", MPI_COMM_WORLD);
  CCUTILS_MPI_SECTION_END(run);

  CCUTILS_MPI_CLEANUP;
  MPI_Finalize();
  return 0;
}
