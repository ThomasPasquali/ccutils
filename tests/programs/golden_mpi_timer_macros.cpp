// Macro hygiene of the MPI timer shortcuts. CCUTILS_MPI_TIMER_CLOSE expands to
// several statements; it must behave as ONE statement after an unbraced `if`.
// Single rank, so a skipped collective cannot deadlock.
#include <mpi.h>
#include <cstdio>
#include <ccutils/ccutils.hpp>
#include <ccutils/mpi/mpi_macros.hpp>
#include <ccutils/mpi/mpi_timers.hpp>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  CCUTILS_MPI_INIT;

  CCUTILS_MPI_TIMER_INIT(t);              // running
  volatile int never = 0;
  if (never) CCUTILS_MPI_TIMER_CLOSE(t, "skipped", MPI_COMM_WORLD);   // must do nothing at all
  CCUTILS_MPI_TIMER_STOP(t);
  CCUTILS_MPI_TIMER_PRINT(t, "after_skip", MPI_COMM_WORLD);
  CCUTILS_MPI_TIMER_DESTROY(t);

  if (ccutils_inmacro_myid == 0) CCUTILS_MPI_TIMER_START(t);          // START/STOP/DESTROY as single statements
  if (ccutils_inmacro_myid == 0) CCUTILS_MPI_TIMER_CLOSE(t, "taken", MPI_COMM_WORLD);

  CCUTILS_MPI_CLEANUP;
  MPI_Finalize();
  return 0;
}
