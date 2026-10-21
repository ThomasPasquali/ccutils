#include <cstdio>
#include <mpi.h>
#include <ccutils/ccutils.hpp>
#include <ccutils/mpi/mpi_macros.hpp>
#include <ccutils/mpi/mpi_timers.hpp>
#include "worker.hpp"

void do_work(int iterations) {
  CCUTILS_MPI_TIMER_USE(halo);                               // main's timer, by name
  CCUTILS_MPI_JSON_USE(stats);                               // main's JSON object, by name
  CCUTILS_MPI_SUBSECTION_DEF(run, work, "Work");             // inside main's section

  for (int i = 0; i < iterations; ++i) {
    CCUTILS_MPI_TIMER_START(halo);
    CCUTILS_MPI_TIMER_STOP(halo);
  }

  CCUTILS_MPI_GLOBAL_JSON_PUT(run, "iterations", iterations);        // main's section, rank 0 owns it
  CCUTILS_MPI_LOCAL_JSON_PUT(run, "rank", ccutils::mpi_rank());      // main's section, per rank
  CCUTILS_MPI_SUBSECTION_JSON_PUT(run, work, "samples", iterations);
  CCUTILS_MPI_JSON_PUT(stats, "rank", ccutils::mpi_rank());

  CCUTILS_MPI_PRINTF_ONCE("worker ran on %d ranks\n", ccutils::mpi_size());
  CCUTILS_MPI_ALL_PRINT_NAMED(worker, fprintf(fp, "hello from worker, rank %d\n", ccutils::mpi_rank()));

  CCUTILS_MPI_SUBSECTION_END(run, work);
}
