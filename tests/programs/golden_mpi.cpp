// Golden-output test program: MPI API (all-print, sections, JSON, MPI timers).
// Every printed line is produced by rank 0, or is separated from rank 0's
// output by CCUTILS_FLUSH_WAIT, so the expected output is deterministic.
#include <mpi.h>
#include <cstdio>
#include <vector>
#include <ccutils/ccutils.hpp>
#include <ccutils/mpi/mpi_macros.hpp>
#include <ccutils/mpi/mpi_timers.hpp>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  CCUTILS_MPI_INIT;
  const int id = ccutils_inmacro_myid;
  const int np = ccutils_inmacro_ntask;

  CCUTILS_MPI_PRINTF_ONCE("== start np=%d ==\n", np);

  /* ---- all-print: unnamed, named, multi-line, empty ranks, no final newline ---- */
  CCUTILS_MPI_ALL_PRINT(fprintf(fp, "hello from %d\n", id));
  CCUTILS_MPI_ALL_PRINT_NAMED(multiline, {
    fprintf(fp, "first line of %d\n\n   indented\n", id);
    if (id % 2) fprintf(fp, "odd rank\n");
  });
  CCUTILS_MPI_ALL_PRINT_NAMED(only_even, { if (id % 2 == 0) fprintf(fp, "even %d\n", id); });
  CCUTILS_MPI_ALL_PRINT_NAMED(no_newline, { fprintf(fp, "tail-less %d", id); });
  CCUTILS_MPI_ALL_PRINT_NAMED(trailing_blank, { fprintf(fp, "text %d\n\n", id); });
  CCUTILS_MPI_ALL_PRINT_NAMED(nobody, { });
  CCUTILS_MPI_ALL_PRINT(if (id == np - 1) fprintf(fp, "only the last rank\n"));

  /* ---- sections, subsections and JSON ---- */
  CCUTILS_MPI_SECTION_DEF(solve, "Solve phase");
#ifndef CCUTILS_NO_JSON
  CCUTILS_MPI_GLOBAL_JSON_PUT(solve, "np", np);
  CCUTILS_MPI_LOCAL_JSON_PUT(solve, "rank", id);
  CCUTILS_MPI_LOCAL_JSON_PUT(solve, "square", id * id);
  CCUTILS_MPI_LOCAL_JSON_PUT(solve, "many", std::vector<int>(50, 100 + id));
#endif
  CCUTILS_MPI_SUBSECTION_DEF(solve, halo, "Halo exchange");
  CCUTILS_MPI_SUBSECTION_JSON_PUT(solve, halo, "bytes", 64 * (id + 1));
  CCUTILS_MPI_SUBSECTION_JSON_SUB_PUT(solve, halo, "peer", "left", id - 1);
  CCUTILS_MPI_SUBSECTION_END(solve, halo);

  CCUTILS_MPI_JSON_DEF(stats);
  CCUTILS_MPI_JSON_PUT(stats, "x", id);
  CCUTILS_MPI_JSON_SUB_PUT(stats, "a", "b", 2 * id);
  CCUTILS_MPI_JSON_PRINT(stats);
  CCUTILS_MPI_SECTION_END(solve);

  /* ---- MPI timers: WORLD, sub-communicator, explicit root, shortcut macros ---- */
  CCUTILS_MPI_TIMER_DEF(t_world);
  for (int i = 0; i < 3; ++i) {
    CCUTILS_MPI_TIMER_START(t_world);
    CCUTILS_MPI_TIMER_STOP(t_world);
  }
  CCUTILS_MPI_TIMER_PRINT(t_world, "world", MPI_COMM_WORLD);

  MPI_Comm sub;
  MPI_Comm_split(MPI_COMM_WORLD, id < 2 ? 0 : MPI_UNDEFINED, id, &sub);
  if (sub != MPI_COMM_NULL) {
    CCUTILS_MPI_TIMER_DEF(t_sub);
    for (int i = 0; i < 2; ++i) {
      CCUTILS_MPI_TIMER_START(t_sub);
      CCUTILS_MPI_TIMER_STOP(t_sub);
    }
    CCUTILS_MPI_TIMER_PRINT(t_sub, "sub", sub);
    MPI_Comm_free(&sub);
  }

  CCUTILS_MPI_TIMER_INIT(t_short);
  CCUTILS_MPI_TIMER_CLOSE(t_short, "short", MPI_COMM_WORLD);

  /* Rank 0's timer lines above are still in its (pipe-buffered) stdout. Flush and wait so
   * they reach the pipe before rank np-1 prints its own report. */
  CCUTILS_FLUSH_WAIT(200000);

  /* the report is printed by rank np-1 (its own stdout); wait before rank 0 prints again */
  CCUTILS_MPI_TIMER_DEF(t_root);
  CCUTILS_MPI_TIMER_START(t_root);
  CCUTILS_MPI_TIMER_STOP(t_root);
  CCUTILS_MPI_TIMER_PRINT_ROOT(t_root, "root_last", MPI_COMM_WORLD, np - 1);
  CCUTILS_FLUSH_WAIT(200000);

  CCUTILS_MPI_PROCESS_PRINT(MPI_COMM_WORLD, np - 1, fprintf(stdout, "process print %d\n", id));
  CCUTILS_FLUSH_WAIT(200000);
  MPI_Barrier(MPI_COMM_WORLD);

  CCUTILS_MPI_PRINTF_ONCE("== end ==\n");
  CCUTILS_MPI_CLEANUP;
  MPI_Finalize();
  return 0;
}
