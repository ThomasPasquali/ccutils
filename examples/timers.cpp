#include <ccutils/timers.hpp>
#include <ccutils/macros.hpp>
#ifdef WITH_MPI
#include <ccutils/mpi/mpi_timers.hpp>
#include <mpi.h>
#include <unistd.h>
#endif
#include <cstdio>

int main()
{
#ifdef WITH_MPI
  MPI_Init(NULL, NULL);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
  {
#endif
    // Define two timers: one for repeated sum runs, one for single random work
    CCUTILS_CPU_TIMER_DEF(work1)
    CCUTILS_CPU_TIMER_DEF(work2)

    // --- First timer: sum over multiple runs ---
    for (size_t run_i = 0; run_i < 2 + 10; run_i++)
    { // 2 warmup + 10 actual runs
      CCUTILS_CPU_TIMER_START(work1)

      // Simulated CPU work
      volatile double sum_val = 0.0;
      for (size_t i = 0; i < 10'000'000; ++i)
      {
        sum_val += i * 0.000001;
      }

      CCUTILS_CPU_TIMER_STOP(work1)
      printf("[%sRun %2lu] sum = %.1f\n", run_i >= 2 ? "" : "Warmup ", run_i, sum_val);
    }

    // --- Second timer: one-shot random work ---
    CCUTILS_CPU_TIMER_START(work2)

    // Simulated other work
    volatile double work_val = 0.0;
    for (size_t i = 0; i < 5'000'000; ++i)
    {
      work_val += rand() % 100;
    }

    CCUTILS_CPU_TIMER_STOP(work2);
    printf("[Work2] result = %.1f\n\n", work_val);

    // --- Print timers ---
    CCUTILS_SECTION_DEF(work_1, "Work 1 Timers")
    CCUTILS_TIMER_PRINT(work1)                         // mean/std/min/max
    CCUTILS_TIMER_PRINT_WPREFIX(work1, Work1)          // same stats with a custom prefix
    CCUTILS_TIMER_PRINT_EXCLUDING_FIRST_N(work1, 2)    // this will exclude the 2 warm-up runs
    CCUTILS_TIMER_PRINT_LAST(work1)                    // last recorded run of `work1`
    CCUTILS_TIMER_PRINT_LAST_WPREFIX(work1, LatsWork1) // same but with a custom prefix
    CCUTILS_TIMER_PRINT_ALL(work1)                     // every `work1` run individually
    CCUTILS_SECTION_END(work_1)

    CCUTILS_SECTION_DEF(work_2, "Work 2 Timers")
    CCUTILS_TIMER_PRINT(work2)      // since work2 has only one record,
    CCUTILS_TIMER_PRINT_LAST(work2) // TIMER_PRINT and TIMER_PRINT_LAST do the same
    CCUTILS_SECTION_END(work_2)

    CCUTILS_SECTION_DEF(empty, "Empty Timer")
    CCUTILS_CPU_TIMER_DEF(work3)
    CCUTILS_TIMER_PRINT(work3) // this will print a warning
    CCUTILS_SECTION_END(empty)

    // --- Combine timers ---
    // These macros can sum over multiple timers in one go
    // These work with up to 10 timers
    CCUTILS_SECTION_DEF(aggregated, "Aggregated Timers")
    CCUTILS_TIMER_SUM_PRINT(Combined, work1, work2);     // total elapsed of work1 + work2
    CCUTILS_TIMER_SUM_AVG_PRINT(Combined, work1, work2)  // sum of per-timer averages
    CCUTILS_TIMER_SUM_LAST_PRINT(Combined, work1, work2) // last run of each, added
    CCUTILS_SECTION_END(aggregated)

// Specialized timers for MPI (they use MPI_WTime)
#ifdef WITH_MPI
    }
    CCUTILS_SECTION_DEF(mpi_timers, "MPI Timers")

    CCUTILS_MPI_TIMER_INIT(mpi_timer)
    usleep(1000);
    CCUTILS_MPI_TIMER_STOP(mpi_timer)

    // You can use the unified macros to print it
    CCUTILS_TIMER_PRINT(mpi_timer)

    CCUTILS_SECTION_END(mpi_timers)

    // Handy shortcut
    // CCUTILS_MPI_TIMEIT(one_liner_timer, volatile double sum_val = 0.0; for (size_t i = 0; i < 10'000'000; ++i) { sum_val += i * 0.000001; })

    MPI_Finalize();
#endif

  // CUDA is analogous (they use CUDA events)

  return 0;
}
