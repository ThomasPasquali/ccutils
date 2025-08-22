#include <ccutils/timers.h>
#include <ccutils/macros.h>
#include <cstdio>

// Example: using timer macros with multiple runs and summary printing
int main() {
  // Define two timers: one for repeated sum runs, one for single random work
  CPU_TIMER_DEF(work1);
  CPU_TIMER_DEF(work2);


  // --- First timer: sum over multiple runs ---
  for (size_t run_i = 0; run_i < 2 + 10; run_i++) { // 2 warmup + 10 actual runs
    if (run_i >= 2) CPU_TIMER_START(work1);

    // Simulated CPU work (deterministic)
    volatile double sum_val = 0.0;
    for (int i = 0; i < 10'000'000; ++i) {
      sum_val += i * 0.000001;
    }

    if (run_i >= 2) CPU_TIMER_STOP(work1);
    printf("[%sRun %2lu] sum = %.1f\n", run_i >= 2 ? "" : "Warmup ", run_i, sum_val);
  }


  // --- Second timer: one-shot random work ---
  CPU_TIMER_START(work2);

  // Simulated other work
  volatile double work_val = 0.0;
  for (int i = 0; i < 5'000'000; ++i) {
    work_val += rand() % 100; 
  }

  CPU_TIMER_STOP(work2);
  printf("[Work2] result = %.1f\n\n", work_val);


  // --- Print timers ---
  // Print aggregate statistics using utility macros
  PRINT_SPLIT("Work 1 Timers")
  TIMER_PRINT(work1);                           // mean/std/min/max
  TIMER_PRINT_WPREFIX(work1, Work1);            // same stats with a custom prefix
  TIMER_PRINT_LAST(work1);                      // last recorded run of `work1`
  TIMER_PRINT_LAST_WPREFIX(work1, LatsWork1);   // same but with a custom prefix
  TIMER_PRINT_ALL(work1);                       // every `work1` run individually

  PRINT_SPLIT("Work 2 Timer")
  TIMER_PRINT(work2);                           // since work2 has only one record,
  TIMER_PRINT_LAST(work2);                      // TIMER_PRINT and TIMER_PRINT_LAST do the same

  PRINT_SPLIT("Empty Timer")
  CPU_TIMER_DEF(work3);
  TIMER_PRINT(work3);                           // this will print a warning


  // --- Combine timers ---
  // These macros can sum over multiple timers in one go
  PRINT_SPLIT("Combining Different Timers")
  // These work with up to 10 timers
  TIMER_SUM_PRINT(Combined, work1, work2);       // total elapsed of work1 + work2
  TIMER_SUM_AVG_PRINT(Combined, work1, work2);   // sum of per-timer averages
  TIMER_SUM_LAST_PRINT(Combined, work1, work2);  // last run of each, added

  return 0;
}
