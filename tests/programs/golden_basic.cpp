// Golden-output test program: non-MPI API (sections, JSON, CPU timers).
// Timing numbers are normalised by the test runner; everything else must stay
// byte-identical across refactors while CCUTILS_FORMAT is "legacy".
#include <cstdio>
#include <vector>
#include <ccutils/ccutils.hpp>

static void busy() {
  volatile unsigned long x = 0;
  for (int i = 0; i < 2000; ++i) x += i;
}

int main() {
  CCUTILS_SECTION_DEF(solve, "Solve phase");
  CCUTILS_SECTION_JSON_PUT(solve, "residual", 1.5e-9);
  CCUTILS_SECTION_JSON_SUB_PUT(solve, "mesh", "cells", 100);
  CCUTILS_SECTION_JSON_PUT(solve, "many", std::vector<int>(60, 12345));  // wide: exercises pretty printing

  CCUTILS_SUBSECTION_DEF(solve, spmv, "Sparse mat-vec");
  CCUTILS_SUBSECTION_JSON_PUT(solve, spmv, "flops", 12345);
  CCUTILS_SUBSECTION_JSON_SUB_PUT(solve, spmv, "peak", "gflops", 2.5);

  CCUTILS_CPU_TIMER_DEF(kernel);
  CCUTILS_CPU_TIMER_DEF(other);
  for (int i = 0; i < 5; ++i) {
    CCUTILS_CPU_TIMER_START(kernel);
    busy();
    CCUTILS_CPU_TIMER_STOP(kernel);
  }
  for (int i = 0; i < 3; ++i) {
    CCUTILS_CPU_TIMER_START(other);
    busy();
    CCUTILS_CPU_TIMER_STOP(other);
  }
  CCUTILS_TIMER_PRINT(kernel);
  CCUTILS_TIMER_PRINT_LAST(kernel);
  CCUTILS_TIMER_PRINT_ALL(other);
  CCUTILS_TIMER_PRINT_EXCLUDING_FIRST_N(kernel, 2);
  CCUTILS_TIMER_PRINT_WPREFIX_STR(kernel, "My \"quoted\" label");
  CCUTILS_TIMER_PRINT_WPREFIX(other, Custom);
  CCUTILS_TIMER_SUM_PRINT(total, kernel, other);
  CCUTILS_TIMER_SUM_AVG_PRINT(total_avg, kernel, other);
  CCUTILS_TIMER_SUM_LAST_PRINT(total_last, kernel, other);
  CCUTILS_SUBSECTION_END(solve, spmv);

  CCUTILS_JSON_DEF(extra);
  CCUTILS_JSON_PUT(extra, "a", 1);
  CCUTILS_JSON_SUB_PUT(extra, "b", "c", "text");
  CCUTILS_JSON_PRINT(extra);

  CCUTILS_SECTION_TIMER_DEF(solve, inner);
  CCUTILS_SECTION_TIMER_START(solve, inner);
  busy();
  CCUTILS_SECTION_TIMER_STOP(solve, inner);
  CCUTILS_SECTION_TIMER_PRINT(solve, inner);

  CCUTILS_CPU_TIMER_INIT(shortcut);
  busy();
  CCUTILS_CPU_TIMER_CLOSE(shortcut);

  CCUTILS_SECTION_END(solve);
  CCUTILS_PRINT_SPLIT("done");
  return 0;
}
