#include <ccutils/ccutils.hpp>
#include "phases.hpp"

static void tiny_work() {
  volatile unsigned long x = 0;
  for (int i = 0; i < 1000; ++i) x += i;
}

double solve_phase(int iterations) {
  CCUTILS_CPU_TIMER_USE(kernel);                       // the timer main() defined
  CCUTILS_SUBSECTION_DEF(run, solve, "Solve");         // opens inside main's section
  for (int i = 0; i < iterations; ++i) {
    CCUTILS_CPU_TIMER_START(kernel);
    tiny_work();
    CCUTILS_CPU_TIMER_STOP(kernel);
  }
  CCUTILS_SECTION_JSON_PUT(run, "iterations", iterations);            // main's section, by name
  CCUTILS_SUBSECTION_JSON_PUT(run, solve, "residual", 0.25);
  ccutils::section("run").put("note", "set from solver.cpp");         // object API
  CCUTILS_SUBSECTION_END(run, solve);
  return 0.25;
}
