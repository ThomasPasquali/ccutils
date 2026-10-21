// Multi-file test: main() creates a section and timers; solver.cpp and report.cpp
// update them from other translation units.
#include <ccutils/ccutils.hpp>
#include "phases.hpp"

int main() {
  CCUTILS_SECTION_DEF(run, "Whole run");
  CCUTILS_SECTION_JSON_PUT(run, "from_main", 1);

  CCUTILS_CPU_TIMER_DEF(kernel);     // updated from solver.cpp
  CCUTILS_CPU_TIMER_DEF(total);
  CCUTILS_CPU_TIMER_START(total);

  const double residual = solve_phase(4);
  (void)residual; // Also used when CCUTILS_NO_JSON compiles JSON macros away.
  report_phase();

  CCUTILS_CPU_TIMER_STOP(total);
  CCUTILS_TIMER_PRINT(kernel);       // n=5: 4 samples from solver.cpp + 1 from report.cpp
  CCUTILS_TIMER_PRINT(total);
  CCUTILS_SECTION_JSON_PUT(run, "residual_seen_by_main", residual);
  CCUTILS_SECTION_END(run);
  return 0;
}
