#include <ccutils/ccutils.hpp>
#include "phases.hpp"

void report_phase() {
  // Object API: no macro, no local variable.
  ccutils::TimerRef kernel = ccutils::timer("kernel");
  kernel.start();
  kernel.stop();
  CCUTILS_SECTION_JSON_SUB_PUT(run, "report", "kernel_samples", static_cast<int>(kernel.count()));
}
