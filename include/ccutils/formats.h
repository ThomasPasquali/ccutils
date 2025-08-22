
#ifndef __CCUTILS_FORMATS__
#define __CCUTILS_FORMATS__

#include "colors.h"

// Timers string formats
#define CCUTILS_FMT_TIMER_MULTIPLE  BRIGHT_CYAN     "<%s>[%s] n=%zu,avg=%f,stddev=%f,min=%f,max=%f,sum=%f" RESET "\n"
#define CCUTILS_FMT_TIMER_ALL       BRIGHT_CYAN     "<Timer>[%s] All times (ms):\n"
#define CCUTILS_FMT_TIMER_SINGLE    BRIGHT_CYAN     "<%s>[%s] %f ms" RESET "\n"
#define CCUTILS_FMT_TIMER_WARN      BRIGHT_YELLOW   "<%s>[%s] No recorded runs" RESET "\n"
#define CCUTILS_FMT_TIMER_SUM       BRIGHT_MAGENTA  "<%s>[%s] %f ms" RESET "\n"


#endif