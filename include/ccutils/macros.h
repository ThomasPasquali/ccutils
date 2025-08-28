#ifndef __CCUTILS_CPU_MACROS__
#define __CCUTILS_CPU_MACROS__

#include "colors.h"

// Math
#define CEILING(x,y) (((x) + (y) - 1) / (y))

// Testing
#define ASSERT(cond, msg, ...) \
  if (!(cond)) { \
    fprintf(stderr, BRIGHT_RED "Assertion in %s on line %i failed\n" RESET, __FILE__, __LINE__);\
    fprintf(stderr, BRIGHT_RED msg RESET, ##__VA_ARGS__);\
    exit(EXIT_FAILURE); \
  }

// Prints
#define PRINT_SPLIT(s) \
  printf("--------------------  %s  --------------------\n", s); \
  fflush(stdout);

#define DEBUG_PRINT(fmt, ...) printf(BRIGHT_CYAN "[DEBUG] " fmt RESET, ##__VA_ARGS__);

#endif