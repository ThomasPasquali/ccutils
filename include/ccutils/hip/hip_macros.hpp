#ifndef __CCUTILS_HIP_MACROS__
#define __CCUTILS_HIP_MACROS__

#ifndef CCUTILS_ENABLE_HIP
#error "ccutils HIP headers require -DCCUTILS_ENABLE_HIP"
#endif

#include <stdint.h>
#include "../formats.h"
#include <hip/hip_runtime.h>

/**********************************************************************/
/*                            HIP ERROR CHECK                          */
/**********************************************************************/

#define CCUTILS_HIP_CHECK(call) {                                             \
  hipError_t err = call;                                                     \
  if (err != hipSuccess) {                                                   \
    fprintf(stderr, "HIP error in file '%s' in line %i : %s (%u)\n",        \
            __FILE__, __LINE__, hipGetErrorString(err), err);                \
    exit(err);                                                               \
  }                                                                          \
}
#define CCUTILS_CHECK_HIP(call) CCUTILS_HIP_CHECK(call)

#define CCUTILS_HIP_CHECK_SOFT(call) {                                       \
  hipError_t err = call;                                                     \
  if (err == hipErrorMemoryAllocation) {                                     \
    fprintf(stderr, "HIP OUT OF MEMORY in file '%s' in line %i : %s (%u)\n",\
            __FILE__, __LINE__, hipGetErrorString(err), err);                \
    return false;                                                            \
  }                                                                          \
  if (err != hipSuccess) {                                                   \
    fprintf(stderr, "HIP error in file '%s' in line %i : %s.\n",            \
            __FILE__, __LINE__, hipGetErrorString(err));                     \
    exit(err);                                                               \
  }                                                                          \
}

/**********************************************************************/
/*                            HIP ASSERTIONS                           */
/**********************************************************************/

#define CCUTILS_ASSERT_HIP(cond, msg, ...) \
  if (!(cond)) { \
    printf(BRIGHT_RED "Assertion in %s on line %i failed: " msg RESET, __FILE__, __LINE__, ##__VA_ARGS__); \
    return; \
  }
#define CCUTILS_HIP_ASSERT(call) CCUTILS_ASSERT_HIP(call)

/**********************************************************************/
/*                           HIP MEMORY UTILS                           */
/**********************************************************************/

#define CCUTILS_HIP_FREE_SAFE(buf) do { \
  if (buf != nullptr) hipFree(buf); \
} while (0)

#endif // __CCUTILS_HIP_MACROS__
