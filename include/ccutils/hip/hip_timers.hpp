#ifndef __CCUTILS_HIP_TIMERS__
#define __CCUTILS_HIP_TIMERS__

#ifndef CCUTILS_ENABLE_HIP
#error "ccutils HIP headers require -DCCUTILS_ENABLE_HIP"
#endif

#include <hip/hip_runtime.h>
#include "hip_macros.hpp"
#include "../timers.hpp"

/**********************************************************************/
/*                          HIP TIMER DEFINITION                       */
/**********************************************************************/

#define CCUTILS_HIP_TIMER_DEF(name) \
  hipEvent_t __timer_start_##name, __timer_stop_##name; \
  std::vector<float> __timer_vals_##name; \
  hipStream_t __timer_stream_##name = 0; \
  CCUTILS_CHECK_HIP(hipEventCreate(&__timer_start_##name)); \
  CCUTILS_CHECK_HIP(hipEventCreate(&__timer_stop_##name));

/**********************************************************************/
/*                          HIP TIMER OPERATIONS                       */
/**********************************************************************/

#define CCUTILS_HIP_TIMER_START(name, stream) \
  do { \
    __timer_stream_##name = stream; \
    CCUTILS_CHECK_HIP(hipEventRecord(__timer_start_##name, stream)); \
  } while (0);

#define CCUTILS_HIP_TIMER_START_DEFAULT(name) CCUTILS_HIP_TIMER_START(name, 0)

#define CCUTILS_HIP_TIMER_STOP(name) \
  do { \
    CCUTILS_CHECK_HIP(hipEventRecord(__timer_stop_##name, __timer_stream_##name)); \
    CCUTILS_CHECK_HIP(hipEventSynchronize(__timer_stop_##name)); \
    float __elapsed_##name = 0.0f; \
    CCUTILS_CHECK_HIP(hipEventElapsedTime(&__elapsed_##name, __timer_start_##name, __timer_stop_##name)); \
    __timer_vals_##name.push_back(__elapsed_##name); \
  } while (0);

/**********************************************************************/
/*                          HIP TIMER CLEANUP                          */
/**********************************************************************/

#define CCUTILS_HIP_TIMER_DESTROY(name) \
  CCUTILS_CHECK_HIP(hipEventDestroy(__timer_start_##name)); \
  CCUTILS_CHECK_HIP(hipEventDestroy(__timer_stop_##name)); \
  __timer_vals_##name.clear();

/**********************************************************************/
/*                          HIP TIMER SHORTCUTS                        */
/**********************************************************************/

#define HIP_TIMER_INIT(name) CCUTILS_HIP_TIMER_DEF(name) CCUTILS_HIP_TIMER_START_DEFAULT(name)

#define HIP_TIMER_CLOSE(name) CCUTILS_HIP_TIMER_STOP(name) CCUTILS_TIMER_PRINT(name) CCUTILS_HIP_TIMER_DESTROY(name)

#endif
