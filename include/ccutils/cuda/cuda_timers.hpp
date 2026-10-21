#ifndef CCUTILS_CUDA_TIMERS_HPP
#define CCUTILS_CUDA_TIMERS_HPP

// TODO(P0-cuda): these headers got the mechanical __identifier rename but were NEVER COMPILED (no nvcc).
//   Compile them, then add registration by name + a USE macro like the CPU timers.
//   Details: docs/TODO.md  (### P0-cuda)
#ifndef CCUTILS_ENABLE_CUDA
// #error "ccutils CUDA headers require -DCCUTILS_ENABLE_CUDA"
#endif

#include <cuda_runtime.h>
#include "cuda_macros.hpp"
#include "../timers.hpp"

/**********************************************************************/
/*                          CUDA TIMER DEFINITION                       */
/**********************************************************************/

#define CCUTILS_CUDA_TIMER_DEF(name) \
  cudaEvent_t ccutils_timer_start_##name, ccutils_timer_stop_##name; \
  std::vector<double> ccutils_timer_vals_##name; \
  cudaStream_t ccutils_timer_stream_##name = 0; \
  CCUTILS_CHECK_CUDA(cudaEventCreate(&ccutils_timer_start_##name)); \
  CCUTILS_CHECK_CUDA(cudaEventCreate(&ccutils_timer_stop_##name));

/**********************************************************************/
/*                          CUDA TIMER OPERATIONS                       */
/**********************************************************************/

#define CCUTILS_CUDA_TIMER_START(name, stream) \
  do { \
    ccutils_timer_stream_##name = stream; \
    CCUTILS_CHECK_CUDA(cudaEventRecord(ccutils_timer_start_##name, stream)); \
  } while (0);

#define CCUTILS_CUDA_TIMER_START_DEFAULT(name) CCUTILS_CUDA_TIMER_START(name, 0)

// Time is reported in milliseconds
#define CCUTILS_CUDA_TIMER_STOP(name) \
  do { \
    CCUTILS_CHECK_CUDA(cudaEventRecord(ccutils_timer_stop_##name, ccutils_timer_stream_##name)); \
    CCUTILS_CHECK_CUDA(cudaEventSynchronize(ccutils_timer_stop_##name)); \
    float ccutils_elapsed_##name = 0.0f; \
    CCUTILS_CHECK_CUDA(cudaEventElapsedTime(&ccutils_elapsed_##name, ccutils_timer_start_##name, ccutils_timer_stop_##name)); \
    ccutils_timer_vals_##name.push_back(ccutils_elapsed_##name); \
  } while (0);

/**********************************************************************/
/*                          CUDA TIMER CLEANUP                          */
/**********************************************************************/

#define CCUTILS_CUDA_TIMER_DESTROY(name) \
  CCUTILS_CHECK_CUDA(cudaEventDestroy(ccutils_timer_start_##name)); \
  CCUTILS_CHECK_CUDA(cudaEventDestroy(ccutils_timer_stop_##name)); \
  ccutils_timer_vals_##name.clear();

/**********************************************************************/
/*                          CUDA TIMER SHORTCUTS                        */
/**********************************************************************/

#define CUDA_TIMER_INIT(name) CCUTILS_CUDA_TIMER_DEF(name) CCUTILS_CUDA_TIMER_START_DEFAULT(name)

#define CUDA_TIMER_CLOSE(name) CCUTILS_CUDA_TIMER_STOP(name) CCUTILS_TIMER_PRINT(name) CCUTILS_CUDA_TIMER_DESTROY(name)

#endif
