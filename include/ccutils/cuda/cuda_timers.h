#ifndef __CCUTILS_CUDA_TIMERS__
#define __CCUTILS_CUDA_TIMERS__
#ifndef CCUTILS_ENABLE_CUDA
// #error "ccutils CUDA headers require -DCCUTILS_ENABLE_CUDA"
#endif

#include <cuda_runtime.h>
#include "cuda_macros.h"
#include "../timers.h"

#define CUDA_TIMER_DEF(name) \
  cudaEvent_t __timer_start_##name, __timer_stop_##name; \
  std::vector<float> __timer_vals_##name; \
  cudaStream_t __timer_stream_##name = 0; \
  CHECK_CUDA(cudaEventCreate(&__timer_start_##name)); \
  CHECK_CUDA(cudaEventCreate(&__timer_stop_##name));

#define CUDA_TIMER_START(name, stream) \
  do { \
    __timer_stream_##name = stream; \
    CHECK_CUDA(cudaEventRecord(__timer_start_##name, stream)); \
  } while (0);

#define CUDA_TIMER_START_DEFAULT(name) CUDA_TIMER_START(name, 0)

#define CUDA_TIMER_STOP(name) \
  do { \
    CHECK_CUDA(cudaEventRecord(__timer_stop_##name, __timer_stream_##name)); \
    CHECK_CUDA(cudaEventSynchronize(__timer_stop_##name)); \
    float __elapsed_##name = 0.0f; \
    CHECK_CUDA(cudaEventElapsedTime(&__elapsed_##name, __timer_start_##name, __timer_stop_##name)); \
    __timer_vals_##name.push_back(__elapsed_##name); \
  } while (0);

#define CUDA_TIMER_DESTROY(name) \
  CHECK_CUDA(cudaEventDestroy(__timer_start_##name)); \
  CHECK_CUDA(cudaEventDestroy(__timer_stop_##name)); \
  __timer_vals_##name.clear();

#define CUDA_TIMER_INIT(name) CUDA_TIMER_DEF(name) CUDA_TIMER_START_DEFAULT(name)

#define CUDA_TIMER_CLOSE(name) CUDA_TIMER_STOP(name) TIMER_PRINT(name) CUDA_TIMER_DESTROY(name)


#endif
