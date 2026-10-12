# CUDA utilities

Configure CCUTILS with `-DCCUTILS_ENABLE_CUDA=ON` and link `ccutils::ccutils_cuda` (which carries the CUDA runtime dependency). CUDA headers are opt-in; include them from CUDA-aware source files.

```cpp
#include <ccutils/cuda/cuda_timers.hpp>

void launch_work(cudaStream_t stream) {
    CCUTILS_CUDA_TIMER_DEF(kernel)
    CCUTILS_CUDA_TIMER_START(kernel, stream)
    my_kernel<<<grid, block, 0, stream>>>();
    CCUTILS_CUDA_CHECK(cudaGetLastError())
    CCUTILS_CUDA_TIMER_STOP(kernel)
    CCUTILS_TIMER_PRINT(kernel)
    CCUTILS_CUDA_TIMER_DESTROY(kernel)
}
```

`CCUTILS_CUDA_CHECK(call)` reports CUDA runtime failures and exits. `CCUTILS_CUDA_FREE_SAFE(ptr)` frees a non-null pointer. `cuda_utils.hpp` provides `h2d_copy`, `d2h_copy`, and `d2d_copy` helpers; returned allocations are owned by the caller. CUDA timers use events and synchronize at stop to obtain elapsed milliseconds.
