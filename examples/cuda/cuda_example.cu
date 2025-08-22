#include <ccutils/timers.h>
#include <ccutils/cuda/cuda_timers.h>
#include <ccutils/cuda/cuda_utils.hpp>
#include <cstdio>

__global__ void add_kernel(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  const int N = 1 << 20; // 1M floats
  const size_t size = N * sizeof(float);

  // host memory
  std::vector<float> h_a(N, 1.0f);
  std::vector<float> h_b(N, 2.0f);

  // device memory
  float *d_a = h2d_copy(h_a.data(), N);
  float *d_b = h2d_copy(h_b.data(), N);
  float *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_c, size));

  // timing
  CUDA_TIMER_DEF(add);

  // kernel config
  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  // run kernel multiple times
  for (int iter = 0; iter < 10; iter++) {
    CUDA_TIMER_START(add, 0);
    add_kernel<<<grid, block>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CUDA_TIMER_STOP(add);
  }

  // copy back result
  float *h_c = d2h_copy(d_c, N);

  // check correctness
  bool ok = true;
  for (int i = 0; i < N; i++) {
    if (h_c[i] != 3.0f) { ok = false; break; }
  }
  printf("Result check: %s\n", ok ? "PASSED" : "FAILED");

  // print stats
  ccutils_timers::TimerStats stats = TIMER_STATS(add)
  printf("<CUDA>[add_kernel] min=%.3f, max=%.3f\n", stats.min, stats.max);
  TIMER_PRINT(add)

  // cleanup
  CUDA_TIMER_DESTROY(add);
  CUDA_FREE_SAFE(d_a);
  CUDA_FREE_SAFE(d_b);
  CUDA_FREE_SAFE(d_c);
  free(h_c);

  return 0;
}