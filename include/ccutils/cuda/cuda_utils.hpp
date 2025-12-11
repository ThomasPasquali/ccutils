#ifndef __CCUTILS_CUDA_UTILS__
#define __CCUTILS_CUDA_UTILS__

#ifndef CCUTILS_ENABLE_CUDA
#error "ccutils CUDA headers require -DCCUTILS_ENABLE_CUDA"
#endif

#include <cuda_runtime.h>
#include <stdint.h>
#include "cuda_macros.hpp"
#include "../macros.hpp"

/**********************************************************************/
/*                       DEVICE → HOST COPY                              */
/**********************************************************************/

template <typename T>
inline T * d2h_copy(T * d_buf, uint64_t n) {
  if (n==0) return nullptr;

  CCUTILS_ASSERT((d_buf != nullptr), "Tried to copy from device nullptr to host\n");

  T * h_buf = (T*)malloc(sizeof(T)*n); 
  CCUTILS_CUDA_CHECK(cudaMemcpy(h_buf, d_buf, sizeof(T)*n, cudaMemcpyDeviceToHost));

  return h_buf;
}

template <typename T>
inline void d2h_copy(T * h_buf, uint64_t n, T * d_buf, cudaStream_t stream = 0) {
  if (n==0) return;

  CCUTILS_ASSERT((d_buf != nullptr), "Tried to copy from device nullptr to host\n");
  CCUTILS_ASSERT((h_buf != nullptr), "Tried to copy from device to host nullptr\n");

  if (stream != 0) {
    CCUTILS_CUDA_CHECK(cudaMemcpyAsync(h_buf, d_buf, sizeof(T)*n, cudaMemcpyDeviceToHost, stream));
  } else {
    CCUTILS_CUDA_CHECK(cudaMemcpy(h_buf, d_buf, sizeof(T)*n, cudaMemcpyDeviceToHost));
  }
}

/**********************************************************************/
/*                       HOST → DEVICE COPY                              */
/**********************************************************************/

template <typename T>
inline T * h2d_copy(T * h_buf, uint64_t n) {
  if (n==0) return nullptr;

  CCUTILS_ASSERT((h_buf != nullptr), "Tried to copy from host nullptr\n");

  T * d_buf;
  CCUTILS_CUDA_CHECK(cudaMalloc(&d_buf, sizeof(T)*n));
  CCUTILS_CUDA_CHECK(cudaMemcpy(d_buf, h_buf, sizeof(T)*n, cudaMemcpyHostToDevice));

  return d_buf;
}

template <typename T>
inline void h2d_copy(T * d_buf, uint64_t n, T * h_buf, cudaStream_t stream = 0) {
  if (n==0) return;

  CCUTILS_ASSERT((d_buf != nullptr), "Tried to copy from host to device nullptr\n");
  CCUTILS_ASSERT((h_buf != nullptr), "Tried to copy from host nullptr\n");

  if (stream != 0) {
    CCUTILS_CUDA_CHECK(cudaMemcpyAsync(d_buf, h_buf, sizeof(T)*n, cudaMemcpyHostToDevice, stream));
  } else {
    CCUTILS_CUDA_CHECK(cudaMemcpy(d_buf, h_buf, sizeof(T)*n, cudaMemcpyHostToDevice));
  }
}

/**********************************************************************/
/*                       DEVICE → DEVICE COPY                            */
/**********************************************************************/

template <typename T>
inline void d2d_copy(T * d_dst, T * d_src, uint64_t n) {
  if (n==0)           return;
  if (d_dst==nullptr) CCUTILS_CUDA_CHECK(cudaMalloc(&d_dst, sizeof(T)*n));

  CCUTILS_CUDA_CHECK(cudaMemcpy(d_dst, d_src, sizeof(T)*n, cudaMemcpyDeviceToDevice));
}

template <typename T>
inline void d2d_copy(T ** d_dst, T * d_src, uint64_t n) {
  if (n==0)           return;
  if (*d_dst==nullptr) CCUTILS_CUDA_CHECK(cudaMalloc(d_dst, sizeof(T)*n));

  CCUTILS_CUDA_CHECK(cudaMemcpy(*d_dst, d_src, sizeof(T)*n, cudaMemcpyDeviceToDevice));
}

#endif
