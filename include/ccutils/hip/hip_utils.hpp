#ifndef __CCUTILS_HIP_UTILS__
#define __CCUTILS_HIP_UTILS__

#ifndef CCUTILS_ENABLE_HIP
#error "ccutils HIP headers require -DCCUTILS_ENABLE_HIP"
#endif

#include <hip/hip_runtime.h>
#include <stdint.h>
#include "hip_macros.hpp"
#include "../macros.hpp"

/**********************************************************************/
/*                       DEVICE → HOST COPY                              */
/**********************************************************************/

template <typename T>
inline T * d2h_copy(T * d_buf, uint64_t n) {
  if (n == 0) return nullptr;

  CCUTILS_ASSERT((d_buf != nullptr), "Tried to copy from device nullptr to host\n");

  T * h_buf = (T*)malloc(sizeof(T) * n);
  CCUTILS_HIP_CHECK(hipMemcpy(h_buf, d_buf, sizeof(T) * n, hipMemcpyDeviceToHost));

  return h_buf;
}

template <typename T>
inline void d2h_copy(T * h_buf, uint64_t n, T * d_buf, hipStream_t stream = 0) {
  if (n == 0) return;

  CCUTILS_ASSERT((d_buf != nullptr), "Tried to copy from device nullptr to host\n");
  CCUTILS_ASSERT((h_buf != nullptr), "Tried to copy from device to host nullptr\n");

  if (stream != 0) {
    CCUTILS_HIP_CHECK(hipMemcpyAsync(h_buf, d_buf, sizeof(T) * n, hipMemcpyDeviceToHost, stream));
  } else {
    CCUTILS_HIP_CHECK(hipMemcpy(h_buf, d_buf, sizeof(T) * n, hipMemcpyDeviceToHost));
  }
}

/**********************************************************************/
/*                       HOST → DEVICE COPY                              */
/**********************************************************************/

template <typename T>
inline T * h2d_copy(T * h_buf, uint64_t n) {
  if (n == 0) return nullptr;

  CCUTILS_ASSERT((h_buf != nullptr), "Tried to copy from host nullptr\n");

  T * d_buf;
  CCUTILS_HIP_CHECK(hipMalloc(&d_buf, sizeof(T) * n));
  CCUTILS_HIP_CHECK(hipMemcpy(d_buf, h_buf, sizeof(T) * n, hipMemcpyHostToDevice));

  return d_buf;
}

template <typename T>
inline void h2d_copy(T * d_buf, uint64_t n, T * h_buf, hipStream_t stream = 0) {
  if (n == 0) return;

  CCUTILS_ASSERT((d_buf != nullptr), "Tried to copy from host to device nullptr\n");
  CCUTILS_ASSERT((h_buf != nullptr), "Tried to copy from host nullptr\n");

  if (stream != 0) {
    CCUTILS_HIP_CHECK(hipMemcpyAsync(d_buf, h_buf, sizeof(T) * n, hipMemcpyHostToDevice, stream));
  } else {
    CCUTILS_HIP_CHECK(hipMemcpy(d_buf, h_buf, sizeof(T) * n, hipMemcpyHostToDevice));
  }
}

/**********************************************************************/
/*                       DEVICE → DEVICE COPY                            */
/**********************************************************************/

template <typename T>
inline void d2d_copy(T * d_dst, T * d_src, uint64_t n) {
  if (n == 0) return;
  if (d_dst == nullptr) CCUTILS_HIP_CHECK(hipMalloc(&d_dst, sizeof(T) * n));

  CCUTILS_HIP_CHECK(hipMemcpy(d_dst, d_src, sizeof(T) * n, hipMemcpyDeviceToDevice));
}

template <typename T>
inline void d2d_copy(T ** d_dst, T * d_src, uint64_t n) {
  if (n == 0) return;
  if (*d_dst == nullptr) CCUTILS_HIP_CHECK(hipMalloc(d_dst, sizeof(T) * n));

  CCUTILS_HIP_CHECK(hipMemcpy(*d_dst, d_src, sizeof(T) * n, hipMemcpyDeviceToDevice));
}

#endif // __CCUTILS_HIP_UTILS__
