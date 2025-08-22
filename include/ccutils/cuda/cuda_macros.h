#ifndef __CCUTILS_CUDA_MACROS__
#define __CCUTILS_CUDA_MACROS__
#ifndef CCUTILS_ENABLE_CUDA
#error "ccutils CUDA headers require -DCCUTILS_ENABLE_CUDA"
#endif

#include <stdint.h>

// CUDA
#define CUDA_CHECK(call) {                                             \
  cudaError_t err = call;                                              \
  if (err != cudaSuccess) {                                            \
    fprintf(stderr, "CUDA error in file '%s' in line %i : %s (%u)\n",  \
            __FILE__, __LINE__, cudaGetErrorString(err), err);         \
    exit(err);                                                         \
  }                                                                    \
}
#define CHECK_CUDA(call) CUDA_CHECK(call)

#define CUDA_CHECK_SOFT(call) {                                                 \
  cudaError_t err = call;                                                       \
  if (err == cudaErrorMemoryAllocation) {                                       \
    fprintf(stderr, "CUDA OUT OF MEMORY in file '%s' in line %i : %s (%u)\n",   \
            __FILE__, __LINE__, cudaGetErrorString(err), err);                  \
    return false;                                                               \
  }                                                                             \
  if (err != cudaSuccess) {                                                     \
    fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",               \
            __FILE__, __LINE__, cudaGetErrorString(err));                       \
    exit(err);                                                                  \
  }                                                                             \
}

#define ASSERT_CUDA(cond, msg, ...) \
  if (!(cond)) { \
    printf(BRIGHT_RED "Assertion in %s on line %i failed: " msg RESET, __FILE__, __LINE__, ##__VA_ARGS__); \
    return; \
  }
#define CUDA_ASSERT(call) ASSERT_CUDA(call)

#define CUDA_FREE_SAFE(buf) do { \
    if (buf != nullptr) cudaFree(buf); \
} while (0)


// TODO CUSPARSE documentation
#if defined(CCUTILS_ENABLE_CUSPARSE) && (CCUTILS_ENABLE_CUSPARSE > 0)
    #define CUSPARSE_CHECK(call) do {                                    \
        cusparseStatus_t err = call;                                     \
        if (err != CUSPARSE_STATUS_SUCCESS) {                            \
            fprintf(stderr, "cuSPARSE error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cusparseGetErrorString(err));    \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while(0)
#endif


// TODO NVTX documentation
#if defined(CCUTILS_ENABLE_NVTX) && (CCUTILS_ENABLE_NVTX > 0)
    #include <nvtx3/nvToolsExt.h>
    // Colors:                  Green,      Blue,       Yellow,     Magenta,    Cyan,       Red,        White
    const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
    const int num_colors = sizeof(colors)/sizeof(uint32_t);
    #define NVTX_PUSH_RANGE(name,cid) { \
        int color_id = cid; \
        color_id = color_id%num_colors;\
        nvtxEventAttributes_t eventAttrib = {0}; \
        eventAttrib.version = NVTX_VERSION; \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
        eventAttrib.colorType = NVTX_COLOR_ARGB; \
        eventAttrib.color = colors[color_id]; \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name; \
        nvtxRangePushEx(&eventAttrib); \
    }
    #define NVTX_POP_RANGE nvtxRangePop();
#else
    #define NVTX_PUSH_RANGE(name,cid)
    #define NVTX_POP_RANGE
#endif


#endif