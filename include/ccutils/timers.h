#ifndef __CCUTILS_CPU_TIMERS__
#define __CCUTILS_CPU_TIMERS__

#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <limits>
#include "formats.h"
#include "colors.h"

namespace ccutils_timers {
  struct TimerStats {
    float avg    = 0.0f;
    float stddev = 0.0f;
    float min    = std::numeric_limits<float>::infinity();
    float max    = -std::numeric_limits<float>::infinity();
    float sum    = 0.0f;
    size_t n     = 0;
  };

  inline TimerStats compute_stats(const std::vector<float>& values, uint32_t exclude_first_n=0) {
    TimerStats stats;
    if (exclude_first_n>=values.size()) return stats;

    stats.n = values.size() - exclude_first_n;
    if (stats.n == 0) return stats; // return default (warning will be printed by print_stats)

    // Compute sum, min, max
    for (int i=exclude_first_n; i<values.size(); i++) {
      stats.sum += values[i];
      if (values[i] < stats.min) stats.min = values[i];
      if (values[i] > stats.max) stats.max = values[i];
    }

    stats.avg = stats.sum / stats.n;

    // Compute stddev if n > 1
    if (stats.n > 1) {
      float var = 0.0f;
      for (int i=exclude_first_n; i<values.size(); i++) {
        float diff = values[i] - stats.avg;
        var += diff * diff;
      }
      stats.stddev = std::sqrt(var / (stats.n - 1));
    }

    return stats;
  }

  inline void print_stats(const std::vector<float>& values, const char* name, const char* prefix, uint32_t exclude_first_n=0) {
    TimerStats stats = compute_stats(values, exclude_first_n);

    if (stats.n == 0) {
      printf(CCUTILS_FMT_TIMER_WARN, prefix, name);
      return;
    }

    if (stats.n == 1) {
      printf(CCUTILS_FMT_TIMER_SINGLE, prefix, name, values.front());
      return;
    }

    printf(CCUTILS_FMT_TIMER_MULTIPLE,
           prefix, name,
           stats.n, stats.avg, stats.stddev,
           stats.min, stats.max, stats.sum);
  }

  inline void print_last_time(const std::vector<float>& values, const char* name, const char* prefix) {
    if (values.empty()) {
      printf(CCUTILS_FMT_TIMER_WARN, prefix, name);
      return;
    }
    printf(CCUTILS_FMT_TIMER_SINGLE, prefix, name, values.back());
  }

  inline void print_all_times(const std::vector<float>& values, const char* name) {
    if (values.empty()) {
      printf(CCUTILS_FMT_TIMER_WARN, "Timer", name);
      return;
    }
    printf(CCUTILS_FMT_TIMER_ALL, name);
    for (size_t i = 0; i < values.size(); i++) {
      printf("Run %2zu: %f ms\n", i, values[i]);
    }
    printf(RESET);
  }

  // The returned TimerStats is stack-allocated
  inline TimerStats get_timer_stats(const std::vector<float>& values) {
    return compute_stats(values);
  }

} // End of namespace "ccutils_timers"

#define __TIMER_PTR(name) &__timer_vals_##name
#define __EXPAND(x) x
#define __MAP_1(m, x) m(x)
#define __MAP_2(m, x, ...) m(x), __MAP_1(m, __VA_ARGS__)
#define __MAP_3(m, x, ...) m(x), __MAP_2(m, __VA_ARGS__)
#define __MAP_4(m, x, ...) m(x), __MAP_3(m, __VA_ARGS__)
#define __MAP_5(m, x, ...) m(x), __MAP_4(m, __VA_ARGS__)
#define __MAP_6(m, x, ...) m(x), __MAP_5(m, __VA_ARGS__)
#define __MAP_7(m, x, ...) m(x), __MAP_6(m, __VA_ARGS__)
#define __MAP_8(m, x, ...) m(x), __MAP_7(m, __VA_ARGS__)
#define __MAP_9(m, x, ...) m(x), __MAP_8(m, __VA_ARGS__)
#define __MAP_10(m, x, ...) m(x), __MAP_9(m, __VA_ARGS__)
#define __GET_11TH_ARG(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,N,...) N
#define __MAP_CHOOSER(...) \
  __EXPAND(__GET_11TH_ARG(__VA_ARGS__, __MAP_10, __MAP_9, __MAP_8, __MAP_7, __MAP_6, __MAP_5, __MAP_4, __MAP_3, __MAP_2, __MAP_1))
#define __MAP(m, ...) __EXPAND(__MAP_CHOOSER(__VA_ARGS__)(m, __VA_ARGS__))

#define TIMER_SUM(...) \
  ([&]() -> float { \
    float __sum = 0.0f; \
    std::vector<std::vector<float>*> __vecs = {__MAP(__TIMER_PTR, __VA_ARGS__)}; \
    for (auto* v : __vecs) { for (float t : *v) __sum += t; } \
    return __sum; \
  }())

#define TIMER_SUM_AVG(...) \
  ([&]() -> float { \
    float __sum = 0.0f; \
    std::vector<std::vector<float>*> __vecs = {__MAP(__TIMER_PTR, __VA_ARGS__)}; \
    for (auto* v : __vecs) { \
      size_t __n = v->size(); \
      if (__n > 0) { \
        float __avg = 0.0f; \
        for (float t : *v) __avg += t; \
        __sum += __avg / __n; \
      } \
    } \
    return __sum; \
  }())

#define TIMER_SUM_LAST(...) \
  ([&]() -> float { \
    float __sum = 0.0f; \
    std::vector<std::vector<float>*> __vecs = {__MAP(__TIMER_PTR, __VA_ARGS__)}; \
    for (auto* v : __vecs) { if (!v->empty()) __sum += v->back(); } \
    return __sum; \
  }())

#define TIMER_SUM_PRINT(label, ...) \
  printf(CCUTILS_FMT_TIMER_SUM, "TimerSum", #label, TIMER_SUM(__VA_ARGS__));

#define TIMER_SUM_AVG_PRINT(label, ...) \
  printf(CCUTILS_FMT_TIMER_SUM, "TimerSumAvg", #label, TIMER_SUM_AVG(__VA_ARGS__));

#define TIMER_SUM_LAST_PRINT(label, ...) \
  printf(CCUTILS_FMT_TIMER_SUM, "TimerSumLast", #label, TIMER_SUM_LAST(__VA_ARGS__));

// The returned TimerStats is stack-allocated
#define TIMER_STATS(name) \
  ccutils_timers::get_timer_stats(__timer_vals_##name);

#define TIMER_PRINT(name) \
  ccutils_timers::print_stats(__timer_vals_##name, #name, "Timer");

#define TIMER_PRINT_LAST(name) \
  ccutils_timers::print_last_time(__timer_vals_##name, #name, "Timer");

#define TIMER_PRINT_ALL(name) \
  ccutils_timers::print_all_times(__timer_vals_##name, #name);

#define TIMER_PRINT_WPREFIX(name, prefix) \
  ccutils_timers::print_stats(__timer_vals_##name, #name, #prefix);

#define TIMER_PRINT_LAST_WPREFIX(name, prefix) \
  ccutils_timers::print_last_time(__timer_vals_##name, #name, #prefix);

#define TIMER_PRINT_EXCLUDING_FIRST_N(name, nexclude) \
  ccutils_timers::print_stats(__timer_vals_##name, #name, "Timer", nexclude);

#define TIMER_PRINT_WPREFIX_STR(name, prefix) \
  ccutils_timers::print_stats(__timer_vals_##name, #name, prefix);

#define CPU_TIMER_DEF(name) \
  std::chrono::high_resolution_clock::time_point __timer_start_##name, __timer_stop_##name; \
  std::vector<float> __timer_vals_##name;

#define CPU_TIMER_START(name) \
  __timer_start_##name = std::chrono::high_resolution_clock::now();

#define CPU_TIMER_STOP(name) \
  do { \
    __timer_stop_##name = std::chrono::high_resolution_clock::now(); \
    float __elapsed_##name = std::chrono::duration<float>(__timer_stop_##name - __timer_start_##name).count() * 1e3f; \
    __timer_vals_##name.push_back(__elapsed_##name); \
  } while (0);

#define CPU_TIMER_STATS(name, avg, stddev) \
  do { \
    size_t n_##name = __timer_vals_##name.size(); \
    float sum_##name = 0.0f; \
    for (float v : __timer_vals_##name) sum_##name += v; \
    avg = sum_##name / n_##name; \
    float var_##name = 0.0f; \
    for (float v : __timer_vals_##name) var_##name += (v - avg) * (v - avg); \
    stddev = (n_##name > 1) ? std::sqrt(var_##name / (n_##name - 1)) : 0.0f; \
  } while (0);

#define CPU_TIMER_INIT(name) CPU_TIMER_DEF(name) CPU_TIMER_START(name)

#define CPU_TIMER_CLOSE(name) CPU_TIMER_STOP(name) TIMER_PRINT(name)


#endif
