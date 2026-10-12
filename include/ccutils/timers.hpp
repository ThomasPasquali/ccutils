#ifndef __CCUTILS_CPU_TIMERS__
#define __CCUTILS_CPU_TIMERS__

#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <limits>
#include <cstdint>
#include <cstddef>
#include <string>
#include "formats.h"
#include "colors.h"

/**********************************************************************/
/*                            NAMESPACE                                */
/**********************************************************************/

namespace ccutils_timers {

  // Quote labels as JSON strings so key/value records remain one line and
  // callers can safely use spaces, quotes, backslashes, or control characters.
  inline std::string quote_field(const char* value) {
    std::string result("\"");
    if (value != nullptr) {
      static const char hex[] = "0123456789abcdef";
      for (const unsigned char ch : std::string(value)) {
        switch (ch) {
          case '"': result += "\\\""; break;
          case '\\': result += "\\\\"; break;
          case '\b': result += "\\b"; break;
          case '\f': result += "\\f"; break;
          case '\n': result += "\\n"; break;
          case '\r': result += "\\r"; break;
          case '\t': result += "\\t"; break;
          default:
            if (ch < 0x20) {
              result += "\\u00";
              result += hex[ch >> 4];
              result += hex[ch & 0x0f];
            } else {
              result += static_cast<char>(ch);
            }
        }
      }
    }
    result += '"';
    return result;
  }
  
  /********************************************************************/
  /*                           TIMER STATS                             */
  /********************************************************************/
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
    if (exclude_first_n >= values.size()) return stats;

    stats.n = values.size() - exclude_first_n;
    if (stats.n == 0) return stats;

    for (std::size_t i = exclude_first_n; i < values.size(); i++) {
      stats.sum += values[i];
      if (values[i] < stats.min) stats.min = values[i];
      if (values[i] > stats.max) stats.max = values[i];
    }

    stats.avg = stats.sum / stats.n;

    if (stats.n > 1) {
      float var = 0.0f;
      for (std::size_t i = exclude_first_n; i < values.size(); i++) {
        float diff = values[i] - stats.avg;
        var += diff * diff;
      }
      stats.stddev = std::sqrt(var / (stats.n - 1));
    }

    return stats;
  }

  /********************************************************************/
  /*                         PRINT FUNCTIONS                           */
  /********************************************************************/

  inline void print_stats(const std::vector<float>& values, const char* name, const char* prefix, uint32_t exclude_first_n=0) {
    TimerStats stats = compute_stats(values, exclude_first_n);

    if (stats.n == 0) {
      return;
    }

    printf(CCUTILS_FMT_TIMER_MULTIPLE,
           quote_field(prefix).c_str(), quote_field(name).c_str(),
           stats.n, stats.avg, stats.stddev,
           stats.min, stats.max, stats.sum);
  }

  inline void print_last_time(const std::vector<float>& values, const char* name, const char* prefix) {
    if (values.empty()) {
      return;
    }
    printf(CCUTILS_FMT_TIMER_SINGLE,
           quote_field(prefix).c_str(), quote_field(name).c_str(), values.back());
  }

  inline void print_all_times(const std::vector<float>& values, const char* name) {
    if (values.empty()) {
      return;
    }
    for (size_t i = 0; i < values.size(); i++) {
      printf(CCUTILS_FMT_TIMER_ALL, quote_field(name).c_str(), i, values[i]);
    }
  }

  inline TimerStats get_timer_stats(const std::vector<float>& values) {
    return compute_stats(values);
  }

} // End of namespace ccutils_timers


/**********************************************************************/
/*                           HELPER MACROS                             */
/**********************************************************************/

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

/**********************************************************************/
/*                         TIMER PRINTING                              */
/**********************************************************************/

#define CCUTILS_TIMER_VALUES(name) __timer_vals_##name

#define CCUTILS_TIMER_STATS(name) \
  ccutils_timers::get_timer_stats(CCUTILS_TIMER_VALUES(name));

#define CCUTILS_TIMER_PRINT(name) \
  ccutils_timers::print_stats(CCUTILS_TIMER_VALUES(name), #name, "Timer");

#define CCUTILS_TIMER_PRINT_LAST(name) \
  ccutils_timers::print_last_time(CCUTILS_TIMER_VALUES(name), #name, "Timer");

#define CCUTILS_TIMER_PRINT_LAST_WPREFIX_STR(name, prefix) \
  ccutils_timers::print_last_time(CCUTILS_TIMER_VALUES(name), #name, prefix);

#define CCUTILS_TIMER_PRINT_ALL(name) \
  ccutils_timers::print_all_times(CCUTILS_TIMER_VALUES(name), #name);

#define CCUTILS_TIMER_PRINT_WPREFIX(name, prefix) \
  ccutils_timers::print_stats(CCUTILS_TIMER_VALUES(name), #name, #prefix);

#define CCUTILS_TIMER_PRINT_LAST_WPREFIX(name, prefix) \
  ccutils_timers::print_last_time(CCUTILS_TIMER_VALUES(name), #name, #prefix);

#define CCUTILS_TIMER_PRINT_EXCLUDING_FIRST_N(name, nexclude) \
  ccutils_timers::print_stats(CCUTILS_TIMER_VALUES(name), #name, "Timer", nexclude);

#define CCUTILS_TIMER_PRINT_WPREFIX_STR(name, prefix) \
  ccutils_timers::print_stats(CCUTILS_TIMER_VALUES(name), #name, prefix);

/**********************************************************************/
/*                         TIMER OPERATIONS                             */
/**********************************************************************/

#define CCUTILS_TIMER_SUM(...) \
  ([&]() -> float { \
    float __sum = 0.0f; \
    std::vector<std::vector<float>*> __vecs = {__MAP(__TIMER_PTR, __VA_ARGS__)}; \
    for (auto* v : __vecs) for (float t : *v) __sum += t; \
    return __sum; \
  }())

#define CCUTILS_TIMER_SUM_AVG(...) \
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

#define CCUTILS_TIMER_SUM_LAST(...) \
  ([&]() -> float { \
    float __sum = 0.0f; \
    std::vector<std::vector<float>*> __vecs = {__MAP(__TIMER_PTR, __VA_ARGS__)}; \
    for (auto* v : __vecs) if (!v->empty()) __sum += v->back(); \
    return __sum; \
  }())

#define CCUTILS_TIMER_HAS_VALUES(...) \
  ([&]() -> bool { \
    std::vector<std::vector<float>*> __timer_vectors = {__MAP(__TIMER_PTR, __VA_ARGS__)}; \
    for (const auto* __timer_vector : __timer_vectors) \
      if (!__timer_vector->empty()) return true; \
    return false; \
  }())

#define CCUTILS_TIMER_SUM_PRINT(label, ...) \
  do { if (CCUTILS_TIMER_HAS_VALUES(__VA_ARGS__)) \
    printf(CCUTILS_FMT_TIMER_SUM, "sum", ccutils_timers::quote_field(#label).c_str(), CCUTILS_TIMER_SUM(__VA_ARGS__)); \
  } while (0);

#define CCUTILS_TIMER_SUM_AVG_PRINT(label, ...) \
  do { if (CCUTILS_TIMER_HAS_VALUES(__VA_ARGS__)) \
    printf(CCUTILS_FMT_TIMER_SUM, "sum_avg", ccutils_timers::quote_field(#label).c_str(), CCUTILS_TIMER_SUM_AVG(__VA_ARGS__)); \
  } while (0);

#define CCUTILS_TIMER_SUM_LAST_PRINT(label, ...) \
  do { if (CCUTILS_TIMER_HAS_VALUES(__VA_ARGS__)) \
    printf(CCUTILS_FMT_TIMER_SUM, "sum_last", ccutils_timers::quote_field(#label).c_str(), CCUTILS_TIMER_SUM_LAST(__VA_ARGS__)); \
  } while (0);


/**********************************************************************/
/*                           TIMER DEFINITIONS                          */
/**********************************************************************/

#define CCUTILS_CPU_TIMER_DEF(name) \
  std::chrono::steady_clock::time_point __timer_start_##name, __timer_stop_##name; \
  std::vector<float> __timer_vals_##name;

#define CCUTILS_CPU_TIMER_IMPORT(name)                                   \
  extern std::chrono::steady_clock::time_point __timer_start_##name; \
  extern std::chrono::steady_clock::time_point __timer_stop_##name;  \
  extern std::vector<float> __timer_vals_##name;

#define CCUTILS_CPU_TIMER_START(name) \
  __timer_start_##name = std::chrono::steady_clock::now();

// Time is reported in milliseconds
#define CCUTILS_CPU_TIMER_STOP(name) \
  do { \
    __timer_stop_##name = std::chrono::steady_clock::now(); \
    float __elapsed_##name = std::chrono::duration<float>(__timer_stop_##name - __timer_start_##name).count() * 1e3f; \
    __timer_vals_##name.push_back(__elapsed_##name); \
  } while (0);

#define CCUTILS_CPU_TIMER_STATS(name, avg, stddev) \
  do { \
    auto stats_##name = ccutils_timers::compute_stats(__timer_vals_##name); \
    avg = stats_##name.avg; \
    stddev = stats_##name.stddev; \
  } while (0);

#define CCUTILS_CPU_TIMER_INIT(name) CCUTILS_CPU_TIMER_DEF(name) CCUTILS_CPU_TIMER_START(name)
#define CCUTILS_CPU_TIMER_CLOSE(name) CCUTILS_CPU_TIMER_STOP(name) CCUTILS_TIMER_PRINT(name)

#endif
