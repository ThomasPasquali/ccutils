#ifndef CCUTILS_TIMERS_HPP
#define CCUTILS_TIMERS_HPP

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
#include "emitter.hpp"
#include "machine.hpp"
#include "registry.hpp"
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
    double avg    = 0.0;
    double stddev = 0.0;
    double min    = std::numeric_limits<double>::infinity();
    double max    = -std::numeric_limits<double>::infinity();
    double sum    = 0.0;
    size_t n     = 0;
  };

  inline TimerStats compute_stats(const std::vector<double>& values, uint32_t exclude_first_n=0) {
    TimerStats stats;
    if (exclude_first_n >= values.size()) return stats;

    stats.n = values.size() - exclude_first_n;
    if (stats.n == 0) return stats;

    double sum = 0.0, correction = 0.0;
    for (std::size_t i = exclude_first_n; i < values.size(); i++) {
      const double value = values[i];
      const double next = sum + value;
      correction += std::fabs(sum) >= std::fabs(value) ? (sum - next) + value : (value - next) + sum;
      sum = next;
      if (values[i] < stats.min) stats.min = values[i];
      if (values[i] > stats.max) stats.max = values[i];
    }
    stats.sum = sum + correction;

    stats.avg = stats.sum / stats.n;

    if (stats.n > 1) {
      double var = 0.0, var_correction = 0.0;
      for (std::size_t i = exclude_first_n; i < values.size(); i++) {
        const double diff = values[i] - stats.avg;
        const double square = diff * diff;
        const double next = var + square;
        var_correction += std::fabs(var) >= std::fabs(square) ? (var - next) + square : (square - next) + var;
        var = next;
      }
      stats.stddev = std::sqrt((var + var_correction) / (stats.n - 1));
    }

    return stats;
  }

  /********************************************************************/
  /*                         PRINT FUNCTIONS                           */
  /********************************************************************/

  inline void print_stats(const std::vector<double>& values, const char* name, const char* prefix, uint32_t exclude_first_n=0) {
    TimerStats stats = compute_stats(values, exclude_first_n);

    if (stats.n == 0) {
      return;
    }

    if (ccutils::human_v2()) {
      ccutils::human_v2_timer("", name, stats.n, stats.avg, stats.stddev,
                              stats.min, stats.max, stats.sum);
    } else {
      ccutils::human_printf(CCUTILS_FMT_TIMER_MULTIPLE,
             quote_field(prefix).c_str(), quote_field(name).c_str(),
             stats.n, stats.avg, stats.stddev,
             stats.min, stats.max, stats.sum);
    }
    ccutils::machine::timer_summary(prefix, name, stats.n, stats.avg, stats.stddev,
                                    stats.min, stats.max, stats.sum);
  }

  inline void print_last_time(const std::vector<double>& values, const char* name, const char* prefix) {
    if (values.empty()) {
      return;
    }
    if (ccutils::human_v2()) ccutils::human_v2_sample(name, values.size()-1, values.back());
    else ccutils::human_printf(CCUTILS_FMT_TIMER_SINGLE,
           quote_field(prefix).c_str(), quote_field(name).c_str(), values.back());
    ccutils::machine::timer_sample(prefix, name, values.size() - 1, values.back(), true);
  }

  inline void print_all_times(const std::vector<double>& values, const char* name) {
    if (values.empty()) {
      return;
    }
    for (size_t i = 0; i < values.size(); i++) {
      if (ccutils::human_v2()) ccutils::human_v2_sample(name, i, values[i]);
      else ccutils::human_printf(CCUTILS_FMT_TIMER_ALL, quote_field(name).c_str(), i, values[i]);
      ccutils::machine::timer_sample("Timer", name, i, values[i], false);
    }
  }

  inline TimerStats get_timer_stats(const std::vector<double>& values) {
    return compute_stats(values);
  }

} // End of namespace ccutils_timers

/* ---- Timers addressable by name ------------------------------------------------
 *
 * CCUTILS_CPU_TIMER_DEF registers the timer it declares under its name for as long
 * as the variables live. Another function (or file) reaches the SAME timer with
 *
 *     CCUTILS_CPU_TIMER_USE(kernel);              // then START/STOP/PRINT work as usual
 *     ccutils::TimerRef t = ccutils::timer("kernel");
 *     CCUTILS_TIMER_SCOPE(kernel);                // times the enclosing scope
 *
 * If no timer with that name is alive, a persistent one is created. */
namespace ccutils {
namespace detail {

struct CpuTimerData {
  std::chrono::steady_clock::time_point* start = nullptr;
  std::chrono::steady_clock::time_point* stop = nullptr;
  std::vector<double>* vals = nullptr;
};

struct CpuTimerStorage {
  std::chrono::steady_clock::time_point start, stop;
  std::vector<double> vals;
};

typedef NamedRegistry<CpuTimerData> CpuTimerRegistry;

class CpuTimerRegistration {
 public:
  CpuTimerRegistration(const char* name, std::chrono::steady_clock::time_point* start,
                       std::chrono::steady_clock::time_point* stop, std::vector<double>* vals)
      : registration_(name, vals, make(start, stop, vals)) {}

 private:
  static CpuTimerData make(std::chrono::steady_clock::time_point* start,
                           std::chrono::steady_clock::time_point* stop, std::vector<double>* vals) {
    CpuTimerData data;
    data.start = start;
    data.stop = stop;
    data.vals = vals;
    return data;
  }
  Registration<CpuTimerData> registration_;
};

/* The live timer called `name`, or a new persistent one. */
inline CpuTimerData cpu_timer_data(const std::string& name) {
  CpuTimerData data;
  if (CpuTimerRegistry::instance().find(name, &data)) return data;
  std::shared_ptr<CpuTimerStorage> storage = std::make_shared<CpuTimerStorage>();
  data.start = &storage->start;
  data.stop = &storage->stop;
  data.vals = &storage->vals;
  CpuTimerRegistry::instance().add(name, storage.get(), data, storage);
  return data;
}

}  // namespace detail

class TimerRef {
 public:
  TimerRef(const std::string& name, const detail::CpuTimerData& data) : name_(name), data_(data) {}

  void start() { *data_.start = std::chrono::steady_clock::now(); }
  /* Records the time since start() in milliseconds. */
  void stop() {
    *data_.stop = std::chrono::steady_clock::now();
    data_.vals->push_back(std::chrono::duration<double>(*data_.stop - *data_.start).count() * 1e3);
  }
  void clear() { data_.vals->clear(); }

  std::size_t count() const { return data_.vals->size(); }
  double last() const { return data_.vals->empty() ? 0.0 : data_.vals->back(); }
  const std::vector<double>& values() const { return *data_.vals; }
  const std::string& name() const { return name_; }

  ccutils_timers::TimerStats stats(uint32_t exclude_first_n = 0) const {
    return ccutils_timers::compute_stats(*data_.vals, exclude_first_n);
  }
  void print(const char* prefix = "Timer", uint32_t exclude_first_n = 0) const {
    ccutils_timers::print_stats(*data_.vals, name_.c_str(), prefix, exclude_first_n);
  }
  void print_last(const char* prefix = "Timer") const {
    ccutils_timers::print_last_time(*data_.vals, name_.c_str(), prefix);
  }
  void print_all() const { ccutils_timers::print_all_times(*data_.vals, name_.c_str()); }

 private:
  std::string name_;
  detail::CpuTimerData data_;
};

inline TimerRef timer(const std::string& name) { return TimerRef(name, detail::cpu_timer_data(name)); }

/* Starts the named timer now and records a sample when the object goes out of scope. */
class ScopedTimer {
 public:
  explicit ScopedTimer(const std::string& name) : timer_(timer(name)) { timer_.start(); }
  ~ScopedTimer() { timer_.stop(); }
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  TimerRef timer_;
};

}  // namespace ccutils


/**********************************************************************/
/*                           HELPER MACROS                             */
/**********************************************************************/

#define CCUTILS_DETAIL_TIMER_PTR(name) &ccutils_timer_vals_##name
#define CCUTILS_DETAIL_EXPAND(x) x
#define CCUTILS_DETAIL_MAP_1(m, x) m(x)
#define CCUTILS_DETAIL_MAP_2(m, x, ...) m(x), CCUTILS_DETAIL_MAP_1(m, __VA_ARGS__)
#define CCUTILS_DETAIL_MAP_3(m, x, ...) m(x), CCUTILS_DETAIL_MAP_2(m, __VA_ARGS__)
#define CCUTILS_DETAIL_MAP_4(m, x, ...) m(x), CCUTILS_DETAIL_MAP_3(m, __VA_ARGS__)
#define CCUTILS_DETAIL_MAP_5(m, x, ...) m(x), CCUTILS_DETAIL_MAP_4(m, __VA_ARGS__)
#define CCUTILS_DETAIL_MAP_6(m, x, ...) m(x), CCUTILS_DETAIL_MAP_5(m, __VA_ARGS__)
#define CCUTILS_DETAIL_MAP_7(m, x, ...) m(x), CCUTILS_DETAIL_MAP_6(m, __VA_ARGS__)
#define CCUTILS_DETAIL_MAP_8(m, x, ...) m(x), CCUTILS_DETAIL_MAP_7(m, __VA_ARGS__)
#define CCUTILS_DETAIL_MAP_9(m, x, ...) m(x), CCUTILS_DETAIL_MAP_8(m, __VA_ARGS__)
#define CCUTILS_DETAIL_MAP_10(m, x, ...) m(x), CCUTILS_DETAIL_MAP_9(m, __VA_ARGS__)
#define CCUTILS_DETAIL_GET_11TH_ARG(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,N,...) N
#define CCUTILS_DETAIL_MAP_CHOOSER(...) \
  CCUTILS_DETAIL_EXPAND(CCUTILS_DETAIL_GET_11TH_ARG(__VA_ARGS__, CCUTILS_DETAIL_MAP_10, CCUTILS_DETAIL_MAP_9, CCUTILS_DETAIL_MAP_8, CCUTILS_DETAIL_MAP_7, CCUTILS_DETAIL_MAP_6, CCUTILS_DETAIL_MAP_5, CCUTILS_DETAIL_MAP_4, CCUTILS_DETAIL_MAP_3, CCUTILS_DETAIL_MAP_2, CCUTILS_DETAIL_MAP_1))
#define CCUTILS_DETAIL_MAP(m, ...) CCUTILS_DETAIL_EXPAND(CCUTILS_DETAIL_MAP_CHOOSER(__VA_ARGS__)(m, __VA_ARGS__))

/**********************************************************************/
/*                         TIMER PRINTING                              */
/**********************************************************************/

#define CCUTILS_TIMER_VALUES(name) ccutils_timer_vals_##name

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
  ([&]() -> double { \
    double ccutils_sum_ = 0.0; \
    std::vector<std::vector<double>*> ccutils_vecs_ = {CCUTILS_DETAIL_MAP(CCUTILS_DETAIL_TIMER_PTR, __VA_ARGS__)}; \
    for (auto* v : ccutils_vecs_) for (double t : *v) ccutils_sum_ += t; \
    return ccutils_sum_; \
  }())

#define CCUTILS_TIMER_SUM_AVG(...) \
  ([&]() -> double { \
    double ccutils_sum_ = 0.0; \
    std::vector<std::vector<double>*> ccutils_vecs_ = {CCUTILS_DETAIL_MAP(CCUTILS_DETAIL_TIMER_PTR, __VA_ARGS__)}; \
    for (auto* v : ccutils_vecs_) { \
      size_t ccutils_n_ = v->size(); \
      if (ccutils_n_ > 0) { \
        double ccutils_avg_ = 0.0; \
        for (double t : *v) ccutils_avg_ += t; \
        ccutils_sum_ += ccutils_avg_ / ccutils_n_; \
      } \
    } \
    return ccutils_sum_; \
  }())

#define CCUTILS_TIMER_SUM_LAST(...) \
  ([&]() -> double { \
    double ccutils_sum_ = 0.0; \
    std::vector<std::vector<double>*> ccutils_vecs_ = {CCUTILS_DETAIL_MAP(CCUTILS_DETAIL_TIMER_PTR, __VA_ARGS__)}; \
    for (auto* v : ccutils_vecs_) if (!v->empty()) ccutils_sum_ += v->back(); \
    return ccutils_sum_; \
  }())

#define CCUTILS_TIMER_HAS_VALUES(...) \
  ([&]() -> bool { \
    std::vector<std::vector<double>*> ccutils_timer_vectors_ = {CCUTILS_DETAIL_MAP(CCUTILS_DETAIL_TIMER_PTR, __VA_ARGS__)}; \
    for (const auto* ccutils_timer_vector_ : ccutils_timer_vectors_) \
      if (!ccutils_timer_vector_->empty()) return true; \
    return false; \
  }())

#define CCUTILS_TIMER_HAS_VALUES(...) \
  ([&]() -> bool { \
    std::vector<std::vector<float>*> __timer_vectors = {__MAP(__TIMER_PTR, __VA_ARGS__)}; \
    for (const auto* __timer_vector : __timer_vectors) \
      if (!__timer_vector->empty()) return true; \
    return false; \
  }())

#define CCUTILS_TIMER_SUM_PRINT(label, ...) \
  do { if (CCUTILS_TIMER_HAS_VALUES(__VA_ARGS__)) { \
    const double ccutils_agg_ = (CCUTILS_TIMER_SUM(__VA_ARGS__)); \
    ccutils::human_printf(CCUTILS_FMT_TIMER_SUM, "sum", ccutils_timers::quote_field(#label).c_str(), ccutils_agg_); \
    ccutils::machine::timer_agg(#label, "sum", ccutils_agg_); \
  } } while (0);

#define CCUTILS_TIMER_SUM_AVG_PRINT(label, ...) \
  do { if (CCUTILS_TIMER_HAS_VALUES(__VA_ARGS__)) { \
    const double ccutils_agg_ = (CCUTILS_TIMER_SUM_AVG(__VA_ARGS__)); \
    ccutils::human_printf(CCUTILS_FMT_TIMER_SUM, "sum_avg", ccutils_timers::quote_field(#label).c_str(), ccutils_agg_); \
    ccutils::machine::timer_agg(#label, "sum_avg", ccutils_agg_); \
  } } while (0);

#define CCUTILS_TIMER_SUM_LAST_PRINT(label, ...) \
  do { if (CCUTILS_TIMER_HAS_VALUES(__VA_ARGS__)) { \
    const double ccutils_agg_ = (CCUTILS_TIMER_SUM_LAST(__VA_ARGS__)); \
    ccutils::human_printf(CCUTILS_FMT_TIMER_SUM, "sum_last", ccutils_timers::quote_field(#label).c_str(), ccutils_agg_); \
    ccutils::machine::timer_agg(#label, "sum_last", ccutils_agg_); \
  } } while (0);


/**********************************************************************/
/*                           TIMER DEFINITIONS                          */
/**********************************************************************/

#define CCUTILS_CPU_TIMER_DEF(name) \
  std::chrono::steady_clock::time_point ccutils_timer_start_##name, ccutils_timer_stop_##name; \
  std::vector<double> ccutils_timer_vals_##name; \
  ccutils::detail::CpuTimerRegistration ccutils_timer_reg_##name( \
      #name, &ccutils_timer_start_##name, &ccutils_timer_stop_##name, &ccutils_timer_vals_##name);

/* Binds the timer called `name` (defined elsewhere with CCUTILS_CPU_TIMER_DEF) to
 * local references, so the usual START/STOP/PRINT macros work on it in this scope. */
#define CCUTILS_CPU_TIMER_USE(name) \
  const ccutils::detail::CpuTimerData ccutils_timer_use_##name = ccutils::detail::cpu_timer_data(#name); \
  std::chrono::steady_clock::time_point& ccutils_timer_start_##name = *ccutils_timer_use_##name.start; \
  std::chrono::steady_clock::time_point& ccutils_timer_stop_##name = *ccutils_timer_use_##name.stop; \
  std::vector<double>& ccutils_timer_vals_##name = *ccutils_timer_use_##name.vals; \
  (void)ccutils_timer_start_##name; (void)ccutils_timer_stop_##name; (void)ccutils_timer_vals_##name;

#define CCUTILS_TIMER_SCOPE(name) ccutils::ScopedTimer ccutils_scoped_timer_##name(#name);

#define CCUTILS_CPU_TIMER_IMPORT(name)                                   \
  extern std::chrono::steady_clock::time_point ccutils_timer_start_##name; \
  extern std::chrono::steady_clock::time_point ccutils_timer_stop_##name;  \
  extern std::vector<double> ccutils_timer_vals_##name;

#define CCUTILS_CPU_TIMER_START(name) \
  ccutils_timer_start_##name = std::chrono::steady_clock::now();

// Time is reported in milliseconds
#define CCUTILS_CPU_TIMER_STOP(name) \
  do { \
    ccutils_timer_stop_##name = std::chrono::steady_clock::now(); \
    double ccutils_elapsed_##name = std::chrono::duration<double>(ccutils_timer_stop_##name - ccutils_timer_start_##name).count() * 1e3; \
    ccutils_timer_vals_##name.push_back(ccutils_elapsed_##name); \
  } while (0);

#define CCUTILS_CPU_TIMER_STATS(name, avg, stddev) \
  do { \
    auto stats_##name = ccutils_timers::compute_stats(ccutils_timer_vals_##name); \
    avg = stats_##name.avg; \
    stddev = stats_##name.stddev; \
  } while (0);

#define CCUTILS_CPU_TIMER_INIT(name) CCUTILS_CPU_TIMER_DEF(name) CCUTILS_CPU_TIMER_START(name)
#define CCUTILS_CPU_TIMER_CLOSE(name) CCUTILS_CPU_TIMER_STOP(name) CCUTILS_TIMER_PRINT(name)

#endif
