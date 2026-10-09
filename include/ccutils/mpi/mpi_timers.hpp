#ifndef __CCUTILS_MPI_TIMERS__
#define __CCUTILS_MPI_TIMERS__

#ifndef CCUTILS_ENABLE_MPI
#error "ccutils MPI headers require -DCCUTILS_ENABLE_MPI"
#endif

#include <ccutils/timers.hpp>
#include <ccutils/formats.h>
#include <mpi.h>
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <stdexcept>


// Records one or more local intervals. print_stats() reduces every recorded
// sample across the supplied communicator and prints aggregate statistics on
// the selected root. All communicator members must call print_stats together.
class TimerStats {
 public:
  void start() {
    if (running_) throw std::logic_error("MPI timer is already running");
    start_seconds_ = MPI_Wtime();
    running_ = true;
  }

  void stop() {
    if (!running_) throw std::logic_error("MPI timer is not running");
    samples_seconds_.push_back(MPI_Wtime() - start_seconds_);
    running_ = false;
  }

  void clear() {
    if (running_) throw std::logic_error("cannot clear a running MPI timer");
    samples_seconds_.clear();
  }

  std::size_t local_sample_count() const { return samples_seconds_.size(); }

  void print_stats(const char* label, MPI_Comm comm, int root = 0) const {
    std::uint64_t local_count = samples_seconds_.size(), total_count = 0;
    double local_sum_ms = 0.0, local_sum_sq_ms = 0.0;
    double local_min_ms = std::numeric_limits<double>::infinity();
    double local_max_ms = -std::numeric_limits<double>::infinity();
    for (double seconds : samples_seconds_) {
      const double ms = seconds * 1000.0;
      local_sum_ms += ms;
      local_sum_sq_ms += ms * ms;
      local_min_ms = std::min(local_min_ms, ms);
      local_max_ms = std::max(local_max_ms, ms);
    }

    double global_min_ms = 0.0, global_max_ms = 0.0;
    double global_sum_ms = 0.0, global_sum_sq_ms = 0.0;
    MPI_Reduce(&local_count, &total_count, 1, MPI_UINT64_T, MPI_SUM, root, comm);
    MPI_Reduce(&local_min_ms, &global_min_ms, 1, MPI_DOUBLE, MPI_MIN, root, comm);
    MPI_Reduce(&local_max_ms, &global_max_ms, 1, MPI_DOUBLE, MPI_MAX, root, comm);
    MPI_Reduce(&local_sum_ms, &global_sum_ms, 1, MPI_DOUBLE, MPI_SUM, root, comm);
    MPI_Reduce(&local_sum_sq_ms, &global_sum_sq_ms, 1, MPI_DOUBLE, MPI_SUM, root, comm);

    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank != root) return;
    if (total_count == 0) {
      std::printf(CCUTILS_FMT_TIMER_WARN, "MPI Timer", label);
      return;
    }

    const double n = static_cast<double>(total_count);
    const double average_ms = global_sum_ms / n;
    const double variance = total_count > 1
                                ? std::max(0.0, (global_sum_sq_ms -
                                                      global_sum_ms * global_sum_ms / n) /
                                                     (n - 1.0))
                                : 0.0;
    std::printf(CCUTILS_FMT_TIMER_MULTIPLE, "MPI Timer", label,
                static_cast<std::size_t>(total_count), average_ms,
                std::sqrt(variance), global_min_ms, global_max_ms,
                global_sum_ms);
    std::fflush(stdout);
  }

 private:
  double start_seconds_ = 0.0;
  bool running_ = false;
  std::vector<double> samples_seconds_;
};


/**********************************************************************/
/*                           MPI TIMER DEF                            */
/**********************************************************************/

#define CCUTILS_MPI_TIMER_DEF(name) TimerStats name

/**********************************************************************/
/*                          MPI TIMER OPERATIONS                      */
/**********************************************************************/

#define CCUTILS_MPI_TIMER_START(name) \
  do { (name).start(); } while (0)

#define CCUTILS_MPI_TIMER_STOP(name) \
  do { (name).stop(); } while (0)

#define CCUTILS_MPI_TIMER_DESTROY(name) \
  do { (name).clear(); } while (0)

#define CCUTILS_MPI_TIMER_PRINT(name, label, comm) \
  do { (name).print_stats((label), (comm)); } while (0)

#define CCUTILS_MPI_TIMER_PRINT_ROOT(name, label, comm, root) \
  do { (name).print_stats((label), (comm), (root)); } while (0)

/**********************************************************************/
/*                          MPI TIMER SHORTCUTS                       */
/**********************************************************************/

#define CCUTILS_MPI_TIMER_INIT(name) \
    CCUTILS_MPI_TIMER_DEF(name)      \
    CCUTILS_MPI_TIMER_START(name)

#define CCUTILS_MPI_TIMER_CLOSE(name)   \
    CCUTILS_MPI_TIMER_STOP(name)        \
    CCUTILS_MPI_TIMER_PRINT_ROOT(name)  \
    CCUTILS_MPI_TIMER_DESTROY(name)               


// #define CCUTILS_MPI_TIMEIT(name, statement) \
//     CCUTILS_MPI_TIMER_INIT(name) \
//     statement; \
//     CCUTILS_MPI_TIMER_CLOSE(name) \

#endif
