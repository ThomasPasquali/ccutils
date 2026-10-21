#ifndef CCUTILS_MPI_TIMERS_HPP
#define CCUTILS_MPI_TIMERS_HPP

#ifndef CCUTILS_ENABLE_MPI
#error "ccutils MPI headers require -DCCUTILS_ENABLE_MPI"
#endif

#include "../timers.hpp"
#include "../formats.h"
#include "../emitter.hpp"
#include "../registry.hpp"
#include "mpi_info.hpp"
#include <mpi.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <climits>
#include <limits>
#include <stdexcept>


// Records one or more local intervals. print_stats() reduces every recorded
// sample across the supplied communicator and prints aggregate statistics on
// the selected root. All communicator members must call print_stats together.
namespace ccutils {

// Legacy reports keep the original five-reduction path. V2 reports use one
// stable-statistics gather, with optional rank details, warm-up exclusion, samples.
class MpiTimer {
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

  void print_stats(const char* label, MPI_Comm comm, int root = 0,
                   std::size_t exclude_first_n = 0, bool include_samples = false) const {
    if (ccutils::human_v2()) {
      print_stats_v2(label, comm, root, exclude_first_n, include_samples);
      return;
    }
    const std::size_t begin = std::min(exclude_first_n, samples_seconds_.size());
    std::uint64_t local_count = samples_seconds_.size() - begin, total_count = 0;
    double local_sum_ms = 0.0, local_sum_sq_ms = 0.0;
    double local_min_ms = std::numeric_limits<double>::infinity();
    double local_max_ms = -std::numeric_limits<double>::infinity();
    for (std::size_t i = begin; i < samples_seconds_.size(); ++i) {
      const double seconds = samples_seconds_[i];
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

    if (include_samples) gather_samples(label, comm, root, begin);

    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    ccutils::set_world_rank(world_rank);

    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank != root) return;
    if (total_count == 0) return;

    const double n = static_cast<double>(total_count);
    const double average_ms = global_sum_ms / n;
    const double variance = total_count > 1
                                ? std::max(0.0, (global_sum_sq_ms -
                                                      global_sum_ms * global_sum_ms / n) /
                                                     (n - 1.0))
                                : 0.0;
    if (ccutils::human_v2()) {
      ccutils::human_v2_timer("", label, static_cast<std::size_t>(total_count), average_ms,
                              std::sqrt(variance), global_min_ms, global_max_ms,
                              global_sum_ms, "comm MPI_COMM_WORLD, 1 ranks");
    } else {
      ccutils::human_printf(CCUTILS_FMT_TIMER_MULTIPLE,
                  ccutils_timers::quote_field("MPI Timer").c_str(),
                  ccutils_timers::quote_field(label).c_str(),
                  static_cast<std::size_t>(total_count), average_ms,
                  std::sqrt(variance), global_min_ms, global_max_ms,
                  global_sum_ms);
    }
    ccutils::human_flush();
    if (ccutils::machine_enabled()) {
      ccutils::capture_mpi_facts();
      char comm_name[MPI_MAX_OBJECT_NAME + 1] = "";
      int name_length = 0, comm_size = 0;
      MPI_Comm_get_name(comm, comm_name, &name_length);
      comm_name[name_length] = '\0';
      MPI_Comm_size(comm, &comm_size);
      ccutils::machine::timer_comm("MPI Timer", label, comm_name, comm_size, world_rank,
                                   static_cast<std::size_t>(total_count), average_ms,
                                   std::sqrt(variance), global_min_ms, global_max_ms,
                                   global_sum_ms);
    }
  }

 private:
  void gather_samples(const char* label, MPI_Comm comm, int root, std::size_t begin) const {
    int comm_rank = 0, comm_size = 1, world_rank = 0;
    MPI_Comm_rank(comm, &comm_rank); MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    const std::size_t local_size = samples_seconds_.size() - begin;
    if (local_size > static_cast<std::size_t>(INT_MAX)) MPI_Abort(comm, 1);
    const int local_count = static_cast<int>(local_size);
    std::vector<int> counts(comm_rank == root ? comm_size : 0);
    MPI_Gather(&local_count, 1, MPI_INT, comm_rank == root ? counts.data() : nullptr,
               1, MPI_INT, root, comm);
    std::vector<int> displs(comm_rank == root ? comm_size : 0);
    std::vector<int> world_ranks(comm_rank == root ? comm_size : 0);
    MPI_Gather(&world_rank, 1, MPI_INT, comm_rank == root ? world_ranks.data() : nullptr,
               1, MPI_INT, root, comm);
    int total = 0;
    if (comm_rank == root) {
      for (int i = 0; i < comm_size; ++i) {
        displs[i] = total;
        if (counts[i] > INT_MAX - total) MPI_Abort(comm, 1);
        total += counts[i];
      }
    }
    std::vector<double> local(local_size);
    for (std::size_t i = 0; i < local_size; ++i) local[i] = samples_seconds_[begin + i] * 1000.0;
    std::vector<double> gathered(comm_rank == root ? static_cast<std::size_t>(total) : 0);
    MPI_Gatherv(local.empty() ? nullptr : local.data(), local_count, MPI_DOUBLE,
                comm_rank == root && !gathered.empty() ? gathered.data() : nullptr,
                comm_rank == root ? counts.data() : nullptr,
                comm_rank == root ? displs.data() : nullptr, MPI_DOUBLE, root, comm);
    if (comm_rank != root || !ccutils::machine_enabled()) return;
    const std::size_t chunk_size = 1024;
    for (int i = 0; i < comm_size; ++i) {
      for (int offset = 0; offset < counts[i]; offset += static_cast<int>(chunk_size)) {
        const int count = std::min(static_cast<int>(chunk_size), counts[i] - offset);
        std::string values = "[";
        for (int j = 0; j < count; ++j) {
          if (j) values += ',';
          values += ccutils::machine::format_real(gathered[displs[i] + offset + j]);
        }
        values += ']';
        ccutils::machine::Record record("timer_samples", ccutils::machine::Scope::Rank);
        record.rank(world_ranks[i]).str("name", label).str("unit", "ms")
          .integer("part", offset / static_cast<int>(chunk_size))
          .boolean("last", offset + count == counts[i]).raw("values", values).emit();
      }
    }
  }

  void print_stats_v2(const char* label, MPI_Comm comm, int root,
                      std::size_t exclude_first_n, bool include_samples) const {
    struct Local {
      int world_rank;
      unsigned long long n;
      double sum, mean, m2, min, max;
      int errors;
    } local;
    int comm_rank = 0, comm_size = 1, world_rank = 0;
    MPI_Comm_rank(comm, &comm_rank); MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    ccutils::set_world_rank(world_rank);
    const std::size_t begin = std::min(exclude_first_n, samples_seconds_.size());
    local.world_rank = world_rank; local.n = samples_seconds_.size() - begin;
    local.sum = local.mean = local.m2 = 0.0;
    local.min = std::numeric_limits<double>::infinity();
    local.max = -std::numeric_limits<double>::infinity();
    local.errors = running_ ? 1 : 0;
    /* Never throw on one rank before the collective: peers may be waiting in
     * MPI_Gather. Report the misuse in the gathered record instead. */
    for (std::size_t i=begin;i<samples_seconds_.size();++i) {
      const double ms = samples_seconds_[i] * 1000.0;
      local.sum += ms; local.min = std::min(local.min, ms); local.max = std::max(local.max, ms);
    }
    if (local.n) local.mean = local.sum / static_cast<double>(local.n);
    for (std::size_t i=begin;i<samples_seconds_.size();++i) {
      const double d = samples_seconds_[i] * 1000.0 - local.mean; local.m2 += d*d;
    }
    if (!local.n) local.min = local.max = 0.0;
    std::vector<Local> all(comm_rank == root ? comm_size : 0);
    MPI_Gather(&local, static_cast<int>(sizeof(Local)), MPI_BYTE,
               comm_rank == root ? all.data() : nullptr, static_cast<int>(sizeof(Local)),
               MPI_BYTE, root, comm);
    if (include_samples) gather_samples(label, comm, root, begin);
    if (comm_rank != root) return;
    unsigned long long total_n = 0; int errors = 0, reporting = 0, slowest = -1;
    double sum = 0, mean = 0, m2 = 0, min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity(), max_sum = -1;
    double rank_sum_min = std::numeric_limits<double>::infinity();
    double rank_sum_max = -std::numeric_limits<double>::infinity();
    double rank_avg_min = std::numeric_limits<double>::infinity();
    double rank_avg_max = -std::numeric_limits<double>::infinity();
    int rank_sum_min_arg = -1, rank_sum_max_arg = -1;
    int rank_avg_min_arg = -1, rank_avg_max_arg = -1;
    std::string per_rank = "[";
    for (int i=0;i<comm_size;++i) {
      const Local& x = all[i]; errors += x.errors;
      if (!x.n) continue;
      if (reporting) per_rank += ',';
      ++reporting; total_n += x.n; sum += x.sum;
      min = std::min(min, x.min); max = std::max(max, x.max); max_sum = std::max(max_sum, x.sum);
      const double delta = x.mean - mean;
      const unsigned long long combined_n = total_n;
      m2 += x.m2 + delta*delta * (static_cast<double>(total_n - x.n) * x.n / combined_n);
      mean += delta * static_cast<double>(x.n) / combined_n;
      const double avg = x.sum / x.n;
      if (x.sum < rank_sum_min) { rank_sum_min = x.sum; rank_sum_min_arg = x.world_rank; }
      if (x.sum > rank_sum_max) { rank_sum_max = x.sum; rank_sum_max_arg = x.world_rank; }
      if (avg < rank_avg_min) { rank_avg_min = avg; rank_avg_min_arg = x.world_rank; }
      if (avg > rank_avg_max) { rank_avg_max = avg; rank_avg_max_arg = x.world_rank; }
      if (x.sum > max_sum) { max_sum = x.sum; slowest = x.world_rank; }
      per_rank += std::string("{\"rank\":") + std::to_string(x.world_rank) +
        ",\"n\":" + std::to_string(x.n) + ",\"sum\":" + ccutils::machine::format_real(x.sum) +
        ",\"avg\":" + ccutils::machine::format_real(avg) +
        ",\"sd\":" + ccutils::machine::format_real(x.n > 1 ? std::sqrt(x.m2/(x.n-1)) : 0.0) +
        ",\"min\":" + ccutils::machine::format_real(x.min) +
        ",\"max\":" + ccutils::machine::format_real(x.max) +
        ",\"errors\":" + std::to_string(x.errors) + "}";
    }
    per_rank += ']';
    if (!total_n) min = max = 0.0;
    if (!reporting) rank_sum_min = rank_sum_max = rank_avg_min = rank_avg_max = 0.0;
    const double sd = total_n > 1 ? std::sqrt(m2 / (total_n - 1)) : 0.0;
    const double imbalance = reporting && sum > 0 ? max_sum / (sum / reporting) : 0.0;
    const std::string distributions = std::string("{\"sum\":{\"min\":") +
      ccutils::machine::format_real(rank_sum_min) + ",\"avg\":" +
      ccutils::machine::format_real(reporting ? sum / reporting : 0.0) + ",\"max\":" +
      ccutils::machine::format_real(rank_sum_max) + ",\"argmin\":" + std::to_string(rank_sum_min_arg) +
      ",\"argmax\":" + std::to_string(rank_sum_max_arg) + "},\"avg\":{\"min\":" +
      ccutils::machine::format_real(rank_avg_min) + ",\"avg\":" +
      ccutils::machine::format_real(total_n ? sum / total_n : 0.0) + ",\"max\":" +
      ccutils::machine::format_real(rank_avg_max) + ",\"argmin\":" + std::to_string(rank_avg_min_arg) +
      ",\"argmax\":" + std::to_string(rank_avg_max_arg) + "}}";
    char comm_name[MPI_MAX_OBJECT_NAME + 1] = ""; int name_len = 0;
    MPI_Comm_get_name(comm, comm_name, &name_len); comm_name[name_len] = '\0';
    int root_world_rank = 0;
    if (comm_rank == root) MPI_Comm_rank(MPI_COMM_WORLD, &root_world_rank);
    char location[256]; std::snprintf(location, sizeof(location), "comm %s, %d ranks", comm_name, comm_size);
    ccutils::human_v2_timer("", label, static_cast<std::size_t>(total_n), mean, sd, min, max, sum, location);
    ccutils::human_flush();
    if (ccutils::machine_enabled()) {
      ccutils::capture_mpi_facts();
      ccutils::machine::Record record("timer", ccutils::machine::Scope::Comm);
      record.str("name", label).str("group", "MPI Timer").str("unit", "ms").str("clock", "mpi_wtime")
        .raw("comm", std::string("{\"name\":") + ccutils::machine::quote(comm_name) +
             ",\"size\":" + std::to_string(comm_size) + ",\"root\":" + std::to_string(root_world_rank) + "}")
        .integer("n", static_cast<long long>(total_n)).real("avg", mean).real("sd", sd)
        .real("min", min).real("max", max).real("sum", sum)
        .raw("per_rank", distributions).raw("rank_stats", per_rank)
        .real("imbalance", imbalance).integer("slowest_rank", slowest)
        .integer("ranks_reporting", reporting).integer("errors", errors).emit();
      const char* policy = std::getenv("CCUTILS_TIMER_RANKS");
      std::string selection = policy && *policy ? policy : "summary";
      int top = comm_size;
      if (selection.compare(0, 4, "top:") == 0) {
        top = std::max(0, std::atoi(selection.c_str() + 4));
        selection = "top";
      }
      if (selection == "all" || selection == "top") {
        std::vector<const Local*> chosen;
        for (int i=0;i<comm_size;++i) if (all[i].n) chosen.push_back(&all[i]);
        std::sort(chosen.begin(), chosen.end(), [](const Local* a, const Local* b) {
          return a->sum > b->sum;
        });
        if (selection == "top" && static_cast<int>(chosen.size()) > top) chosen.resize(top);
        for (std::size_t i=0;i<chosen.size();++i) {
          const Local& x = *chosen[i];
          const double rank_avg = x.sum / x.n;
          ccutils::machine::Record rank_record("timer", ccutils::machine::Scope::Rank);
          rank_record.rank(x.world_rank).str("name", label).str("group", "MPI Timer")
            .str("unit", "ms").str("clock", "mpi_wtime")
            .integer("n", static_cast<long long>(x.n)).real("avg", rank_avg)
            .real("sd", x.n > 1 ? std::sqrt(x.m2/(x.n-1)) : 0.0)
            .real("min", x.min).real("max", x.max).real("sum", x.sum)
            .integer("errors", x.errors).emit();
        }
      }
    }
  }

  double start_seconds_ = 0.0;
  bool running_ = false;
  std::vector<double> samples_seconds_;
};

namespace detail {
typedef NamedRegistry<MpiTimer*> MpiTimerRegistry;

/* Registers an MpiTimer under a name while it is alive (see CCUTILS_MPI_TIMER_DEF). */
class MpiTimerRegistration : public Registration<MpiTimer*> {
 public:
  MpiTimerRegistration(const char* name, MpiTimer* timer)
      : Registration<MpiTimer*>(name, timer, timer) {}
};
}  // namespace detail

/* The live MPI timer called `name`, or a new persistent one. Like every MPI timer
 * report, printing it is collective over the communicator you pass. */
inline MpiTimer& mpi_timer(const std::string& name) {
  MpiTimer* found = nullptr;
  if (detail::MpiTimerRegistry::instance().find(name, &found)) return *found;
  std::shared_ptr<MpiTimer> owner = std::make_shared<MpiTimer>();
  detail::MpiTimerRegistry::instance().add(name, owner.get(), owner.get(), owner);
  return *owner;
}

}  // namespace ccutils

// Historical global name, kept so existing code compiles. It is unrelated to
// ccutils_timers::TimerStats (timers.hpp); the macros below use ccutils::MpiTimer.
using TimerStats = ccutils::MpiTimer;


/**********************************************************************/
/*                           MPI TIMER DEF                            */
/**********************************************************************/

#define CCUTILS_MPI_TIMER_DEF(name) \
  ccutils::MpiTimer name; \
  ccutils::detail::MpiTimerRegistration ccutils_mpi_timer_reg_##name(#name, &name);

/* Binds the MPI timer called `name` (defined elsewhere) to a local reference. */
#define CCUTILS_MPI_TIMER_USE(name) ccutils::MpiTimer& name = ccutils::mpi_timer(#name);

/**********************************************************************/
/*                          MPI TIMER OPERATIONS                      */
/**********************************************************************/

#define CCUTILS_MPI_TIMER_START(name) \
  do { (name).start(); } while (0);

#define CCUTILS_MPI_TIMER_STOP(name) \
  do { (name).stop(); } while (0);

#define CCUTILS_MPI_TIMER_DESTROY(name) \
  do { (name).clear(); } while (0);

#define CCUTILS_MPI_TIMER_PRINT(name, label, comm) \
  do { (name).print_stats((label), (comm)); } while (0);

#define CCUTILS_MPI_TIMER_PRINT_ROOT(name, label, comm, root) \
  do { (name).print_stats((label), (comm), (root)); } while (0);

/**********************************************************************/
/*                          MPI TIMER SHORTCUTS                       */
/**********************************************************************/

#define CCUTILS_MPI_TIMER_INIT(name) \
    CCUTILS_MPI_TIMER_DEF(name)      \
    CCUTILS_MPI_TIMER_START(name)

/* One statement, so it is safe after an unbraced if. The trailing ';' is kept on
 * purpose: callers may (as documented) use these macros without a semicolon. */
#define CCUTILS_MPI_TIMER_CLOSE(name, label, comm) \
    do {                                           \
        CCUTILS_MPI_TIMER_STOP(name)               \
        CCUTILS_MPI_TIMER_PRINT(name, label, comm) \
        CCUTILS_MPI_TIMER_DESTROY(name)            \
    } while (0);


// #define CCUTILS_MPI_TIMEIT(name, statement)
//     CCUTILS_MPI_TIMER_INIT(name)
//     statement;
//     CCUTILS_MPI_TIMER_CLOSE(name, "label", MPI_COMM_WORLD)

#endif
