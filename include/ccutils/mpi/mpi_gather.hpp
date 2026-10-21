#ifndef CCUTILS_MPI_GATHER_HPP
#define CCUTILS_MPI_GATHER_HPP

#include <mpi.h>
#include <climits>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdint.h>
#include <cstring>
#include "../emitter.hpp"
#include "../formats.h"

namespace ccutils { namespace mpi {

inline void append_frame(std::string* blob, char kind, const std::string& payload) {
  blob->push_back(kind);
  const uint64_t length = static_cast<uint64_t>(payload.size());
  for (int shift = 56; shift >= 0; shift -= 8)
    blob->push_back(static_cast<char>((length >> shift) & 0xff));
  blob->append(payload);
}

template <typename Callback>
inline bool visit_frames(const std::string& blob, Callback callback) {
  std::size_t pos = 0;
  while (pos < blob.size()) {
    if (blob.size() - pos < 9) return false;
    const char kind = blob[pos++];
    uint64_t length = 0;
    for (int i = 0; i < 8; ++i)
      length = (length << 8) | static_cast<unsigned char>(blob[pos++]);
    if (length > blob.size() - pos) return false;
    callback(kind, blob.data() + pos, static_cast<std::size_t>(length));
    pos += static_cast<std::size_t>(length);
  }
  return true;
}

inline void gatherv_to_root(MPI_Comm comm, const std::string& local,
                            std::vector<std::string>* out, int root);

inline void all_print(MPI_Comm comm, const char* print_name,
                      const std::string& human, const std::string& machine) {
  int rank = 0, size = 1, world_rank = 0;
  MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  std::vector<int> world_ranks(rank == 0 ? size : 0);
  MPI_Gather(&world_rank, 1, MPI_INT, rank == 0 ? world_ranks.data() : nullptr,
             1, MPI_INT, 0, comm);

  std::string blob;
  if (ccutils::human_enabled()) append_frame(&blob, 'H', human);
  if (ccutils::machine_enabled()) {
    std::size_t begin = 0;
    while (begin < machine.size()) {
      std::size_t end = machine.find('\n', begin);
      if (end == std::string::npos) end = machine.size();
      if (end > begin) append_frame(&blob, 'M', machine.substr(begin, end - begin));
      begin = end == machine.size() ? end : end + 1;
    }
  }
  std::vector<std::string> by_rank;
  gatherv_to_root(comm, blob, &by_rank, 0);

  if (rank == 0) {
    bool any_human = false;
    for (int i = 0; i < size; ++i)
      visit_frames(by_rank[i], [&](char kind, const char*, std::size_t n) {
        if (kind == 'H' && n) any_human = true;
      });
    if (any_human) {
      if (ccutils::human_v2())
        ccutils::human_printf("-- all-ranks: %s --\n", print_name ? print_name : "");
      else if (print_name)
        ccutils::human_printf(CCUTILS_FMT_MPI_PRINT_ALL_NAMED_START, print_name);
    }
    for (int i = 0; i < size; ++i) {
      bool has_human = false;
      visit_frames(by_rank[i], [&](char kind, const char* bytes, std::size_t n) {
        if (kind == 'M') {
          ccutils::machine_write_line_for_rank(std::string(bytes, n), world_ranks[i]);
        } else if (kind == 'H' && n) {
          has_human = true;
          if (ccutils::human_v2())
            ccutils::human_v2_text_block("", world_ranks[i], std::string(bytes, n));
          else {
            ccutils::human_printf(CCUTILS_FMT_MPI_PRINT_ALL_START, i);
            ccutils::human_write(bytes, n);
            if (bytes[n - 1] != '\n') ccutils::human_putc('\n');
            ccutils::human_printf(CCUTILS_FMT_MPI_PRINT_ALL_END, i);
          }
        }
      });
      (void)has_human;
    }
    if (any_human) {
      if (ccutils::human_v2())
        ccutils::human_printf("-- end all-ranks: %s --\n", print_name ? print_name : "");
      else if (print_name)
        ccutils::human_printf(CCUTILS_FMT_MPI_PRINT_ALL_NAMED_END, print_name);
    }
    ccutils::human_flush();
  }
  MPI_Barrier(comm);
}

/* Gather arbitrary rank payloads in rank order. Small payloads use one size
 * Gather and one Gatherv; larger payloads use bounded Gatherv rounds. */
inline void gatherv_to_root(MPI_Comm comm, const std::string& local,
                            std::vector<std::string>* out, int root = 0) {
  int rank = 0, size = 1;
  MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);
  const unsigned long long local_size = static_cast<unsigned long long>(local.size());
  std::vector<unsigned long long> sizes(rank == root ? size : 0);
  MPI_Gather(&local_size, 1, MPI_UNSIGNED_LONG_LONG,
             rank == root ? sizes.data() : nullptr, 1, MPI_UNSIGNED_LONG_LONG, root, comm);
  unsigned long long max_size = 0, total_size = 0;
  int chunked = 0;
  if (rank == root) {
    for (int i=0;i<size;++i) {
      max_size = std::max(max_size, sizes[i]);
      if (sizes[i] > static_cast<unsigned long long>(INT_MAX) - total_size) chunked = 1;
      else total_size += sizes[i];
    }
    if (total_size > static_cast<unsigned long long>(INT_MAX)) chunked = 1;
    if (out) { out->assign(size, std::string()); for (int i=0;i<size;++i) (*out)[i].reserve(static_cast<std::size_t>(sizes[i])); }
  }
  MPI_Bcast(&chunked, 1, MPI_INT, root, comm);
  std::vector<int> counts(rank == root ? size : 0), displs(rank == root ? size : 0);
  if (!chunked) {
    int total = 0;
    if (rank == root) for (int i=0;i<size;++i) {
      counts[i] = static_cast<int>(sizes[i]); displs[i] = total; total += counts[i];
    }
    std::vector<char> bytes(rank == root ? static_cast<std::size_t>(total) : 0);
    MPI_Gatherv(local.empty() ? nullptr : local.data(), static_cast<int>(local.size()), MPI_CHAR,
                rank == root && !bytes.empty() ? bytes.data() : nullptr,
                rank == root ? counts.data() : nullptr, rank == root ? displs.data() : nullptr,
                MPI_CHAR, root, comm);
    if (rank == root && out)
      for (int i=0;i<size;++i) if (counts[i])
        (*out)[i].assign(bytes.data()+displs[i], static_cast<std::size_t>(counts[i]));
    return;
  }
  MPI_Bcast(&max_size, 1, MPI_UNSIGNED_LONG_LONG, root, comm);
  const int chunk = std::max(1, std::min(1024 * 1024, INT_MAX / std::max(size, 1)));
  for (unsigned long long offset = 0; offset < max_size; offset += static_cast<unsigned long long>(chunk)) {
    const int send_count = offset < local_size
        ? static_cast<int>(std::min<unsigned long long>(chunk, local_size - offset)) : 0;
    int total = 0;
    if (rank == root) for (int i=0;i<size;++i) {
      const unsigned long long remain = sizes[i] > offset ? sizes[i] - offset : 0;
      counts[i] = static_cast<int>(std::min<unsigned long long>(chunk, remain));
      displs[i] = total; total += counts[i];
    }
    std::vector<char> bytes(rank == root ? static_cast<std::size_t>(total) : 0);
    const char* send = send_count ? local.data() + static_cast<std::size_t>(offset) : nullptr;
    MPI_Gatherv(send, send_count, MPI_CHAR,
                rank == root && !bytes.empty() ? bytes.data() : nullptr,
                rank == root ? counts.data() : nullptr, rank == root ? displs.data() : nullptr,
                MPI_CHAR, root, comm);
    if (rank == root && out)
      for (int i=0;i<size;++i) if (counts[i]) (*out)[i].append(bytes.data()+displs[i], counts[i]);
  }
}

} } // namespace ccutils::mpi
#endif
