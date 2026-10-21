#ifndef CCUTILS_MPI_MACROS_HPP
#define CCUTILS_MPI_MACROS_HPP

#ifndef CCUTILS_ENABLE_MPI
#error "ccutils MPI headers require -DCCUTILS_ENABLE_MPI"
#endif

#include <unistd.h>
#include <limits.h>
#include <errno.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include "../formats.h"
#include "../macros.hpp"
#include "mpi_info.hpp"
#include "mpi_gather.hpp"

/**********************************************************************/
/*                           INIT                                     */
/**********************************************************************/

#define CCUTILS_MPI_INIT                                  \
    int ccutils_inmacro_myid;                             \
    int ccutils_inmacro_ntask;                            \
    MPI_Comm_rank(MPI_COMM_WORLD, &ccutils_inmacro_myid); \
    MPI_Comm_size(MPI_COMM_WORLD, &ccutils_inmacro_ntask); \
    ccutils::set_world_rank(ccutils_inmacro_myid);        \
    ccutils::capture_mpi_facts();

namespace ccutils {
/* MPI_COMM_WORLD rank and size, usable in any function without CCUTILS_MPI_INIT
 * (the rank is cached after the first call). */
inline int mpi_rank() {
  capture_mpi_facts();
  int rank = announced_world_rank();
  if (rank < 0) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    set_world_rank(rank);
  }
  return rank;
}
inline int mpi_size() {
  capture_mpi_facts();
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
}

inline unsigned int mpi_flush_wait_us() {
  const char* value = getenv("CCUTILS_FLUSH_WAIT_US");
  if (!value || !*value) return 200000U;
  errno = 0;
  char* end = nullptr;
  unsigned long parsed = strtoul(value, &end, 10);
  if (errno == ERANGE || end == value || *end != '\0' || parsed > UINT_MAX)
    return 200000U;
  return static_cast<unsigned int>(parsed);
}

inline void mpi_flush_wait(unsigned int useconds) {
  human_flush();
  fflush(stdout);
  fflush(stderr);
  while (useconds) {
    useconds_t chunk = static_cast<useconds_t>(useconds > 999999U ? 999999U : useconds);
    usleep(chunk);
    useconds -= chunk;
  }
}

inline bool human_write_file(const char* path) {
  FILE* input = fopen(path, "rb");
  if (!input) {
    human_printf("[ccutils] Cannot open temporary output '%s' for reading.\n", path);
    return false;
  }
  char buffer[8192];
  size_t count;
  while ((count = fread(buffer, 1, sizeof(buffer), input)) != 0)
    human_write(buffer, count);
  bool ok = !ferror(input);
  fclose(input);
  if (!ok)
    human_printf("[ccutils] Error reading temporary output '%s'.\n", path);
  return ok;
}
}  // namespace ccutils

/**********************************************************************/
/*                           FLUSH UTILITY                            */
/**********************************************************************/

#define CCUTILS_FLUSH_WAIT(useconds) \
    ccutils::mpi_flush_wait(useconds);

/**********************************************************************/
/*                        MPI ONCE MACROS                              */
/**********************************************************************/

#define CCUTILS_MPI_ONCE(statement) \
    if (ccutils::mpi_rank() == 0)  \
    {                               \
        statement;                  \
    }

#define CCUTILS_MPI_PRINT_ONCE(print_statement) \
    if (ccutils::mpi_rank() == 0)              \
    {                                           \
        ccutils::human_capture_begin();          \
        print_statement;                        \
        ccutils::machine::text_records(NULL, ccutils::human_capture_end(), \
                                       ccutils::machine::Scope::Global); \
    }                                           \
    CCUTILS_FLUSH_WAIT(ccutils::mpi_flush_wait_us())

#define CCUTILS_MPI_PRINTF_ONCE(...) \
    CCUTILS_MPI_PRINT_ONCE(ccutils::human_printf(__VA_ARGS__))

/**********************************************************************/
/*                        MPI ALL PRINT MACROS                        */
/**********************************************************************/

// Initial implementation retained below for source history; the active framed
// macro is defined after it and routes H/M data through mpi::all_print().
#define CCUTILS_MPI_ALL_PRINT_IMPL(_CCUTILS_COMM, _CCUTILS_PRINT_NAME, PRINTS_CODE_BLOCK)           \
do {                                                                                  \
    int _ccutils_myid, _ccutils_numprocs;                                             \
    MPI_Comm_rank(_CCUTILS_COMM, &_ccutils_myid);                                    \
    MPI_Comm_size(_CCUTILS_COMM, &_ccutils_numprocs);                                \
    int _ccutils_world_rank = 0;                                                       \
    MPI_Comm_rank(MPI_COMM_WORLD, &_ccutils_world_rank);                              \
    ccutils::set_world_rank(_ccutils_world_rank);                                     \
    ccutils::capture_mpi_facts();                                                     \
    const char *_ccutils_print_name = (_CCUTILS_PRINT_NAME);                          \
                                                                                      \
    /* STEP 1: Each rank writes output to local memory */                             \
    char  *_ccutils_buf = NULL;                                                       \
    size_t _ccutils_bsz = 0;                                                          \
    FILE  *fp = open_memstream(&_ccutils_buf, &_ccutils_bsz);                         \
    ccutils::machine::emit_run_start();                                               \
    ccutils::machine_capture_begin();                                                 \
    char _ccutils_comm_name[MPI_MAX_OBJECT_NAME + 1] = "";                           \
    int _ccutils_comm_name_length = 0;                                                \
    MPI_Comm_get_name(_CCUTILS_COMM, _ccutils_comm_name, &_ccutils_comm_name_length); \
    _ccutils_comm_name[_ccutils_comm_name_length] = '\0';                            \
    if (fp != NULL) {                                                                 \
        PRINTS_CODE_BLOCK;                                                            \
        fclose(fp);                                                                   \
    } else {                                                                          \
        fprintf(stderr, "[Rank %d] WARNING: open_memstream failed, "                  \
                        "output will be missing.\n", _ccutils_myid);                  \
        _ccutils_buf = (char *)calloc(1, 1);                                          \
        _ccutils_bsz = 0;                                                             \
    }                                                                                 \
    ccutils::machine::text_records(_ccutils_print_name,                              \
        std::string(_ccutils_buf, _ccutils_bsz), ccutils::machine::Scope::Rank,      \
        _ccutils_comm_name);                                                           \
    std::vector<std::string> _ccutils_machine_lines = ccutils::machine_capture_end(); \
    std::string _ccutils_machine_blob;                                                \
    for (size_t _ccutils_mi = 0; _ccutils_mi < _ccutils_machine_lines.size(); ++_ccutils_mi) { \
        _ccutils_machine_blob += _ccutils_machine_lines[_ccutils_mi];                \
        _ccutils_machine_blob += '\n';                                               \
    }                                                                                 \
                                                                                      \
    /* Guard: abort if any rank's output exceeds INT_MAX */                           \
    if (_ccutils_bsz > (size_t)INT_MAX) {                                             \
        fprintf(stderr, "[Rank %d] FATAL: per-rank output exceeds INT_MAX "           \
                        "(%zu bytes), aborting.\n", _ccutils_myid, _ccutils_bsz);     \
        MPI_Abort(_CCUTILS_COMM, 1);                                                 \
    }                                                                                 \
    int _ccutils_local_count = (int)_ccutils_bsz;                                     \
    std::vector<int> _ccutils_world_ranks(_ccutils_myid == 0 ? _ccutils_numprocs : 0); \
    MPI_Gather(&_ccutils_world_rank, 1, MPI_INT,                                    \
               _ccutils_myid == 0 ? _ccutils_world_ranks.data() : NULL, 1, MPI_INT, 0, _CCUTILS_COMM); \
                                                                                      \
    int *_ccutils_recvcounts = NULL;                                                  \
    int *_ccutils_displs     = NULL;                                                  \
    char *_ccutils_recvbuf   = NULL;                                                  \
                                                                                      \
    if (_ccutils_myid == 0) {                                                         \
        _ccutils_recvcounts = (int *)malloc(_ccutils_numprocs * sizeof(int));         \
        _ccutils_displs     = (int *)malloc(_ccutils_numprocs * sizeof(int));         \
        if (!_ccutils_recvcounts || !_ccutils_displs) {                               \
            fprintf(stderr, "[Rank 0] FATAL: metadata malloc failed.\n");             \
            MPI_Abort(_CCUTILS_COMM, 1);                                             \
        }                                                                             \
    }                                                                                 \
                                                                                      \
    /* STEP 2: Gather sizes */                                                        \
    MPI_Gather(&_ccutils_local_count, 1, MPI_INT,                                     \
               _ccutils_recvcounts, 1, MPI_INT, 0, _CCUTILS_COMM);                   \
                                                                                      \
    /* Rank 0: compute displacements and guard against total overflow */              \
    if (_ccutils_myid == 0) {                                                         \
        long long _ccutils_total = 0;                                                 \
        for (int i = 0; i < _ccutils_numprocs; i++) {                                 \
            _ccutils_displs[i] = (int)_ccutils_total;                                 \
            _ccutils_total += _ccutils_recvcounts[i];                                 \
            if (_ccutils_total > (long long)INT_MAX) {                                \
                fprintf(stderr, "[Rank 0] FATAL: combined output exceeds INT_MAX "    \
                                "after rank %d, aborting.\n", i);                     \
                MPI_Abort(_CCUTILS_COMM, 1);                                         \
            }                                                                         \
        }                                                                             \
        int _ccutils_total_chars = (int)_ccutils_total;                               \
        _ccutils_recvbuf = (char *)malloc(_ccutils_total_chars + 1);                  \
        if (!_ccutils_recvbuf) {                                                      \
            fprintf(stderr, "[Rank 0] FATAL: combined buffer malloc failed.\n");      \
            MPI_Abort(_CCUTILS_COMM, 1);                                             \
        }                                                                             \
        _ccutils_recvbuf[_ccutils_total_chars] = '\0';                                \
    }                                                                                 \
                                                                                      \
    /* STEP 3: Gather actual strings */                                               \
    MPI_Gatherv(_ccutils_buf, _ccutils_local_count, MPI_CHAR,                         \
                _ccutils_recvbuf, _ccutils_recvcounts, _ccutils_displs, MPI_CHAR,     \
                0, _CCUTILS_COMM);                                                   \
    std::vector<std::string> _ccutils_machine_by_rank;                              \
    ccutils::mpi::gatherv_to_root(_CCUTILS_COMM, _ccutils_machine_blob,             \
                                   &_ccutils_machine_by_rank, 0);                    \
                                                                                      \
    /* STEP 4: Rank 0 prints and cleans up */                                         \
    if (_ccutils_myid == 0) {                                                         \
        for (int _ccutils_rank = 0; _ccutils_rank < _ccutils_numprocs; ++_ccutils_rank) { \
            const std::string& _ccutils_mblob = _ccutils_machine_by_rank[_ccutils_rank]; \
            size_t _ccutils_pos = 0;                                                  \
            while (_ccutils_pos < _ccutils_mblob.size()) {                           \
                size_t _ccutils_end = _ccutils_mblob.find('\n', _ccutils_pos);       \
                if (_ccutils_end == std::string::npos) break;                        \
                ccutils::machine_write_line_for_rank(                               \
                    _ccutils_mblob.substr(_ccutils_pos, _ccutils_end - _ccutils_pos), _ccutils_world_ranks[_ccutils_rank]); \
                _ccutils_pos = _ccutils_end + 1;                                     \
            }                                                                         \
        }                                                                             \
        int _ccutils_total_size = _ccutils_displs[_ccutils_numprocs - 1]              \
                                + _ccutils_recvcounts[_ccutils_numprocs - 1];         \
        if (_ccutils_total_size > 0) {                                                \
            if (ccutils::human_v2()) ccutils::human_printf("-- all-ranks: %s --\n", _ccutils_print_name ? _ccutils_print_name : ""); \
            else if (_ccutils_print_name != NULL) ccutils::human_printf(CCUTILS_FMT_MPI_PRINT_ALL_NAMED_START, _ccutils_print_name); \
            for (int _ccutils_rank = 0; _ccutils_rank < _ccutils_numprocs;            \
                 ++_ccutils_rank) {                                                  \
                int _ccutils_rank_size = _ccutils_recvcounts[_ccutils_rank];          \
                if (_ccutils_rank_size == 0) continue;                               \
                if (ccutils::human_v2()) {                                               \
                    ccutils::human_v2_text_block("", _ccutils_rank,                    \
                        std::string(_ccutils_recvbuf + _ccutils_displs[_ccutils_rank], (size_t)_ccutils_rank_size)); \
                } else {                                                                 \
                    ccutils::human_printf(CCUTILS_FMT_MPI_PRINT_ALL_START, _ccutils_rank); \
                    ccutils::human_write(_ccutils_recvbuf + _ccutils_displs[_ccutils_rank], (size_t)_ccutils_rank_size); \
                    if (_ccutils_recvbuf[_ccutils_displs[_ccutils_rank] + _ccutils_rank_size - 1] != '\n') ccutils::human_putc('\n'); \
                    ccutils::human_printf(CCUTILS_FMT_MPI_PRINT_ALL_END, _ccutils_rank);  \
                }                                                                         \
            }                                                                         \
            if (ccutils::human_v2()) ccutils::human_printf("-- end all-ranks: %s --\n", _ccutils_print_name ? _ccutils_print_name : ""); \
            else if (_ccutils_print_name != NULL) ccutils::human_printf(CCUTILS_FMT_MPI_PRINT_ALL_NAMED_END, _ccutils_print_name); \
            ccutils::human_flush();                                                          \
        }                                                                             \
        free(_ccutils_recvcounts);                                                    \
        free(_ccutils_displs);                                                        \
        free(_ccutils_recvbuf);                                                       \
    }                                                                                 \
                                                                                      \
    free(_ccutils_buf);                                                               \
    MPI_Barrier(_CCUTILS_COMM);                                                      \
} while (0); 

/* Framed implementation: user bytes and captured machine records travel in
 * the same bounded gather. H/M payloads are interpreted only by their sink. */
#undef CCUTILS_MPI_ALL_PRINT_IMPL
#define CCUTILS_MPI_ALL_PRINT_IMPL(_CCUTILS_COMM, _CCUTILS_PRINT_NAME, PRINTS_CODE_BLOCK) \
do { \
    int ccutils_ap_world_rank_ = 0; \
    MPI_Comm_rank(MPI_COMM_WORLD, &ccutils_ap_world_rank_); \
    ccutils::set_world_rank(ccutils_ap_world_rank_); \
    ccutils::capture_mpi_facts(); \
    const char* ccutils_ap_name_ = (_CCUTILS_PRINT_NAME); \
    char* ccutils_ap_buffer_ = NULL; \
    size_t ccutils_ap_size_ = 0; \
    FILE* fp = open_memstream(&ccutils_ap_buffer_, &ccutils_ap_size_); \
    ccutils::machine::emit_run_start(); \
    ccutils::machine_capture_begin(); \
    if (fp != NULL) { PRINTS_CODE_BLOCK; fclose(fp); } \
    else { ccutils_ap_buffer_ = (char*)calloc(1, 1); ccutils_ap_size_ = 0; } \
    char ccutils_ap_comm_name_[MPI_MAX_OBJECT_NAME + 1] = ""; \
    int ccutils_ap_comm_name_size_ = 0; \
    MPI_Comm_get_name(_CCUTILS_COMM, ccutils_ap_comm_name_, &ccutils_ap_comm_name_size_); \
    ccutils_ap_comm_name_[ccutils_ap_comm_name_size_] = '\0'; \
    ccutils::machine::text_records(ccutils_ap_name_, \
        std::string(ccutils_ap_buffer_, ccutils_ap_size_), ccutils::machine::Scope::Rank, \
        ccutils_ap_comm_name_); \
    std::vector<std::string> ccutils_ap_records_ = ccutils::machine_capture_end(); \
    std::string ccutils_ap_machine_; \
    for (size_t ccutils_ap_i_ = 0; ccutils_ap_i_ < ccutils_ap_records_.size(); ++ccutils_ap_i_) { \
        ccutils_ap_machine_ += ccutils_ap_records_[ccutils_ap_i_]; \
        ccutils_ap_machine_ += '\n'; \
    } \
    std::string ccutils_ap_human_(ccutils_ap_buffer_, ccutils_ap_size_); \
    free(ccutils_ap_buffer_); \
    ccutils::mpi::all_print(_CCUTILS_COMM, ccutils_ap_name_, ccutils_ap_human_, ccutils_ap_machine_); \
} while (0);

#define CCUTILS_MPI_ALL_PRINT(PRINTS_CODE_BLOCK) \
    CCUTILS_MPI_ALL_PRINT_IMPL(MPI_COMM_WORLD, NULL, PRINTS_CODE_BLOCK)

#define CCUTILS_MPI_ALL_PRINT_NAMED(print_name, PRINTS_CODE_BLOCK) \
    CCUTILS_MPI_ALL_PRINT_IMPL(MPI_COMM_WORLD, #print_name, PRINTS_CODE_BLOCK)

#define CCUTILS_MPI_ALL_PRINT_COMM(comm, PRINTS_CODE_BLOCK) \
    CCUTILS_MPI_ALL_PRINT_IMPL(comm, NULL, PRINTS_CODE_BLOCK)

#define CCUTILS_MPI_ALL_PRINT_NAMED_COMM(comm, print_name, PRINTS_CODE_BLOCK) \
    CCUTILS_MPI_ALL_PRINT_IMPL(comm, #print_name, PRINTS_CODE_BLOCK)

namespace ccutils {
/* MPI counterpart to Section. end() is collective and every communicator
 * member must call it. Destruction is always local to remain exception-safe. */
class MpiSection {
 public:
  MpiSection(const std::string& name, const char* title)
      : name_(name), ref_(), uncaught_(uncaught_exception_count()), active_(true) {
    start(name_, title);
    ref_ = section(name_);
  }
  ~MpiSection() {
    if (active_) close_locally(name_, uncaught_exception_count() > uncaught_);
  }
  MpiSection(const MpiSection&) = delete;
  MpiSection& operator=(const MpiSection&) = delete;

  SectionRef ref() { return ref_; }
  template <class T> MpiSection& put(const std::string& key, T&& value) {
    ref_.put(key, std::forward<T>(value)); return *this;
  }
  template <class T> MpiSection& put(const std::string& key1, const std::string& key2, T&& value) {
    ref_.put(key1, key2, std::forward<T>(value)); return *this;
  }

  void end() {
    if (!active_) return;
    active_ = false;
    end_named(name_, "ccutils_rank_json");
  }
  static void start(const std::string& name, const char* title) {
    mpi_rank();
    machine::section_push(name.c_str());
    CCUTILS_MPI_PRINT_ONCE(
      if (human_v2()) human_section_start(machine::section_path(), title);
      else human_printf(CCUTILS_FMT_SECTION_START, name.c_str(), title);
      human_flush(); machine::emit_section_start(title));
    machine::begin_mpi_section_records();
  }
  static void end_named(const std::string& name, const char* print_name = "ccutils_rank_json") {
    if (uncaught_exception_count() > 0) { close_locally(name, true); return; }
    const int _ccutils_defer_depth = machine::suspend_mpi_section_records();
#ifndef CCUTILS_NO_JSON
    SectionRef ref = section(name);
    CCUTILS_MPI_ALL_PRINT_IMPL(MPI_COMM_WORLD, print_name,
      machine::flush_deferred_records();
      machine::data_object(ref.json(), machine::Scope::Rank);
      if (!ref.json().empty()) fprintf(fp, "%s\n", CCUTILS_JSON_DUMP(ref.json()).c_str()));
    ref.json().clear();
    if (mpi_rank() == 0 && !ref.global_json().empty()) {
      machine::data_object(ref.global_json());
      if (human_v2()) {
        for (nlohmann::json::const_iterator it = ref.global_json().begin(); it != ref.global_json().end(); ++it)
          human_printf("%s = %s\n", it.key().c_str(), it.value().dump().c_str());
      } else {
        human_printf(CCUTILS_FMT_GLOBAL_JSON_START, "ccutils_global_json");
        human_printf("%s\n", CCUTILS_JSON_DUMP(ref.global_json()).c_str());
        human_printf(CCUTILS_FMT_GLOBAL_JSON_END, "ccutils_global_json");
      }
      human_flush();
    }
#endif
    machine::flush_deferred_records();
    machine::resume_mpi_section_records(_ccutils_defer_depth > 0 ? _ccutils_defer_depth - 1 : 0);
    CCUTILS_MPI_PRINT_ONCE(
      if (human_v2()) human_section_end(machine::section_path());
      else human_printf(CCUTILS_FMT_SECTION_END, name.c_str());
      human_flush());
    machine::section_end(name.c_str(), mpi_rank() == 0, true);
  }

 private:
  static int uncaught_exception_count() {
#if __cplusplus >= 201703L
    return std::uncaught_exceptions();
#else
    return std::uncaught_exception() ? 1 : 0;
#endif
  }
  static void close_locally(const std::string& name, bool unwinding) {
    machine::flush_deferred_records();
    machine::end_mpi_section_records();
    if (!unwinding && mpi_rank() == 0) {
      human_printf(CCUTILS_FMT_SECTION_END, name.c_str());
      human_flush();
    }
    machine::section_end_local(name.c_str(), !unwinding);
  }
  std::string name_;
  SectionRef ref_;
  int uncaught_;
  bool active_;
};
}  // namespace ccutils

/**********************************************************************/
/*                       MPI PROCESS PRINT MACROS                         */
/**********************************************************************/

#define CCUTILS_MPI_COMMUNICATOR_PRINT(CM, X)                                                                                           \
    {                                                                                                                                   \
        int inmacro_myid, inmacro_ntask;                                                                                                \
        MPI_Comm_rank(CM, &inmacro_myid);                                                                                               \
        MPI_Comm_size(CM, &inmacro_ntask);                                                                                              \
        char name[MPI_MAX_OBJECT_NAME];                                                                                                 \
        int name_length;                                                                                                                \
        MPI_Comm_get_name(CM, name, &name_length);                                                                                      \
        FILE *fp;                                                                                                                       \
        char s[MPI_MAX_OBJECT_NAME + 64];                                                                                               \
        snprintf(s, sizeof(s), "temp_%s_%d_%d.txt", name, inmacro_myid, ccutils::mpi_rank());                                          \
        fp = fopen(s, "w");                                                                                                             \
        if (!fp) {                                                                                                                       \
            ccutils::human_printf("[ccutils] Cannot create temporary output '%s'.\n", s);                                             \
            MPI_Abort(CM, 1);                                                                                                            \
        }                                                                                                                                \
        fclose(fp);                                                                                                                     \
        fp = fopen(s, "a+");                                                                                                            \
        if (!fp) {                                                                                                                       \
            ccutils::human_printf("[ccutils] Cannot open temporary output '%s'.\n", s);                                                \
            MPI_Abort(CM, 1);                                                                                                            \
        }                                                                                                                                \
        fprintf(fp, "------------------------- Proc %d File %s Line %d -------------------------\n", inmacro_myid, __FILE__, __LINE__); \
        X;                                                                                                                              \
        if (inmacro_myid == inmacro_ntask - 1)                                                                                          \
            fprintf(fp, "--------------------------------------------------------------------------\n");                                \
        fclose(fp);                                                                                                                     \
        for (int i = 0; i < inmacro_ntask; i++)                                                                                         \
        {                                                                                                                               \
            if (inmacro_myid == i)                                                                                                      \
            {                                                                                                                           \
                fflush(stdout);                                                                                                          \
                ccutils::human_write_file(s);                                                                                            \
                ccutils::human_flush();                                                                                                  \
                fflush(stdout);                                                                                                          \
                if (remove(s) != 0)                                                                                                     \
                    ccutils::human_printf("[ccutils] Cannot remove temporary output '%s'.\n", s);                                     \
            }                                                                                                                           \
            MPI_Barrier(CM);                                                                                                            \
        }                                                                                                                               \
    }

#define CCUTILS_MPI_PROCESS_PRINT(CM, P, X)                                                                                                   \
    {                                                                                                                                         \
        int myid, ntask;                                                                                                                      \
        MPI_Comm_rank(CM, &myid);                                                                                                             \
        MPI_Comm_size(CM, &ntask);                                                                                                            \
        if (myid == P)                                                                                                                        \
        {                                                                                                                                     \
            fflush(stdout);                                                                                                                   \
            ccutils::human_printf("--------------------- Proc %d of %d. File %s Line %d ---------------------\n", myid, ntask, __FILE__, __LINE__); \
            ccutils::human_flush();                                                                                                           \
            X;                                                                                                                                \
            fflush(stdout);                                                                                                                   \
            ccutils::human_printf("--------------------------------------------------------------------------\n");                                  \
            ccutils::human_flush();                                                                                                           \
        }                                                                                                                                     \
    }

// TODO(P0-golden-gaps): no golden covers the BUFFERED_PRINT_* macros yet (they create temp files in the cwd).
//   Details: docs/TODO.md  (### P0-golden-gaps)
#define CCUTILS_MPI_BUFFERED_PRINT_ADD(PRINTS_CODE_BLOCK)                   \
    {                                                                       \
        FILE *fp;                                                           \
        char fname[256];                                                    \
        char job_id[64];                                                    \
        char *slurm_job_id = getenv("SLURM_JOB_ID");                        \
                                                                            \
        if (slurm_job_id == NULL)                                           \
            sprintf(job_id, "%d", getpid());                                \
        else                                                                \
            sprintf(job_id, "%s", slurm_job_id);                            \
                                                                            \
        sprintf(fname, "ccutils_bufprint_%s_rank%d.txt",                    \
                job_id, ccutils::mpi_rank());                              \
                                                                            \
        fp = fopen(fname, "a"); /* creates file if not existing */          \
        if (fp != NULL)                                                     \
        {                                                                   \
            PRINTS_CODE_BLOCK;                                              \
            fclose(fp);                                                     \
        }                                                                   \
        else                                                                \
        {                                                                   \
            fprintf(stderr, CCUTILS_FMT_ERROR,                              \
                    __LINE__, __FILE__,                                     \
                    "MPI_BUFFERED_PRINT_ADD: could not open buffer file."); \
        }                                                                   \
    }

/**
 * This macro will flush only the calling rank buffer
 */
#define CCUTILS_MPI_BUFFERED_PRINT_FLUSH                                     \
    {                                                                        \
        FILE *fp;                                                            \
        char fname[256];                                                     \
        char job_id[64];                                                     \
        char *slurm_job_id = getenv("SLURM_JOB_ID");                         \
                                                                             \
        if (slurm_job_id == NULL)                                            \
            sprintf(job_id, "%d", getpid());                                 \
        else                                                                 \
            sprintf(job_id, "%s", slurm_job_id);                             \
                                                                             \
        sprintf(fname, "ccutils_bufprint_%s_rank%d.txt",                     \
                job_id, ccutils::mpi_rank());                               \
                                                                             \
        fp = fopen(fname, "r");                                              \
        if (fp != NULL)                                                      \
        {                                                                    \
            int c = fgetc(fp);                                              \
            int has_content = c != EOF;                                     \
            if (has_content)                                                 \
            {                                                                \
                ccutils::human_printf(CCUTILS_FMT_MPI_PRINT_BUFFERED_START,                 \
                       ccutils::mpi_rank());                                \
                ccutils::human_putc((char)c);                                                  \
                while ((c = fgetc(fp)) != EOF)                              \
                    ccutils::human_putc((char)c);                                              \
            }                                                                \
                                                                             \
            fclose(fp);                                                      \
                                                                             \
            if (remove(fname) != 0)                                          \
            {                                                                \
                fprintf(stderr, CCUTILS_FMT_ERROR,                           \
                        __LINE__, __FILE__,                                  \
                        "MPI_BUFFERED_PRINT_FLUSH: could not remove file."); \
            }                                                                \
                                                                             \
            if (has_content)                                                 \
                ccutils::human_printf(CCUTILS_FMT_MPI_PRINT_BUFFERED_END,                   \
                       ccutils::mpi_rank());                                \
        }                                                                    \
    }

/**
 * This macro MUST be called by all ranks
 */
#define CCUTILS_MPI_BUFFERED_PRINT_ALL_FLUSH                                    \
    {                                                                           \
        char fname[256];                                                        \
        char job_id[64];                                                        \
        char *slurm_job_id = getenv("SLURM_JOB_ID");                            \
                                                                                \
        if (slurm_job_id == NULL)                                               \
            sprintf(job_id, "%d", getpid());                                    \
        else                                                                    \
            sprintf(job_id, "%s", slurm_job_id);                                \
                                                                                \
        for (int ccutils_i_ = 0; ccutils_i_ < ccutils::mpi_size();          \
             ++ccutils_i_)                                                     \
        {                                                                       \
            if (ccutils::mpi_rank() == ccutils_i_)                            \
            {                                                                   \
                sprintf(fname, "ccutils_bufprint_%s_rank%d.txt",                \
                        job_id, ccutils::mpi_rank());                          \
                                                                                \
                FILE *fp = fopen(fname, "r");                                   \
                if (fp != NULL)                                                 \
                {                                                               \
                    int c = fgetc(fp);                                          \
                    int has_content = c != EOF;                                 \
                    if (has_content)                                            \
                    {                                                           \
                        ccutils::human_printf(CCUTILS_FMT_MPI_PRINT_BUFFERED_START,            \
                               ccutils::mpi_rank());                           \
                        ccutils::human_putc((char)c);                                             \
                        while ((c = fgetc(fp)) != EOF)                          \
                            ccutils::human_putc((char)c);                                         \
                    }                                                           \
                                                                                \
                    fclose(fp);                                                 \
                                                                                \
                    if (remove(fname) != 0)                                     \
                    {                                                           \
                        fprintf(stderr, CCUTILS_FMT_ERROR,                      \
                                __LINE__, __FILE__,                             \
                                "MPI_BUFFERED_PRINT_ALL_FLUSH: could not rm."); \
                    }                                                           \
                                                                                \
                    if (has_content)                                            \
                    {                                                           \
                        ccutils::human_printf(CCUTILS_FMT_MPI_PRINT_BUFFERED_END,              \
                               ccutils::mpi_rank());                           \
                        ccutils::human_flush();                                         \
                    }                                                           \
                }                                                               \
            }                                                                   \
                                                                                \
            MPI_Barrier(MPI_COMM_WORLD);                                        \
        }                                                                       \
    }

/**********************************************************************/
/*                            MPI JSON MACROS                         */
/**********************************************************************/

#ifndef CCUTILS_NO_JSON
#define CCUTILS_DECLARE_GLOBAL_JSON(name)   /* sections live in the registry now */
#define CCUTILS_DECLARE_LOCAL_JSON(name)

/* Data goes to the open section called `name`, from any function or file. */
#define CCUTILS_MPI_GLOBAL_JSON_PUT(name, key, ...) \
  do { \
    if (ccutils::mpi_rank() == 0) \
      ccutils::section(#name).put_global(key, (__VA_ARGS__)); \
  } while (0)

#define CCUTILS_MPI_LOCAL_JSON_PUT(name, key, ...) \
  do { \
    ccutils::section(#name).put(key, (__VA_ARGS__)); \
  } while (0)

/* Rank-local named JSON object. PRINT gathers nonempty objects and preserves
 * rank identity in the standard MPI ALL PRINT format. */
#define CCUTILS_MPI_JSON_DEF(name) \
  nlohmann::json ccutils_mpi_json_##name; \
  ccutils::detail::JsonRegistration ccutils_mpi_json_reg_##name(#name, &ccutils_mpi_json_##name)
/* Binds the rank-local JSON object called `name` (defined elsewhere) to a local reference. */
#define CCUTILS_MPI_JSON_USE(name) \
  nlohmann::json& ccutils_mpi_json_##name = ccutils::named_json(#name);
#define CCUTILS_MPI_JSON_PUT(name, key, ...) \
  (ccutils_mpi_json_##name[key] = (__VA_ARGS__));
#define CCUTILS_MPI_JSON_SUB_PUT(name, key1, key2, ...) \
  (ccutils_mpi_json_##name[key1][key2] = (__VA_ARGS__));
#define CCUTILS_MPI_JSON_PRINT(name) \
  do { \
    CCUTILS_MPI_ALL_PRINT_NAMED(name, \
      if (!ccutils_mpi_json_##name.empty()) \
        fprintf(fp, "%s\n", CCUTILS_JSON_DUMP(ccutils_mpi_json_##name).c_str())); \
    ccutils_mpi_json_##name.clear(); \
  } while (0);

#define CCUTILS_MPI_SUBSECTION_DEF(section, subsection, title) \
  ccutils::MpiSection::start(#subsection, title);
#define CCUTILS_MPI_SUBSECTION_JSON_PUT(section_name, subsection, key, ...) \
  (ccutils::section(#subsection).put(key, (__VA_ARGS__)));
#define CCUTILS_MPI_SUBSECTION_JSON_SUB_PUT(section_name, subsection, key1, key2, ...) \
  (ccutils::section(#subsection).put(key1, key2, (__VA_ARGS__)));
#define CCUTILS_MPI_SUBSECTION_END(section_name, subsection) \
  ccutils::MpiSection::end_named(#subsection, "ccutils_subsection_json");

#else
#define CCUTILS_DECLARE_GLOBAL_JSON(name)
#define CCUTILS_DECLARE_LOCAL_JSON(name)
#define CCUTILS_MPI_GLOBAL_JSON_PUT(name, key, ...) ((void)0)
#define CCUTILS_MPI_LOCAL_JSON_PUT(name, key, ...) ((void)0)
#define CCUTILS_MPI_JSON_DEF(name)
#define CCUTILS_MPI_JSON_USE(name)
#define CCUTILS_MPI_JSON_PUT(name, key, ...) ((void)0);
#define CCUTILS_MPI_JSON_SUB_PUT(name, key1, key2, ...) ((void)0);
#define CCUTILS_MPI_JSON_PRINT(name) ((void)0);
#define CCUTILS_MPI_SUBSECTION_DEF(section, subsection, title) \
  ccutils::MpiSection::start(#subsection, title);
#define CCUTILS_MPI_SUBSECTION_JSON_PUT(section, subsection, key, ...) ((void)0);
#define CCUTILS_MPI_SUBSECTION_JSON_SUB_PUT(section, subsection, key1, key2, ...) ((void)0);
#define CCUTILS_MPI_SUBSECTION_END(section, subsection) \
  ccutils::MpiSection::end_named(#subsection, "ccutils_subsection_json");
#endif

/**********************************************************************/
/*                        MPI SECTION MACROS                            */
/**********************************************************************/

/* The section markers are printed directly: CCUTILS_SECTION_DEF/END declare and
 * read a non-MPI JSON variable that would be local to the PRINT_ONCE block. */
#define CCUTILS_MPI_SECTION_DEF(name, title) \
    ccutils::MpiSection::start(#name, title);

#ifndef CCUTILS_NO_JSON
#define CCUTILS_MPI_SECTION_END(name) ccutils::MpiSection::end_named(#name, "ccutils_rank_json");
#else
#define CCUTILS_MPI_SECTION_END(name) ccutils::MpiSection::end_named(#name, "ccutils_rank_json");
#endif

/**********************************************************************/
/*                       MPI + CUDA PRINT MACROS                        */
/**********************************************************************/

#ifdef CCUTILS_ENABLE_CUDA
#define CCUTILS_MPI_CUDA_PRINT_DEVICE                                           \
    {                                                                           \
        int dev;                                                                \
        cudaError_t err = cudaGetDevice(&dev);                                  \
        if (err == cudaSuccess)                                                 \
        {                                                                       \
            ccutils::human_printf("[[Rank %d]] CUDA device: %d\n", ccutils::mpi_rank(), dev); \
        }                                                                       \
        else                                                                    \
        {                                                                       \
            ccutils::human_printf("cudaGetDevice failed: %s\n", cudaGetErrorString(err));      \
        }                                                                       \
    }
#endif

/**********************************************************************/
/*                             MPI STATUS CHECK                       */
/**********************************************************************/

#define CCUTILS_MPI_STATUS_CHECK(NREQ, STATV, COMM)                      \
    for (int i = 0; i < NREQ; i++)                                       \
    {                                                                    \
        if (STATV[i].MPI_ERROR != MPI_SUCCESS)                           \
        {                                                                \
            char errstr[MPI_MAX_ERROR_STRING];                           \
            int len;                                                     \
            MPI_Error_string(STATV[i].MPI_ERROR, errstr, &len);          \
            fprintf(stderr, "MPI error in request %d: %s\n", i, errstr); \
            MPI_Abort(COMM, STATV[i].MPI_ERROR);                         \
        }                                                                \
    }


/**********************************************************************/
/*                       MPI FINALIZE                                 */
/**********************************************************************/

#define CCUTILS_MPI_CLEANUP                                            \
    {                                                                  \
        char fname[256];                                               \
        char job_id[64];                                               \
        char *slurm_job_id = getenv("SLURM_JOB_ID");                   \
                                                                       \
        if (slurm_job_id == NULL)                                      \
            sprintf(job_id, "%d", getpid());                           \
        else                                                           \
            sprintf(job_id, "%s", slurm_job_id);                       \
                                                                       \
        /* ---- Buffered print files ---- */                           \
        sprintf(fname, "ccutils_bufprint_%s_rank%d.txt",               \
                job_id, ccutils::mpi_rank());                         \
                                                                       \
        remove(fname); /* safe even if file doesn't exist */           \
                                                                       \
        /* ---- Extend here with additional tmp resources ---- */      \
        /* Example future extension:                                   \
           sprintf(fname, "ccutils_otherprefix_%s_rank%d.tmp", job_id, \
                   ccutils::mpi_rank());                              \
           remove(fname);                                              \
        */                                                             \
                                                                       \
        MPI_Barrier(MPI_COMM_WORLD);                                   \
    }

#endif
