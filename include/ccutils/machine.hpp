#ifndef CCUTILS_MACHINE_HPP
#define CCUTILS_MACHINE_HPP

/* Machine-readable stream (JSON Lines, schema version 2).
 *
 * Every record is ONE physical line:
 *   {"v":2,"type":..,"section":..,"scope":..,"rank":..,<fields>,"seq":N}
 * section : full path of the innermost open section ("solve/spmv") or null
 * scope   : "global" (once per run), "rank" (one process) or "comm" (a communicator)
 * rank    : MPI_COMM_WORLD rank for scope "rank", otherwise null (also null when unknown)
 * seq     : per-process counter, starts at 1
 *
 * Records are only produced when CCUTILS_MACHINE_OUT is not "none"; the legacy
 * human output is unaffected. Numbers use the shortest text that reads back as the
 * same value (assumes the "C" numeric locale).
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>
#include <unistd.h>
#include "emitter.hpp"
#include "sections.hpp"

#ifndef CCUTILS_NO_JSON
#include <nlohmann/json.hpp>
#endif

namespace ccutils {
namespace machine {

#ifndef CCUTILS_VERSION
#define CCUTILS_VERSION "1.0.0"
#endif

/* ---- JSON text helpers ------------------------------------------------ */

/* Quoted, escaped JSON string. Invalid UTF-8 bytes become U+FFFD. */
inline std::string quote(const std::string& in) {
  std::string out = "\"";
  const std::size_t n = in.size();
  for (std::size_t i = 0; i < n;) {
    const unsigned char c = static_cast<unsigned char>(in[i]);
    if (c == '"') { out += "\\\""; ++i; }
    else if (c == '\\') { out += "\\\\"; ++i; }
    else if (c == '\n') { out += "\\n"; ++i; }
    else if (c == '\r') { out += "\\r"; ++i; }
    else if (c == '\t') { out += "\\t"; ++i; }
    else if (c < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", c);
      out += buf;
      ++i;
    } else if (c < 0x80) { out += static_cast<char>(c); ++i; }
    else {
      std::size_t len = 0;
      if (c >= 0xC2 && c <= 0xDF) len = 2;
      else if (c >= 0xE0 && c <= 0xEF) len = 3;
      else if (c >= 0xF0 && c <= 0xF4) len = 4;
      bool ok = len > 0 && i + len <= n;
      for (std::size_t k = 1; ok && k < len; ++k)
        ok = (static_cast<unsigned char>(in[i + k]) & 0xC0) == 0x80;
      if (ok) { out.append(in, i, len); i += len; }
      else { out += "\xEF\xBF\xBD"; ++i; }
    }
  }
  out += '"';
  return out;
}

/* Shortest representation that parses back to exactly `v`; null for nan/inf. */
inline std::string format_real(double v) {
  if (!std::isfinite(v)) return "null";
  char buf[40];
  for (int precision = 15; precision <= 17; ++precision) {
    std::snprintf(buf, sizeof(buf), "%.*g", precision, v);
    if (std::strtod(buf, nullptr) == v) break;
  }
  return buf;
}

inline std::string format_real32(float v) {
  if (!std::isfinite(v)) return "null";
  char buf[40];
  for (int precision = 6; precision <= 9; ++precision) {
    std::snprintf(buf, sizeof(buf), "%.*g", precision, static_cast<double>(v));
    if (std::strtof(buf, nullptr) == v) break;
  }
  return buf;
}

/* ---- section context --------------------------------------------------- */

namespace detail {
struct Context {
  std::mutex mutex;
  unsigned long long seq = 0;
  bool run_start_emitted = false;
  bool has_mpi = false;
  std::string mpi_library;
  int mpi_world_size = 1;
  bool mpi_wtime_is_global = false;
  double mpi_wtick_s = 0.0;
};
inline Context& context() {
  static Context instance;
  return instance;
}
inline int& mpi_defer_depth() { static thread_local int depth = 0; return depth; }
inline std::vector<std::string>& deferred_records() { static thread_local std::vector<std::string> records; return records; }
}  // namespace detail

inline void begin_mpi_section_records() { ++detail::mpi_defer_depth(); }
inline void end_mpi_section_records() { if (detail::mpi_defer_depth() > 0) --detail::mpi_defer_depth(); }
inline int suspend_mpi_section_records() { int depth = detail::mpi_defer_depth(); detail::mpi_defer_depth() = 0; return depth; }
inline void resume_mpi_section_records(int depth) { detail::mpi_defer_depth() = depth; }
inline void flush_deferred_records() {
  std::vector<std::string>& records = detail::deferred_records();
  const int depth = detail::mpi_defer_depth();
  detail::mpi_defer_depth() = 0;
  for (std::size_t i=0;i<records.size();++i) machine_write_line(records[i]);
  records.clear();
  detail::mpi_defer_depth() = depth;
}

/* The section stack lives in sections.hpp so that sections can also carry data. */
inline void section_push(const char* name) { ccutils::detail::push_section(name); }
inline void section_pop(const char* name) { ccutils::detail::pop_section(name); }
inline std::string section_path() { return ccutils::detail::section_path(); }

/* ---- record builder ------------------------------------------------------ */

enum class Scope { Global, Rank, Comm };

/* Called by the opt-in MPI headers after MPI_Init; kept MPI-free here to
 * preserve the core headers' ODR boundary. */
inline void set_mpi_facts(const std::string& library, int world_size,
                          bool wtime_is_global, double wtick_s) {
  detail::Context& c = detail::context();
  std::lock_guard<std::mutex> lock(c.mutex);
  c.has_mpi = true;
  c.mpi_library = library;
  c.mpi_world_size = world_size;
  c.mpi_wtime_is_global = wtime_is_global;
  c.mpi_wtick_s = wtick_s;
}

inline void emit_run_start();

class Record {
 public:
  Record(const char* type, Scope scope, bool use_section = true) : type_(type) {
    const std::string path = use_section ? section_path() : std::string();
    initialize(type, scope, path);
  }
  Record(const char* type, Scope scope, const std::string& section) : type_(type) {
    initialize(type, scope, section);
  }

  /* Keys are trusted literals and are not escaped. */
  Record& str(const char* key, const std::string& v) { return add(key, quote(v)); }
  Record& real(const char* key, double v) { return add(key, format_real(v)); }
  Record& real32(const char* key, float v) { return add(key, format_real32(v)); }
  Record& integer(const char* key, long long v) { return add(key, std::to_string(v)); }
  Record& boolean(const char* key, bool v) { return add(key, v ? "true" : "false"); }
  Record& raw(const char* key, const std::string& json) { return add(key, json); }
  Record& rank(int world_rank) {
    const std::string marker = "\"rank\":";
    const std::size_t begin = text_.find(marker);
    if (begin != std::string::npos) {
      const std::size_t found_end = text_.find(',', begin);
      const std::size_t end = found_end == std::string::npos ? text_.size() : found_end;
      text_.replace(begin + marker.size(), end - begin - marker.size(), std::to_string(world_rank));
    }
    return *this;
  }

  void emit() {
    if (type_ != "run_start") {
      const bool capturing = ccutils::detail::machine_capture_active();
      ccutils::detail::machine_capture_active() = false;
      emit_run_start();
      ccutils::detail::machine_capture_active() = capturing;
    }
    unsigned long long n;
    {
      detail::Context& c = detail::context();
      std::lock_guard<std::mutex> lock(c.mutex);
      n = ++c.seq;
    }
    emit_with_seq(n);
  }

  void emit_with_seq(unsigned long long n) {
    text_ += ",\"seq\":" + std::to_string(n) + "}";
    if (detail::mpi_defer_depth() > 0) {
      detail::deferred_records().push_back(text_);
      return;
    }
    machine_write_line(text_);
  }

 private:
  void initialize(const char* type, Scope scope, const std::string& path) {
    text_ = std::string("{\"v\":2,\"type\":") + quote(type) + ",\"section\":" +
            (path.empty() ? std::string("null") : quote(path)) + ",\"scope\":";
    text_ += scope == Scope::Global ? "\"global\"" : scope == Scope::Rank ? "\"rank\"" : "\"comm\"";
    int rank = scope == Scope::Rank ? known_world_rank() : -1;
    text_ += rank >= 0 ? ",\"rank\":" + std::to_string(rank) : std::string(",\"rank\":null");
  }
  Record& add(const char* key, const std::string& value) {
    text_ += std::string(",\"") + key + "\":" + value;
    return *this;
  }
  std::string type_;
  std::string text_;
};

inline void emit_run_start() {
  if (!machine_enabled()) return;
  detail::Context& c = detail::context();
  std::lock_guard<std::mutex> lock(c.mutex);
  if (c.run_start_emitted) return;
  c.run_start_emitted = true;

  char host[256] = "";
  if (gethostname(host, sizeof(host) - 1) != 0) host[0] = '\0';
  const char* job_id = std::getenv("SLURM_JOB_ID");
  Record record("run_start", Scope::Global, false);
  record.str("ccutils_version", CCUTILS_VERSION)
      .str("format", ccutils::format_name());
  if (c.has_mpi) {
    record.raw("mpi", std::string("{\"library\":") + quote(c.mpi_library) +
                          ",\"world_size\":" + std::to_string(c.mpi_world_size) +
                          ",\"wtime_is_global\":" +
                              (c.mpi_wtime_is_global ? "true" : "false") +
                          ",\"wtick_s\":" + format_real(c.mpi_wtick_s) + "}");
  }
  if (job_id != nullptr && *job_id != '\0') record.str("slurm_job_id", job_id);
  record.str("host", host);
  const unsigned long long seq = ++c.seq;
  record.emit_with_seq(seq);
}

/* ---- helpers called from the macros; all are no-ops without a machine sink ---- */

inline void emit_section_start(const char* title) {
  if (!machine_enabled()) return;
  Record("section_start", Scope::Global).str("title", title).emit();
}

/* Single-process variants: also maintain the section stack. */
inline void section_start(const char* name, const char* title) {
  section_push(name);
  emit_section_start(title);
}

inline void section_end(const char* name, bool emit_records = true, bool ok = true) {
  const std::vector<ccutils::detail::ClosedSection> closed =
      ccutils::detail::pop_section(name);
  if (!emit_records || !machine_enabled() || closed.empty()) return;

  const std::string& requested_name = closed.back().name;
  for (std::size_t i = 0; i < closed.size(); ++i) {
    const bool requested_end = (i + 1 == closed.size());
    const bool section_ok = requested_end ? ok : false;
    Record("section_end", Scope::Global, closed[i].path).boolean("ok", section_ok).emit();
    if (!requested_end) {
      const std::string message = "section '" + closed[i].name +
          "' was closed when outer section '" + requested_name +
          "' ended before its end macro";
      Record("warning", Scope::Global, closed[i].path).str("message", message).emit();
    }
  }
}

/* Exception-safe local close used by MPI RAII destructors: never collective. */
inline void section_end_local(const char* name, bool ok = false) {
  const std::vector<ccutils::detail::ClosedSection> closed =
      ccutils::detail::pop_section(name);
  for (std::size_t i = 0; i < closed.size(); ++i)
    Record("section_end", Scope::Rank, closed[i].path).boolean("ok", ok && i + 1 == closed.size()).emit();
}

#ifndef CCUTILS_NO_JSON
/* One "data" record per top-level key of a JSON object. */
inline void data_object(const nlohmann::json& object) {
  if (!machine_enabled() || !object.is_object()) return;
  for (nlohmann::json::const_iterator it = object.begin(); it != object.end(); ++it) {
    Record("data", Scope::Global)
        .str("key", it.key())
        .raw("value", it.value().dump(-1, ' ', false, nlohmann::json::error_handler_t::replace))
        .emit();
  }
}
inline void data_object(const nlohmann::json& object, Scope scope) {
  if (!machine_enabled() || !object.is_object()) return;
  for (nlohmann::json::const_iterator it = object.begin(); it != object.end(); ++it)
    Record("data", scope).str("key", it.key())
        .raw("value", it.value().dump(-1, ' ', false, nlohmann::json::error_handler_t::replace)).emit();
}
#endif

inline void timer_summary(const char* group, const char* name, std::size_t n, double avg,
                          double sd, double min, double max, double sum) {
  if (!machine_enabled()) return;
  Record("timer", Scope::Rank)
      .str("name", name).str("group", group).str("unit", "ms").str("clock", "steady")
      .integer("n", static_cast<long long>(n))
      .real("avg", avg).real("sd", sd).real("min", min).real("max", max).real("sum", sum)
      .emit();
}

inline void timer_sample(const char* group, const char* name, std::size_t index, double value,
                         bool last) {
  if (!machine_enabled()) return;
  Record("timer_sample", Scope::Rank)
      .str("name", name).str("group", group).str("unit", "ms").str("clock", "steady")
      .integer("index", static_cast<long long>(index)).real("value", value)
      .boolean("last", last)
      .emit();
}

inline void timer_agg(const std::string& name, const char* op, double value) {
  if (!machine_enabled()) return;
  Record("timer_agg", Scope::Rank)
      .str("name", name).str("op", op).str("unit", "ms").str("clock", "steady")
      .real("value", value)
      .emit();
}

inline unsigned long long next_text_call(Scope scope) {
  static thread_local unsigned long long rank_call = 0;
  static thread_local unsigned long long global_call = 0;
  return scope == Scope::Rank ? ++rank_call : ++global_call;
}
inline std::string base64(const std::string& input) {
  static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  for (std::size_t i = 0; i < input.size(); i += 3) {
    const unsigned int a = static_cast<unsigned char>(input[i]);
    const unsigned int b = i + 1 < input.size() ? static_cast<unsigned char>(input[i + 1]) : 0;
    const unsigned int c = i + 2 < input.size() ? static_cast<unsigned char>(input[i + 2]) : 0;
    out += alphabet[a >> 2]; out += alphabet[((a & 3) << 4) | (b >> 4)];
    out += i + 1 < input.size() ? alphabet[((b & 15) << 2) | (c >> 6)] : '=';
    out += i + 2 < input.size() ? alphabet[c & 63] : '=';
  }
  return out;
}
inline bool valid_utf8(const std::string& s) {
  for (std::size_t i = 0; i < s.size();) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) { ++i; continue; }
    std::size_t n = c >= 0xC2 && c <= 0xDF ? 2 : c >= 0xE0 && c <= 0xEF ? 3 : c >= 0xF0 && c <= 0xF4 ? 4 : 0;
    if (!n || i + n > s.size()) return false;
    for (std::size_t k = 1; k < n; ++k) if ((static_cast<unsigned char>(s[i+k]) & 0xC0) != 0x80) return false;
    if (n == 3 && ((c == 0xE0 && static_cast<unsigned char>(s[i+1]) < 0xA0) || (c == 0xED && static_cast<unsigned char>(s[i+1]) >= 0xA0))) return false;
    if (n == 4 && ((c == 0xF0 && static_cast<unsigned char>(s[i+1]) < 0x90) || (c == 0xF4 && static_cast<unsigned char>(s[i+1]) >= 0x90))) return false;
    i += n;
  }
  return true;
}
inline void text_records(const char* print, const std::string& text, Scope scope,
                         const char* comm = nullptr) {
  const unsigned long long call = next_text_call(scope);
  if (!machine_enabled()) return;
  const bool trailing = !text.empty() && text[text.size()-1] == '\n';
  const bool utf8 = valid_utf8(text);
  std::vector<std::string> chunks;
  std::size_t begin = 0, lines = 0;
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '\n' && ++lines >= 1000) { chunks.push_back(text.substr(begin, i + 1 - begin)); begin = i + 1; lines = 0; }
    if (i + 1 - begin >= 65536) { chunks.push_back(text.substr(begin, i + 1 - begin)); begin = i + 1; lines = 0; }
  }
  if (begin < text.size() || chunks.empty()) chunks.push_back(text.substr(begin));
  for (std::size_t p = 0; p < chunks.size(); ++p) {
    const std::string& chunk = chunks[p];
    Record record("text", scope);
    if (print) record.str("print", print); else record.raw("print", "null");
    if (comm && *comm) record.str("comm", comm);
    record.integer("call", static_cast<long long>(call)).integer("part", static_cast<long long>(p))
      .boolean("last", p + 1 == chunks.size()).boolean("trailing_newline", trailing);
    if (utf8) {
      std::string values = "["; std::size_t pos = 0; bool first = true;
      while (pos < chunk.size()) {
        std::size_t end = chunk.find('\n', pos);
        if (end == std::string::npos) end = chunk.size();
        std::string line = chunk.substr(pos, end-pos);
        if (!first) values += ',';
        first = false;
        values += quote(line);
        pos = end == chunk.size() ? end : end + 1;
      }
      record.raw("lines", values + "]");
      /* Preserve exact chunk boundaries across arbitrary byte splits. */
      record.str("chunk", base64(chunk));
    } else record.str("encoding", "base64").str("data", base64(chunk));
    record.emit();
  }
}

/* Report of a timer over a communicator (MPI timers). `root` is a world rank. */
inline void timer_comm(const char* group, const char* name, const char* comm_name,
                       int comm_size, int root, std::size_t n, double avg, double sd,
                       double min, double max, double sum) {
  if (!machine_enabled()) return;
  Record("timer", Scope::Comm)
      .str("name", name).str("group", group).str("unit", "ms").str("clock", "mpi_wtime")
      .raw("comm", std::string("{\"name\":") + quote(comm_name) +
                       ",\"size\":" + std::to_string(comm_size) +
                       ",\"root\":" + std::to_string(root) + "}")
      .integer("n", static_cast<long long>(n))
      .real("avg", avg).real("sd", sd).real("min", min).real("max", max).real("sum", sum)
      .emit();
}

}  // namespace machine

/* Configure explicitly and emit run metadata immediately when a machine sink
 * is enabled. This is optional; the first machine record does the same lazily. */
inline void init(const Config& cfg) {
  configure(cfg);
  machine::emit_run_start();
}

}  // namespace ccutils

#endif
