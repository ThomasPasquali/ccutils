#ifndef CCUTILS_EMITTER_HPP
#define CCUTILS_EMITTER_HPP

/* Output sinks.
 *
 * Everything ccutils prints for humans goes through ccutils::human_*(). By
 * default that is the process' stdout (same FILE*, same buffer, so the bytes and
 * their ordering relative to the program's own printf calls are unchanged).
 *
 * Configuration (environment variable, or ccutils::configure()/ccutils::init()):
 *
 *   CCUTILS_HUMAN_OUT    stdout | stderr | none | file:<path> | append:<path>
 *   CCUTILS_MACHINE_OUT  none | stdout | stderr | inline | file:<path> | append:<path>
 *   CCUTILS_FORMAT       v2 | legacy (default: v2; legacy available for migration)
 *
 * Path templates: %r rank, %p pid, %j SLURM_JOB_ID (or pid), %% a percent sign.
 *
 * Who may open a file: a path WITHOUT %r is opened by rank 0 only; any other rank
 * that writes (non-collective prints such as a CPU timer or an MPI timer report on
 * a communicator root) uses the sidecar "<path>.r<rank>", so that several processes
 * never write the same file. A path WITH %r gives every rank its own file.
 *
 * This header has no MPI dependency (an MPI-dependent inline function would be an
 * ODR violation between MPI and non-MPI translation units). The rank comes from
 * ccutils::set_world_rank() (called by the MPI macros), else from the launcher's
 * environment (OMPI_COMM_WORLD_RANK, PMI_RANK, PMIX_RANK, MV2_COMM_WORLD_RANK,
 * SLURM_PROCID), else it is 0.
 */

#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define CCUTILS_PRINTF_FORMAT(fmt_idx, first_arg) \
  __attribute__((format(printf, fmt_idx, first_arg)))
#else
#define CCUTILS_PRINTF_FORMAT(fmt_idx, first_arg)
#endif

namespace ccutils {

enum class SinkKind { None, Stdout, Stderr, File, Append, Inline };

struct SinkSpec {
  SinkKind kind = SinkKind::Stdout;
  std::string path;
};

/* Empty strings mean "use the environment variable, then the default". */
struct Config {
  std::string format;
  std::string human_out;
  std::string machine_out;
};

/* Parses "stdout", "stderr", "none", "inline" (machine stream only),
 * "file:<path>" and "append:<path>". */
inline bool parse_sink(const std::string& text, bool allow_inline, SinkSpec* out,
                       std::string* error) {
  SinkSpec spec;
  if (text == "stdout") {
    spec.kind = SinkKind::Stdout;
  } else if (text == "stderr") {
    spec.kind = SinkKind::Stderr;
  } else if (text == "none") {
    spec.kind = SinkKind::None;
  } else if (text == "inline" && allow_inline) {
    spec.kind = SinkKind::Inline;
  } else if (text.compare(0, 5, "file:") == 0 || text.compare(0, 7, "append:") == 0) {
    const bool is_file = text[0] == 'f';
    spec.kind = is_file ? SinkKind::File : SinkKind::Append;
    spec.path = text.substr(is_file ? 5 : 7);
    if (spec.path.empty()) {
      if (error) *error = "missing path after '" + text + "'";
      return false;
    }
  } else {
    if (error) *error = "unknown sink '" + text + "'";
    return false;
  }
  *out = spec;
  return true;
}

/* Expands %r, %p, %j and %%. Sets *uses_rank if %r was present. */
inline std::string expand_path(const std::string& tmpl, int rank, bool* uses_rank) {
  std::string out;
  if (uses_rank) *uses_rank = false;
  for (std::size_t i = 0; i < tmpl.size(); ++i) {
    if (tmpl[i] != '%' || i + 1 == tmpl.size()) {
      out += tmpl[i];
      continue;
    }
    const char c = tmpl[++i];
    if (c == 'r') {
      out += std::to_string(rank);
      if (uses_rank) *uses_rank = true;
    } else if (c == 'p') {
      out += std::to_string(static_cast<long>(getpid()));
    } else if (c == 'j') {
      const char* job = std::getenv("SLURM_JOB_ID");
      out += (job && *job) ? std::string(job) : std::to_string(static_cast<long>(getpid()));
    } else if (c == '%') {
      out += '%';
    } else {
      out += '%';
      out += c;
    }
  }
  return out;
}

namespace detail {

/* Rank announced by the launcher, or -1. */
inline int launcher_rank() {
  static const char* const names[] = {"OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK",
                                      "MV2_COMM_WORLD_RANK", "SLURM_PROCID", nullptr};
  for (const char* const* name = names; *name != nullptr; ++name) {
    const char* value = std::getenv(*name);
    if (value == nullptr || *value == '\0') continue;
    char* end = nullptr;
    const long rank = std::strtol(value, &end, 10);
    if (*end == '\0' && rank >= 0) return static_cast<int>(rank);
  }
  return -1;
}

class Sink {
 public:
  void set(const SinkSpec& spec) {
    close();
    spec_ = spec;
    warned_ = false;
    const char* no_color = std::getenv("NO_COLOR");
    const char* color = std::getenv("CCUTILS_COLOR");
    if (no_color != nullptr && *no_color != '\0') {
      color_mode_ = ColorMode::Never;
    } else if (color != nullptr && std::strcmp(color, "never") == 0) {
      color_mode_ = ColorMode::Never;
    } else if (color != nullptr && std::strcmp(color, "auto") == 0) {
      color_mode_ = ColorMode::Auto;
    } else if (color != nullptr && std::strcmp(color, "always") == 0) {
      color_mode_ = ColorMode::Always;
    } else if (color == nullptr) {
      color_mode_ = ColorMode::Auto;
    } else {
      color_mode_ = ColorMode::Auto;
    }
  }

  void write(const char* data, std::size_t size, int rank) {
    FILE* stream = resolve(rank);
    if (stream == nullptr || size == 0) return;
    if (strip_color(stream)) {
      write_without_color(data, size, stream);
    } else {
      std::fwrite(data, 1, size, stream);
    }
  }

  void flush() {
    if (file_ != nullptr) {
      std::fflush(file_);
    } else if (spec_.kind == SinkKind::Stdout || spec_.kind == SinkKind::Inline) {
      std::fflush(stdout);
    } else if (spec_.kind == SinkKind::Stderr) {
      std::fflush(stderr);
    }
  }

  void close() {
    if (file_ != nullptr) {
      std::fclose(file_);
      file_ = nullptr;
    }
  }

  SinkKind kind() const { return spec_.kind; }

 private:
  enum class ColorMode { Always, Never, Auto };

  bool strip_color(FILE* stream) const {
    if (color_mode_ == ColorMode::Never) return true;
    if (color_mode_ == ColorMode::Always) return false;
    return ::isatty(::fileno(stream)) != 1;
  }

  static void write_without_color(const char* data, std::size_t size, FILE* stream) {
    std::size_t start = 0;
    std::size_t i = 0;
    while (i < size) {
      if (static_cast<unsigned char>(data[i]) != 0x1b || i + 1 >= size || data[i + 1] != '[') {
        ++i;
        continue;
      }
      std::size_t end = i + 2;
      while (end < size && ((data[end] >= '0' && data[end] <= '9') || data[end] == ';')) ++end;
      if (end < size && data[end] == 'm') {
        std::fwrite(data + start, 1, i - start, stream);
        i = end + 1;
        start = i;
      } else {
        ++i;
      }
    }
    if (start < size) std::fwrite(data + start, 1, size - start, stream);
  }

  FILE* resolve(int rank) {
    switch (spec_.kind) {
      case SinkKind::None: return nullptr;
      case SinkKind::Stdout:
      case SinkKind::Inline: return stdout;
      case SinkKind::Stderr: return stderr;
      case SinkKind::File:
      case SinkKind::Append: break;
    }
    if (file_ != nullptr) return file_;

    bool uses_rank = false;
    std::string path = expand_path(spec_.path, rank, &uses_rank);
    if (!uses_rank && rank > 0) path += ".r" + std::to_string(rank);
    file_ = std::fopen(path.c_str(), spec_.kind == SinkKind::File ? "w" : "a");
    if (file_ == nullptr) {
      if (!warned_) {
        std::fprintf(stderr, "ccutils: cannot open '%s' for output (%s); using stdout instead\n",
                     path.c_str(), std::strerror(errno));
        warned_ = true;
      }
      spec_.kind = SinkKind::Stdout;
      return stdout;
    }
    return file_;
  }

  SinkSpec spec_;
  FILE* file_ = nullptr;
  bool warned_ = false;
  ColorMode color_mode_ = ColorMode::Auto;
};

struct State {
  std::mutex mutex;
  bool configured = false;
  int rank = -1;  // set explicitly by the MPI macros
  std::string format = "v2";
  Sink human;
  Sink machine;

  ~State() {
    human.close();
    machine.close();
  }
};

inline std::vector<std::string>& machine_capture() {
  static thread_local std::vector<std::string> lines;
  return lines;
}
inline bool& machine_capture_active() { static thread_local bool active = false; return active; }
inline std::string& human_capture() { static thread_local std::string text; return text; }
inline bool& human_capture_active() { static thread_local bool active = false; return active; }

inline State& state() {
  static State instance;
  return instance;
}

inline int effective_rank(const State& s) {
  if (s.rank >= 0) return s.rank;
  const int from_launcher = launcher_rank();
  return from_launcher >= 0 ? from_launcher : 0;
}

inline std::string pick(const std::string& explicit_value, const char* env_name,
                        const char* fallback) {
  if (!explicit_value.empty()) return explicit_value;
  const char* env = std::getenv(env_name);
  return (env != nullptr && *env != '\0') ? std::string(env) : std::string(fallback);
}

inline void warn(const State& s, const std::string& message) {
  if (effective_rank(s) == 0) std::fprintf(stderr, "ccutils: %s\n", message.c_str());
}

/* Caller holds the mutex. */
inline void apply(State& s, const Config& cfg) {
  s.human.close();
  s.machine.close();

  // 1.0 defaults to v2. Define CCUTILS_FORMAT_LEGACY for source-compatible
  // legacy output unless the runtime configuration explicitly selects a format.
#ifdef CCUTILS_FORMAT_LEGACY
  const char* default_format = "legacy";
#else
  const char* default_format = "v2";
#endif
  const std::string format = pick(cfg.format, "CCUTILS_FORMAT", default_format);
  if (format != "v2" && format != "legacy") {
    warn(s, "unknown CCUTILS_FORMAT '" + format + "'; using '" + default_format + "'");
  }
  s.format = (format == "v2" || format == "legacy") ? format : std::string(default_format);

  SinkSpec spec;
  std::string error;
  if (parse_sink(pick(cfg.human_out, "CCUTILS_HUMAN_OUT", "stdout"), false, &spec, &error)) {
    s.human.set(spec);
  } else {
    warn(s, "CCUTILS_HUMAN_OUT: " + error + "; using stdout");
    s.human.set(SinkSpec());
  }

  if (parse_sink(pick(cfg.machine_out, "CCUTILS_MACHINE_OUT", "none"), true, &spec, &error)) {
    s.machine.set(spec);
  } else {
    warn(s, "CCUTILS_MACHINE_OUT: " + error + "; machine output disabled");
    SinkSpec none;
    none.kind = SinkKind::None;
    s.machine.set(none);
  }
  s.configured = true;
}

inline State& locked_state(std::unique_lock<std::mutex>& lock) {
  State& s = state();
  lock = std::unique_lock<std::mutex>(s.mutex);
  if (!s.configured) apply(s, Config());
  return s;
}

}  // namespace detail

/* Applies an explicit configuration (overrides the environment). Safe to call
 * before or after output has been produced; open files are closed and re-opened. */
inline void configure(const Config& cfg) {
  detail::State& s = detail::state();
  std::lock_guard<std::mutex> lock(s.mutex);
  detail::apply(s, cfg);
}

/* Announces this process' MPI_COMM_WORLD rank (called by the MPI macros). */
inline void set_world_rank(int rank) {
  detail::State& s = detail::state();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.rank = rank;
}

/* Rank announced by the MPI macros or the launcher, or -1 if unknown (a plain
 * non-MPI run). Machine records use null for an unknown rank. */
inline int known_world_rank() {
  detail::State& s = detail::state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return s.rank >= 0 ? s.rank : detail::launcher_rank();
}

/* Rank announced explicitly through set_world_rank(), or -1. */
inline int announced_world_rank() {
  detail::State& s = detail::state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return s.rank;
}

/* Prints `message` on stderr (rank 0 only) the first time `key` is seen. */
inline void warn_once(const std::string& key, const std::string& message) {
  detail::State& s = detail::state();
  std::lock_guard<std::mutex> lock(s.mutex);
  static std::set<std::string> seen;
  if (!seen.insert(key).second) return;
  detail::warn(s, message);
}

inline const char* format_name() {
  static thread_local std::string selected;
  detail::State& s = detail::state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!s.configured) detail::apply(s, Config());
  selected = s.format;
  return selected.c_str();
}
inline void human_printf(const char* fmt, ...) CCUTILS_PRINTF_FORMAT(1, 2);
inline void human_write(const char* data, std::size_t size);
inline bool human_v2() { return std::strcmp(format_name(), "v2") == 0; }
inline void human_section_start(const std::string& path, const std::string& title) {
  const std::string line = "== " + path + " :: " + title + " ==\n";
  human_write(line.data(), line.size());
}
inline void human_section_end(const std::string& path) {
  const std::string line = "== end " + path + " ==\n";
  human_write(line.data(), line.size());
}
inline void human_v2_text_block(const std::string& name, int rank, const std::string& bytes) {
  std::size_t begin = 0;
  while (begin < bytes.size()) {
    std::size_t end = bytes.find('\n', begin);
    if (end == std::string::npos) end = bytes.size();
    std::string line = "[r" + std::to_string(rank) + "] " + bytes.substr(begin, end - begin) + "\n";
    human_write(line.data(), line.size());
    begin = end == bytes.size() ? end : end + 1;
  }
}
inline void human_v2_timer(const char* title, const char* name, std::size_t n,
                           double avg, double sd, double min, double max, double sum,
                           const char* location = nullptr) {
  const char* precision_text = std::getenv("CCUTILS_HUMAN_PRECISION");
  int precision = precision_text ? std::atoi(precision_text) : 3;
  if (precision < 0 || precision > 12) precision = 3;
  char default_location[32];
  if (!location) {
    std::snprintf(default_location, sizeof(default_location), "rank %d",
                  known_world_rank() < 0 ? 0 : known_world_rank());
    location = default_location;
  }
  human_printf("timers [ms]  (%s)\n  name   n     avg     sd     min     max     sum\n", location);
  human_printf("  %s   %zu  %.*f  %.*f  %.*f  %.*f  %.*f\n\n", name, n,
               precision, avg, precision, sd, precision, min, precision, max, precision, sum);
  (void)title;
}
inline void human_v2_sample(const char* name, std::size_t index, double value) {
  const char* precision_text = std::getenv("CCUTILS_HUMAN_PRECISION");
  int precision = precision_text ? std::atoi(precision_text) : 3;
  if (precision < 0 || precision > 12) precision = 3;
  human_printf("timers [ms]  (rank %d)\n  name   n     avg      sd      min      max      sum\n",
               known_world_rank() < 0 ? 0 : known_world_rank());
  human_printf("  %s   1  %.*f    %.*f   %.*f   %.*f   %.*f\n\n", name,
               precision, value, precision, 0.0, precision, value, precision, value, precision, value);
  (void)index;
}

inline void human_write(const char* data, std::size_t size) {
  if (detail::human_capture_active()) { detail::human_capture().append(data, size); return; }
  std::unique_lock<std::mutex> lock;
  detail::State& s = detail::locked_state(lock);
  s.human.write(data, size, detail::effective_rank(s));
}

inline void human_capture_begin() { detail::human_capture().clear(); detail::human_capture_active() = true; }
inline std::string human_capture_end() {
  detail::human_capture_active() = false;
  const std::string text = detail::human_capture();
  if (!text.empty()) human_write(text.data(), text.size());
  return text;
}

inline void human_putc(char c) { human_write(&c, 1); }

inline void human_flush() {
  std::unique_lock<std::mutex> lock;
  detail::State& s = detail::locked_state(lock);
  s.human.flush();
}

inline void human_printf(const char* fmt, ...) {
  va_list args, copy;
  va_start(args, fmt);
  va_copy(copy, args);
  char small[512];
  const int needed = std::vsnprintf(small, sizeof(small), fmt, args);
  va_end(args);
  if (needed >= 0) {
    if (static_cast<std::size_t>(needed) < sizeof(small)) {
      human_write(small, static_cast<std::size_t>(needed));
    } else {
      std::string big(static_cast<std::size_t>(needed) + 1, '\0');
      std::vsnprintf(&big[0], big.size(), fmt, copy);
      human_write(big.data(), static_cast<std::size_t>(needed));
    }
  }
  va_end(copy);
}

/* One complete machine record (no trailing newline in `line`). With the
 * "inline" sink it is written to the human stream prefixed by "@ccutils ". */
inline void machine_write_line(const std::string& line) {
  if (detail::machine_capture_active()) {
    detail::machine_capture().push_back(line);
    return;
  }
  std::unique_lock<std::mutex> lock;
  detail::State& s = detail::locked_state(lock);
  const int rank = detail::effective_rank(s);
  if (s.machine.kind() == SinkKind::Inline) {
    const std::string text = "@ccutils " + line + "\n";
    s.human.write(text.data(), text.size(), rank);
  } else {
    const std::string text = line + "\n";
    s.machine.write(text.data(), text.size(), rank);
  }
}

inline void machine_write_line_for_rank(const std::string& line, int rank) {
  std::unique_lock<std::mutex> lock;
  detail::State& s = detail::locked_state(lock);
  const std::string text = line + "\n";
  if (s.machine.kind() == SinkKind::Inline) {
    const std::string inline_text = "@ccutils " + text;
    s.human.write(inline_text.data(), inline_text.size(), rank);
  } else s.machine.write(text.data(), text.size(), rank);
}

inline void machine_capture_begin() {
  detail::machine_capture().clear();
  detail::machine_capture_active() = true;
}
inline std::vector<std::string> machine_capture_end() {
  detail::machine_capture_active() = false;
  return detail::machine_capture();
}

inline bool machine_enabled() {
  std::unique_lock<std::mutex> lock;
  detail::State& s = detail::locked_state(lock);
  return s.machine.kind() != SinkKind::None;
}

inline bool human_enabled() {
  std::unique_lock<std::mutex> lock;
  detail::State& s = detail::locked_state(lock);
  return s.human.kind() != SinkKind::None;
}

}  // namespace ccutils

#endif
