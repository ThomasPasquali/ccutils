#ifndef CCUTILS_MACROS_HPP
#define CCUTILS_MACROS_HPP

#include <cstdio>
#include <cstdlib>
#include <exception>
#include "colors.h"
#include "formats.h"
#include "emitter.hpp"
#include "machine.hpp"
#include "registry.hpp"
#include "sections.hpp"

#ifndef CCUTILS_NO_JSON
#include <nlohmann/json.hpp>
#include <iterator>
#include <string>
#include <vector>

namespace ccutils_json_detail {

inline std::string pretty(const nlohmann::json& value, std::size_t depth,
                          std::size_t line_width) {
  const std::string compact = value.dump();
  const std::size_t current_indent = depth * 2;
  if (current_indent + compact.size() <= line_width ||
      (!value.is_object() && !value.is_array()) || value.empty())
    return compact;

  const std::string indent(current_indent, ' ');
  if (value.is_object()) {
    std::string result = "{\n";
    for (nlohmann::json::const_iterator it = value.begin(); it != value.end(); ++it) {
      result += std::string((depth + 1) * 2, ' ') + nlohmann::json(it.key()).dump() + ": ";
      result += pretty(it.value(), depth + 1, line_width);
      if (std::next(it) != value.end()) result += ',';
      result += '\n';
    }
    return result + indent + '}';
  }

  std::string result = "[\n";
  std::string line((depth + 1) * 2, ' ');
  bool line_has_value = false;
  for (nlohmann::json::const_iterator it = value.begin(); it != value.end(); ++it) {
    const std::string item = it->dump();
    const std::size_t separator = line_has_value ? 2 : 0;
    if (line_has_value && line.size() + separator + item.size() > line_width) {
      result += line + ",\n";
      line.assign((depth + 1) * 2, ' ');
      line_has_value = false;
    }

    if (line_has_value) line += ", ";
    if (!line_has_value && line.size() + item.size() > line_width) {
      result += std::string((depth + 1) * 2, ' ');
      result += pretty(*it, depth + 1, line_width);
      result += std::next(it) != value.end() ? ",\n" : "\n";
      continue;
    }
    line += item;
    line_has_value = true;
  }
  if (line_has_value) result += line + '\n';
  return result + indent + ']';
}

}  // namespace ccutils_json_detail

#if defined(CCUTILS_JSON_PRETTY_PRINT)
#ifndef CCUTILS_JSON_LINE_WIDTH
#define CCUTILS_JSON_LINE_WIDTH 120
#endif
#define CCUTILS_JSON_DUMP(value) \
  (::ccutils_json_detail::pretty((value), 0, CCUTILS_JSON_LINE_WIDTH))
#else
#define CCUTILS_JSON_DUMP(value) ((value).dump())
#endif
#endif

namespace ccutils {
/* Scope based counterpart to CCUTILS_SECTION_DEF/END. A normal destructor
 * closes and prints the section; unwinding emits only an ok:false record. */
class Section {
 public:
  Section(const std::string& name, const char* title)
      : name_(name), ref_(), uncaught_(uncaught_exception_count()), active_(true) {
    start(name_, title);
    ref_ = section(name_);
  }
  ~Section() { if (active_) finish(uncaught_exception_count() > uncaught_); }
  Section(const Section&) = delete;
  Section& operator=(const Section&) = delete;

  SectionRef ref() { return ref_; }
  template <class T> Section& put(const std::string& key, T&& value) {
    ref_.put(key, std::forward<T>(value)); return *this;
  }
  template <class T> Section& put(const std::string& key1, const std::string& key2, T&& value) {
    ref_.put(key1, key2, std::forward<T>(value)); return *this;
  }
  void end() { if (active_) { active_ = false; close(name_, uncaught_exception_count() > uncaught_); } }
  static void start(const std::string& name, const char* title) {
    machine::section_start(name.c_str(), title);
    if (human_v2()) human_section_start(machine::section_path(), title);
    else human_printf(CCUTILS_FMT_SECTION_START, name.c_str(), title);
    human_flush();
  }
  static void end_named(const std::string& name) {
    close(name, uncaught_exception_count() > 0);
  }

 private:
  static int uncaught_exception_count() {
#if __cplusplus >= 201703L
    return std::uncaught_exceptions();
#else
    return std::uncaught_exception() ? 1 : 0;
#endif
  }
  static void close(const std::string& name, bool unwinding) {
    if (unwinding) {
      machine::section_end(name.c_str(), true, false);
      return;
    }
#ifndef CCUTILS_NO_JSON
    SectionRef section_ref = section(name);
    machine::data_object(section_ref.json());
    if (!section_ref.json().empty()) {
      if (human_v2()) {
        for (nlohmann::json::const_iterator it = section_ref.json().begin(); it != section_ref.json().end(); ++it)
          human_printf("%s = %s\n", it.key().c_str(), it.value().dump().c_str());
      } else {
        human_printf(CCUTILS_FMT_GLOBAL_JSON_START, "ccutils_json");
        human_printf("%s\n", CCUTILS_JSON_DUMP(section_ref.json()).c_str());
        human_printf(CCUTILS_FMT_GLOBAL_JSON_END, "ccutils_json");
      }
    }
#endif
    if (human_v2()) human_section_end(machine::section_path());
    else human_printf(CCUTILS_FMT_SECTION_END, name.c_str());
    human_flush();
    machine::section_end(name.c_str(), true, true);
  }
  void finish(bool unwinding) { active_ = false; close(name_, unwinding); }
  std::string name_;
  SectionRef ref_;
  int uncaught_;
  bool active_;
};
}  // namespace ccutils

/* Integer ceiling division. The caller must supply y > 0. */
#define CCUTILS_CEILING(x, y) (((x) + (y) - 1) / (y))

#define CCUTILS_ASSERT(cond, msg, ...) do { \
  if (!(cond)) { \
    std::fprintf(stderr, BRIGHT_RED "Assertion in %s:%d failed: " RESET, __FILE__, __LINE__); \
    std::fprintf(stderr, msg, ##__VA_ARGS__); \
    std::fputc('\n', stderr); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

#define CCUTILS_PRINT_SPLIT(title) do { \
  ccutils::human_printf(BG_BLUE BRIGHT_WHITE "--------------------  %s  --------------------" RESET "\n", title); \
  ccutils::human_flush(); \
} while (0)

#define CCUTILS_DEBUG_PRINT(fmt, ...) \
  do { ccutils::human_printf(BRIGHT_CYAN "[DEBUG] " fmt RESET, ##__VA_ARGS__); } while (0)

#ifndef CCUTILS_NO_JSON
namespace ccutils {
namespace detail {
/* Registers a CCUTILS_JSON_DEF / CCUTILS_MPI_JSON_DEF object under its name. */
class JsonRegistration : public Registration<nlohmann::json*> {
 public:
  JsonRegistration(const char* name, nlohmann::json* object)
      : Registration<nlohmann::json*>(name, object, object) {}
};
}  // namespace detail

/* The live JSON object called `name` (defined with CCUTILS_JSON_DEF somewhere), or a
 * new persistent one. Lets any function add to an object that main() defined. */
inline nlohmann::json& named_json(const std::string& name) {
  typedef detail::NamedRegistry<nlohmann::json*> Registry;
  nlohmann::json* found = nullptr;
  if (Registry::instance().find(name, &found)) return *found;
  std::shared_ptr<nlohmann::json> owner = std::make_shared<nlohmann::json>();
  Registry::instance().add(name, owner.get(), owner.get(), owner);
  return *owner;
}
}  // namespace ccutils
#endif

#ifdef CCUTILS_NO_JSON
#define CCUTILS_SECTION_DEF(name, title) \
  ccutils::Section::start(#name, title);
#define CCUTILS_SECTION_END(name) \
  ccutils::Section::end_named(#name);
#define CCUTILS_SECTION_JSON_PUT(name, key, ...) ((void)0);
#define CCUTILS_SECTION_JSON_SUB_PUT(name, key1, key2, ...) ((void)0);
#define CCUTILS_SUBSECTION_JSON_PUT(section, subsection, key, ...) ((void)0);
#define CCUTILS_SUBSECTION_JSON_SUB_PUT(section, subsection, key1, key2, ...) ((void)0);
#define CCUTILS_JSON_DEF(name)
#define CCUTILS_JSON_USE(name)
#define CCUTILS_JSON_PUT(name, key, ...) ((void)0);
#define CCUTILS_JSON_SUB_PUT(name, key1, key2, ...) ((void)0);
#define CCUTILS_JSON_PRINT(name) ((void)0);
#else
#define CCUTILS_SECTION_DEF(name, title) \
  ccutils::Section::start(#name, title);
#define CCUTILS_SECTION_END(name) ccutils::Section::end_named(#name);
#define CCUTILS_SECTION_JSON_PUT(name, key, ...) \
  (ccutils::section(#name).put(key, (__VA_ARGS__)));
#define CCUTILS_SECTION_JSON_SUB_PUT(name, key1, key2, ...) \
  (ccutils::section(#name).put(key1, key2, (__VA_ARGS__)));

#define CCUTILS_SUBSECTION_JSON_PUT(section_name, subsection, key, ...) \
  (ccutils::section(#subsection).put(key, (__VA_ARGS__)));
#define CCUTILS_SUBSECTION_JSON_SUB_PUT(section_name, subsection, key1, key2, ...) \
  (ccutils::section(#subsection).put(key1, key2, (__VA_ARGS__)));

#define CCUTILS_JSON_DEF(name) \
  nlohmann::json ccutils_json_##name; \
  ccutils::detail::JsonRegistration ccutils_json_reg_##name(#name, &ccutils_json_##name)
/* Binds the JSON object called `name` (defined elsewhere) to a local reference. */
#define CCUTILS_JSON_USE(name) \
  nlohmann::json& ccutils_json_##name = ccutils::named_json(#name);
#define CCUTILS_JSON_PUT(name, key, ...) \
  (ccutils_json_##name[key] = (__VA_ARGS__))
#define CCUTILS_JSON_SUB_PUT(name, key1, key2, ...) \
  (ccutils_json_##name[key1][key2] = (__VA_ARGS__))
#define CCUTILS_JSON_PRINT(name) do { \
  if (!ccutils_json_##name.empty()) { \
    ccutils::machine::data_object(ccutils_json_##name); \
    if (ccutils::human_v2()) { \
      for (nlohmann::json::const_iterator it = ccutils_json_##name.begin(); it != ccutils_json_##name.end(); ++it) \
        ccutils::human_printf("%s = %s\n", it.key().c_str(), it.value().dump().c_str()); \
    } else { \
      ccutils::human_printf(CCUTILS_FMT_GLOBAL_JSON_START, #name); \
      ccutils::human_printf("%s\n", CCUTILS_JSON_DUMP(ccutils_json_##name).c_str()); \
      ccutils::human_printf(CCUTILS_FMT_GLOBAL_JSON_END, #name); \
    } \
    ccutils::human_flush(); \
  } \
} while (0);
#endif

/* Subsections use the parser's nestable section markers and become children
 * of the currently open section. */
#define CCUTILS_SUBSECTION_DEF(section, subsection, title) \
  CCUTILS_SUBSECTION_DEF_IMPL(section, subsection, title)
#ifdef CCUTILS_NO_JSON
#define CCUTILS_SUBSECTION_DEF_IMPL(section, subsection, title) do { \
  ccutils::Section::start(#subsection, title); \
} while (0);
#define CCUTILS_SUBSECTION_END(section, subsection) do { \
  ccutils::Section::end_named(#subsection); \
} while (0);
#else
#define CCUTILS_SUBSECTION_DEF_IMPL(section, subsection, title) \
  ccutils::Section::start(#subsection, title);
#define CCUTILS_SUBSECTION_END(section_name, subsection) ccutils::Section::end_named(#subsection);
#endif

#define CCUTILS_SECTION_TIMER_DEF(section_name, timer_name) \
  CCUTILS_CPU_TIMER_DEF(section_name##_##timer_name)
#define CCUTILS_SECTION_TIMER_USE(section_name, timer_name) \
  CCUTILS_CPU_TIMER_USE(section_name##_##timer_name)
#define CCUTILS_SECTION_TIMER_START(section_name, timer_name) \
  CCUTILS_CPU_TIMER_START(section_name##_##timer_name)
#define CCUTILS_SECTION_TIMER_STOP(section_name, timer_name) \
  CCUTILS_CPU_TIMER_STOP(section_name##_##timer_name)
#define CCUTILS_SECTION_TIMER_PRINT(section_name, timer_name) \
  CCUTILS_TIMER_PRINT(section_name##_##timer_name)

#endif
