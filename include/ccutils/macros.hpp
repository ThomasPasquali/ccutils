#ifndef CCUTILS_MACROS_HPP
#define CCUTILS_MACROS_HPP

#include <cstdio>
#include <cstdlib>
#include "colors.h"
#include "formats.h"

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
  std::printf(BG_BLUE BRIGHT_WHITE "--------------------  %s  --------------------" RESET "\n", title); \
  std::fflush(stdout); \
} while (0)

#define CCUTILS_DEBUG_PRINT(fmt, ...) \
  do { std::printf(BRIGHT_CYAN "[DEBUG] " fmt RESET, ##__VA_ARGS__); } while (0)

#ifdef CCUTILS_NO_JSON
#define CCUTILS_SECTION_DEF(name, title) \
  std::printf(CCUTILS_FMT_SECTION_START, #name, title);
#define CCUTILS_SECTION_END(name) \
  std::printf(CCUTILS_FMT_SECTION_END, #name);
#define CCUTILS_SECTION_JSON_PUT(name, key, ...) ((void)0);
#define CCUTILS_SECTION_JSON_SUB_PUT(name, key1, key2, ...) ((void)0);
#define CCUTILS_SUBSECTION_JSON_PUT(section, subsection, key, ...) ((void)0);
#define CCUTILS_SUBSECTION_JSON_SUB_PUT(section, subsection, key1, key2, ...) ((void)0);
#define CCUTILS_JSON_DEF(name)
#define CCUTILS_JSON_PUT(name, key, ...) ((void)0);
#define CCUTILS_JSON_SUB_PUT(name, key1, key2, ...) ((void)0);
#define CCUTILS_JSON_PRINT(name) ((void)0);
#else
#define CCUTILS_SECTION_DEF(name, title) \
  nlohmann::json ccutils_section_json_##name; \
  do { std::printf(CCUTILS_FMT_SECTION_START, #name, title); std::fflush(stdout); } while (0);
#define CCUTILS_SECTION_END(name) do { \
  if (!ccutils_section_json_##name.empty()) { \
    std::printf(CCUTILS_FMT_GLOBAL_JSON_START, "ccutils_json"); \
    std::printf("%s\n", CCUTILS_JSON_DUMP(ccutils_section_json_##name).c_str()); \
    std::printf(CCUTILS_FMT_GLOBAL_JSON_END, "ccutils_json"); \
  } \
  std::printf(CCUTILS_FMT_SECTION_END, #name); \
  std::fflush(stdout); \
} while (0);
#define CCUTILS_SECTION_JSON_PUT(name, key, ...) \
  (ccutils_section_json_##name[key] = (__VA_ARGS__));
#define CCUTILS_SECTION_JSON_SUB_PUT(name, key1, key2, ...) \
  (ccutils_section_json_##name[key1][key2] = (__VA_ARGS__));

#define CCUTILS_SUBSECTION_JSON_PUT(section, subsection, key, ...) \
  (ccutils_subsection_json_##section##_##subsection[key] = (__VA_ARGS__));
#define CCUTILS_SUBSECTION_JSON_SUB_PUT(section, subsection, key1, key2, ...) \
  (ccutils_subsection_json_##section##_##subsection[key1][key2] = (__VA_ARGS__));

#define CCUTILS_JSON_DEF(name) nlohmann::json ccutils_json_##name
#define CCUTILS_JSON_PUT(name, key, ...) \
  (ccutils_json_##name[key] = (__VA_ARGS__))
#define CCUTILS_JSON_SUB_PUT(name, key1, key2, ...) \
  (ccutils_json_##name[key1][key2] = (__VA_ARGS__))
#define CCUTILS_JSON_PRINT(name) do { \
  if (!ccutils_json_##name.empty()) { \
    std::printf(CCUTILS_FMT_GLOBAL_JSON_START, #name); \
    std::printf("%s\n", CCUTILS_JSON_DUMP(ccutils_json_##name).c_str()); \
    std::printf(CCUTILS_FMT_GLOBAL_JSON_END, #name); \
    std::fflush(stdout); \
  } \
} while (0);
#endif

/* Subsections use the parser's nestable section markers and become children
 * of the currently open section. */
#define CCUTILS_SUBSECTION_DEF(section, subsection, title) \
  CCUTILS_SUBSECTION_DEF_IMPL(section, subsection, title)
#ifdef CCUTILS_NO_JSON
#define CCUTILS_SUBSECTION_DEF_IMPL(section, subsection, title) do { \
  std::printf(CCUTILS_FMT_SECTION_START, #subsection, title); \
  std::fflush(stdout); \
} while (0);
#define CCUTILS_SUBSECTION_END(section, subsection) do { \
  std::printf(CCUTILS_FMT_SECTION_END, #subsection); \
  std::fflush(stdout); \
} while (0);
#else
#define CCUTILS_SUBSECTION_DEF_IMPL(section, subsection, title) \
  nlohmann::json ccutils_subsection_json_##section##_##subsection; \
  do { std::printf(CCUTILS_FMT_SECTION_START, #subsection, title); \
       std::fflush(stdout); } while (0);
#define CCUTILS_SUBSECTION_END(section, subsection) do { \
  if (!ccutils_subsection_json_##section##_##subsection.empty()) { \
    std::printf(CCUTILS_FMT_GLOBAL_JSON_START, "ccutils_json"); \
    std::printf("%s\n", CCUTILS_JSON_DUMP(ccutils_subsection_json_##section##_##subsection).c_str()); \
    std::printf(CCUTILS_FMT_GLOBAL_JSON_END, "ccutils_json"); \
  } \
  std::printf(CCUTILS_FMT_SECTION_END, #subsection); \
  std::fflush(stdout); \
} while (0);
#endif

#define CCUTILS_SECTION_TIMER_DEF(section_name, timer_name) \
  CCUTILS_CPU_TIMER_DEF(section_name##_##timer_name)
#define CCUTILS_SECTION_TIMER_START(section_name, timer_name) \
  CCUTILS_CPU_TIMER_START(section_name##_##timer_name)
#define CCUTILS_SECTION_TIMER_STOP(section_name, timer_name) \
  CCUTILS_CPU_TIMER_STOP(section_name##_##timer_name)
#define CCUTILS_SECTION_TIMER_PRINT(section_name, timer_name) \
  CCUTILS_TIMER_PRINT(section_name##_##timer_name)

#endif
