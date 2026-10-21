#ifndef CCUTILS_SECTIONS_HPP
#define CCUTILS_SECTIONS_HPP

/* Open sections, addressable by name from anywhere in the program.
 *
 * Opening a section (CCUTILS_SECTION_DEF, ...) pushes it on a process-wide stack;
 * closing it pops it. Any function, in any file, can then update the section that
 * is currently open under a given name:
 *
 *     CCUTILS_SECTION_JSON_PUT(run, "iterations", n);     // macro, by name
 *     ccutils::section("run").put("iterations", n);       // object API
 *     ccutils::current_section().put("note", "...");      // innermost open section
 *
 * If several open sections share a name, the innermost one wins. Updating a section
 * that is not open is ignored with a single warning on stderr.
 */

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "emitter.hpp"

#ifndef CCUTILS_NO_JSON
#include <nlohmann/json.hpp>
#endif

namespace ccutils {
namespace detail {

struct SectionData {
  std::string name;
#ifndef CCUTILS_NO_JSON
  nlohmann::json json;         /* the section's data (rank-local data in MPI programs) */
  nlohmann::json global_json;  /* MPI: data written once by rank 0 */
#endif
};

struct SectionStack {
  std::mutex mutex;
  std::vector<std::shared_ptr<SectionData> > open;
};

struct ClosedSection {
  std::string name;
  std::string path;
};

inline SectionStack& section_stack() {
  static SectionStack stack;
  return stack;
}

inline std::shared_ptr<SectionData> push_section(const char* name) {
  SectionStack& s = section_stack();
  std::lock_guard<std::mutex> lock(s.mutex);
  std::shared_ptr<SectionData> data = std::make_shared<SectionData>();
  data->name = name;
  s.open.push_back(data);
  return data;
}

/* Closes the innermost section called `name` and everything opened inside it. */
inline std::vector<ClosedSection> pop_section(const char* name) {
  SectionStack& s = section_stack();
  std::lock_guard<std::mutex> lock(s.mutex);
  for (std::size_t i = s.open.size(); i > 0; --i) {
    if (s.open[i - 1]->name == name) {
      std::vector<ClosedSection> closed;
      for (std::size_t end = s.open.size(); end >= i; --end) {
        std::string path;
        for (std::size_t part = 0; part < end; ++part) {
          if (part != 0) path += '/';
          path += s.open[part]->name;
        }
        ClosedSection item;
        item.name = s.open[end - 1]->name;
        item.path = path;
        closed.push_back(item);
      }
      s.open.resize(i - 1);
      return closed;
    }
  }
  return std::vector<ClosedSection>();
}

inline std::shared_ptr<SectionData> find_section(const std::string& name) {
  SectionStack& s = section_stack();
  std::lock_guard<std::mutex> lock(s.mutex);
  for (std::size_t i = s.open.size(); i > 0; --i)
    if (s.open[i - 1]->name == name) return s.open[i - 1];
  return std::shared_ptr<SectionData>();
}

inline std::shared_ptr<SectionData> innermost_section() {
  SectionStack& s = section_stack();
  std::lock_guard<std::mutex> lock(s.mutex);
  return s.open.empty() ? std::shared_ptr<SectionData>() : s.open.back();
}

inline std::string section_path() {
  SectionStack& s = section_stack();
  std::lock_guard<std::mutex> lock(s.mutex);
  std::string path;
  for (std::size_t i = 0; i < s.open.size(); ++i) {
    if (i > 0) path += '/';
    path += s.open[i]->name;
  }
  return path;
}

}  // namespace detail

/* Handle to an open section. Cheap to copy; stays valid after the section closes
 * (updates are then simply never printed). */
class SectionRef {
 public:
  SectionRef() {}
  SectionRef(const std::shared_ptr<detail::SectionData>& data, const std::string& name)
      : data_(data), name_(name) {}

  bool is_open() const { return data_ != nullptr; }
  const std::string& name() const { return name_; }

#ifndef CCUTILS_NO_JSON
  nlohmann::json& json() { return data_ ? data_->json : fallback_; }
  nlohmann::json& global_json() { return data_ ? data_->global_json : fallback_; }

  template <class T>
  SectionRef& put(const std::string& key, T&& value) {
    if (check(key)) data_->json[key] = std::forward<T>(value);
    return *this;
  }
  template <class T>
  SectionRef& put(const std::string& key1, const std::string& key2, T&& value) {
    if (check(key1)) data_->json[key1][key2] = std::forward<T>(value);
    return *this;
  }
  /* MPI: value owned by rank 0 (callers decide who stores it). */
  template <class T>
  SectionRef& put_global(const std::string& key, T&& value) {
    if (check(key)) data_->global_json[key] = std::forward<T>(value);
    return *this;
  }
#else
  template <class T>
  SectionRef& put(const std::string&, T&&) { return *this; }
  template <class T>
  SectionRef& put(const std::string&, const std::string&, T&&) { return *this; }
  template <class T>
  SectionRef& put_global(const std::string&, T&&) { return *this; }
#endif

 private:
  bool check(const std::string& key) const {
    if (data_) return true;
    warn_once("section:" + name_,
              "section '" + name_ + "' is not open; data for key '" + key + "' is ignored");
    return false;
  }

  std::shared_ptr<detail::SectionData> data_;
  std::string name_;
#ifndef CCUTILS_NO_JSON
  nlohmann::json fallback_;
#endif
};

/* The innermost open section called `name` (is_open() is false if there is none). */
inline SectionRef section(const std::string& name) {
  return SectionRef(detail::find_section(name), name);
}

/* The innermost open section of any name. */
inline SectionRef current_section() {
  std::shared_ptr<detail::SectionData> data = detail::innermost_section();
  return SectionRef(data, data ? data->name : std::string());
}

}  // namespace ccutils

#endif
