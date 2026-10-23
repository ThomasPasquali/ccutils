#ifndef CCUTILS_SECTIONS_HPP
#define CCUTILS_SECTIONS_HPP

/* Path-addressed section tree. Paths use slash-separated names; section lifetime
 * is strict LIFO, and every path may be opened only once in a process. */

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include "emitter.hpp"

#ifndef CCUTILS_NO_JSON
#include <nlohmann/json.hpp>
#endif

namespace ccutils {
class TimerRef;
class MpiTimer;
namespace detail {

struct SectionData {
  std::string path;
  std::string title;
  bool open = true;
  std::map<std::string, std::shared_ptr<SectionData> > children;
  std::set<std::string> timer_names;
  std::vector<std::function<void()> > timer_emitters;
#ifndef CCUTILS_NO_JSON
  nlohmann::json json;         /* the section's data (rank-local data in MPI programs) */
  nlohmann::json global_json;  /* MPI: data written once by rank 0 */
#endif
};

struct SectionStack {
  std::mutex mutex;
  std::vector<std::shared_ptr<SectionData> > open;
  std::map<std::string, std::shared_ptr<SectionData> > roots;
  std::set<std::string> seen;
};

inline SectionStack& section_stack() {
  static SectionStack stack;
  return stack;
}

inline void section_error(const std::string& message) {
  std::fprintf(stderr, "ccutils section error: %s\n", message.c_str());
  std::fflush(stderr);
  std::exit(EXIT_FAILURE);
}

inline std::shared_ptr<SectionData> push_section(const std::string& path,
                                                  const std::string& title = std::string()) {
  SectionStack& s = section_stack();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (path.empty() || path[0] == '/' || path[path.size() - 1] == '/' ||
      path.find("//") != std::string::npos)
    section_error("invalid section path '" + path + "' (expected nonempty slash-separated names)");
  if (!s.open.empty() && path.compare(0, s.open.back()->path.size() + 1,
                                      s.open.back()->path + "/") != 0)
    section_error("cannot open section '" + path + "' while '" + s.open.back()->path +
                  "' is open; child paths must begin with the parent path and '/'");
  for (std::size_t i = 0; i < s.open.size(); ++i)
    if (s.open[i]->path == path)
      section_error("section '" + path + "' is already open");
  if (s.seen.count(path))
    section_error("section '" + path + "' cannot be reopened; each path may be opened only once");
  s.seen.insert(path);
  std::shared_ptr<SectionData> data = std::make_shared<SectionData>();
  data->path = path;
  data->title = title;
  if (!s.open.empty()) {
    const std::string child_name = path.substr(s.open.back()->path.size() + 1);
    if (child_name.find('/') != std::string::npos)
      section_error("cannot open section '" + path + "': parent '" + s.open.back()->path +
                    "' is not its immediate parent");
    s.open.back()->children[child_name] = data;
  } else {
    if (path.find('/') != std::string::npos)
      section_error("cannot open section '" + path + "' without its parent section open");
    s.roots[path] = data;
  }
  s.open.push_back(data);
  return data;
}

inline std::shared_ptr<SectionData> pop_section(const std::string& path) {
  SectionStack& s = section_stack();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.open.empty()) section_error("cannot close section '" + path + "': no section is open");
  const std::shared_ptr<SectionData> current = s.open.back();
  if (current->path != path) {
    const std::string parent = current->path.size() > path.size() &&
        current->path.compare(0, path.size(), path) == 0 && current->path[path.size()] == '/'
        ? path : current->path;
    section_error("cannot close section '" + path + "' while child section '" + current->path +
                  "' is still open; close the child first (parent: '" + parent + "')");
  }
  current->open = false;
  s.open.pop_back();
  return current;
}

inline std::shared_ptr<SectionData> find_section(const std::string& name) {
  SectionStack& s = section_stack();
  std::lock_guard<std::mutex> lock(s.mutex);
  for (std::size_t i = s.open.size(); i > 0; --i)
    if (s.open[i - 1]->path == name) return s.open[i - 1];
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
  return s.open.empty() ? std::string() : s.open.back()->path;
}

}  // namespace detail

/* Handle to an open section. Cheap to copy; stays valid after the section closes
 * (updates are then simply never printed). */
class SectionRef {
 public:
  SectionRef() {}
  SectionRef(const std::shared_ptr<detail::SectionData>& data, const std::string& name)
      : data_(data), name_(name) {}

  bool is_open() const { return data_ != nullptr && data_->open; }
  const std::string& name() const { return name_; }
  TimerRef timer(const std::string& timer_name);
  MpiTimer& mpi_timer(const std::string& timer_name);
  void add_timer_emitter(const std::string& timer_name,
                         const std::function<void()>& emit) {
    if (!is_open()) return;
    if (data_->timer_names.insert(timer_name).second)
      data_->timer_emitters.push_back(emit);
  }
  void emit_timers() const {
    if (!is_open()) return;
    for (std::size_t i = 0; i < data_->timer_emitters.size(); ++i)
      data_->timer_emitters[i]();
  }

#ifndef CCUTILS_NO_JSON
  nlohmann::json& json() { return is_open() ? data_->json : fallback_; }
  nlohmann::json& global_json() { return is_open() ? data_->global_json : fallback_; }

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
    if (is_open()) return true;
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
  return SectionRef(data, data ? data->path : std::string());
}

inline std::string section_child_path(const std::string& parent, const std::string& child) {
  return parent.empty() ? child : parent + "/" + child;
}

}  // namespace ccutils

#endif
