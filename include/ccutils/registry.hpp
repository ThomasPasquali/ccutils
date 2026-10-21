#ifndef CCUTILS_REGISTRY_HPP
#define CCUTILS_REGISTRY_HPP

/* Process-wide registry that maps a NAME to the live objects created under that
 * name. It is what lets a function in another file reach the timer or JSON object
 * that main() defined: both sides only need to agree on the name.
 *
 * The registry is an inline function-local static, so every translation unit of a
 * program shares the same instance.
 */

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ccutils {
namespace detail {

template <class T>
class NamedRegistry {
 public:
  static NamedRegistry& instance() {
    static NamedRegistry registry;
    return registry;
  }

  /* `owner` keeps heap storage alive for entries the registry created itself. */
  void add(const std::string& name, const void* id, const T& value,
           std::shared_ptr<void> owner = std::shared_ptr<void>()) {
    std::lock_guard<std::mutex> lock(mutex_);
    Entry entry;
    entry.id = id;
    entry.value = value;
    entry.owner = owner;
    entries_[name].push_back(entry);
  }

  void remove(const std::string& name, const void* id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(name);
    if (it == entries_.end()) return;
    std::vector<Entry>& stack = it->second;
    for (std::size_t i = stack.size(); i > 0; --i) {
      if (stack[i - 1].id == id) {
        stack.erase(stack.begin() + static_cast<std::ptrdiff_t>(i - 1));
        break;
      }
    }
    if (stack.empty()) entries_.erase(it);
  }

  /* Innermost (most recently added) live entry for `name`. */
  bool find(const std::string& name, T* out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(name);
    if (it == entries_.end() || it->second.empty()) return false;
    *out = it->second.back().value;
    return true;
  }

 private:
  struct Entry {
    const void* id;
    T value;
    std::shared_ptr<void> owner;
  };
  mutable std::mutex mutex_;
  std::map<std::string, std::vector<Entry> > entries_;
};

/* Registers `value` under `name` for the lifetime of this object. */
template <class T>
class Registration {
 public:
  Registration(const char* name, const void* id, const T& value) : name_(name), id_(id) {
    NamedRegistry<T>::instance().add(name_, id_, value);
  }
  ~Registration() { NamedRegistry<T>::instance().remove(name_, id_); }
  Registration(const Registration&) = delete;
  Registration& operator=(const Registration&) = delete;

 private:
  std::string name_;
  const void* id_;
};

}  // namespace detail
}  // namespace ccutils

#endif
