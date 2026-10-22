#ifndef CCUTILS_MPI_GATHER_HPP
#define CCUTILS_MPI_GATHER_HPP

#include <mpi.h>
#include <climits>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <vector>
#include <utility>
#include <stdint.h>
#include <cstring>
#include "../emitter.hpp"
#include "../formats.h"
#ifndef CCUTILS_NO_JSON
#include <nlohmann/json.hpp>
#endif

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

#ifndef CCUTILS_NO_JSON
inline bool decode_base64(const std::string& input, std::string* output) {
  int value = 0, bits = -8;
  for (std::size_t i = 0; i < input.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(input[i]);
    if (c == '=') break;
    int digit = -1;
    if (c >= 'A' && c <= 'Z') digit = c - 'A';
    else if (c >= 'a' && c <= 'z') digit = c - 'a' + 26;
    else if (c >= '0' && c <= '9') digit = c - '0' + 52;
    else if (c == '+') digit = 62;
    else if (c == '/') digit = 63;
    else return false;
    value = (value << 6) | digit;
    bits += 6;
    if (bits >= 0) {
      output->push_back(static_cast<char>((value >> bits) & 0xff));
      bits -= 8;
    }
  }
  return true;
}

inline nlohmann::json json_value_schema(const nlohmann::json& value) {
  typedef nlohmann::json Json;
  if (value.is_object()) {
    Json fields = Json::array();
    for (Json::const_iterator it = value.begin(); it != value.end(); ++it)
      fields.push_back(Json::array({it.key(), json_value_schema(it.value())}));
    return Json::array({"object", fields});
  }
  if (value.is_array()) {
    if (value.empty()) return Json::array({"array", Json()});
    Json item_schema = json_value_schema(value[0]);
    for (std::size_t i = 1; i < value.size(); ++i)
      if (json_value_schema(value[i]) != item_schema) return Json::array({"array_raw", Json()});
    return Json::array({"array", item_schema});
  }
  const char* type = value.is_string() ? "string" : value.is_boolean() ? "boolean" :
                     value.is_number() ? "number" : "null";
  return Json::array({"value", type});
}

inline nlohmann::json merge_json_schemas(const nlohmann::json& left,
                                         const nlohmann::json& right,
                                         bool* compatible) {
  typedef nlohmann::json Json;
  if (!left.is_array() || !right.is_array() || left.empty() || right.empty() ||
      left[0] != right[0]) { *compatible = false; return left; }
  const std::string kind = left[0].get<std::string>();
  if (kind == "array") {
    if (left[1].is_null()) return right;
    if (right[1].is_null()) return left;
    Json child = merge_json_schemas(left[1], right[1], compatible);
    return Json::array({"array", child});
  }
  if (kind == "object") {
    if (left[1].size() != right[1].size()) { *compatible = false; return left; }
    Json fields = Json::array();
    for (std::size_t i = 0; i < left[1].size(); ++i) {
      if (left[1][i][0] != right[1][i][0]) { *compatible = false; return left; }
      fields.push_back(Json::array({left[1][i][0],
          merge_json_schemas(left[1][i][1], right[1][i][1], compatible)}));
    }
    return Json::array({"object", fields});
  }
  if (left != right) *compatible = false;
  return left;
}

inline nlohmann::json compact_json_values(const nlohmann::json& value,
                                          const nlohmann::json& schema) {
  typedef nlohmann::json Json;
  const std::string kind = schema[0].get<std::string>();
  if (kind == "object") {
    Json fields = Json::array();
    for (Json::const_iterator it = schema[1].begin(); it != schema[1].end(); ++it) {
      const std::string key = (*it)[0].get<std::string>();
      fields.push_back(compact_json_values(value[key], (*it)[1]));
    }
    return fields;
  }
  if (kind == "array" && !schema[1].is_null()) {
    Json items = Json::array();
    for (Json::const_iterator it = value.begin(); it != value.end(); ++it)
      items.push_back(compact_json_values(*it, schema[1]));
    return items;
  }
  return value;
}

inline nlohmann::json json_schema_template(const nlohmann::json& schema) {
  typedef nlohmann::json Json;
  const std::string kind = schema[0].get<std::string>();
  if (kind == "object") {
    Json value = Json::object();
    for (Json::const_iterator it = schema[1].begin(); it != schema[1].end(); ++it)
      value[(*it)[0].get<std::string>()] = json_schema_template((*it)[1]);
    return value;
  }
  if (kind == "array") {
    Json value = Json::array();
    if (!schema[1].is_null()) value.push_back(json_schema_template(schema[1]));
    return value;
  }
  if (kind == "array_raw") return Json::array({"<values>"});
  if (schema[1] == "string") return "<string>";
  if (schema[1] == "number") return "<number>";
  if (schema[1] == "boolean") return "<boolean>";
  return nullptr;
}
#endif

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
    std::vector<std::string> human_by_rank(static_cast<std::size_t>(size));
    std::vector<bool> has_human_by_rank(static_cast<std::size_t>(size), false);
    for (int i = 0; i < size; ++i)
      visit_frames(by_rank[i], [&](char kind, const char*, std::size_t n) {
        if (kind == 'H' && n && ccutils::human_enabled()) any_human = true;
      });
    for (int i = 0; i < size; ++i)
      visit_frames(by_rank[i], [&](char kind, const char* bytes, std::size_t n) {
        if (kind == 'H' && n) {
          has_human_by_rank[static_cast<std::size_t>(i)] = true;
          human_by_rank[static_cast<std::size_t>(i)].append(bytes, n);
        }
      });
    bool compact_human_json = false;
#ifndef CCUTILS_NO_JSON
    if (ccutils::human_enabled() && size > 1) {
      typedef nlohmann::json Json;
      std::vector<Json> values(static_cast<std::size_t>(size));
      Json schema;
      bool compatible = true;
      for (int i = 0; i < size; ++i) {
        if (!has_human_by_rank[static_cast<std::size_t>(i)]) { compatible = false; break; }
        try { values[static_cast<std::size_t>(i)] = Json::parse(human_by_rank[static_cast<std::size_t>(i)]); }
        catch (...) { compatible = false; break; }
        Json current = ccutils::mpi::json_value_schema(values[static_cast<std::size_t>(i)]);
        if (i == 0) schema = current;
        else schema = ccutils::mpi::merge_json_schemas(schema, current, &compatible);
        if (!compatible) break;
      }
      if (compatible) {
        compact_human_json = true;
        ccutils::human_printf("-- all-ranks: %s --\n", print_name ? print_name : "");
        ccutils::human_printf("schema: %s\n", ccutils::mpi::json_schema_template(schema).dump().c_str());
        for (int i = 0; i < size; ++i) {
          Json compact = ccutils::mpi::compact_json_values(values[static_cast<std::size_t>(i)], schema);
          ccutils::human_printf("[r%d] %s\n", world_ranks[static_cast<std::size_t>(i)], compact.dump().c_str());
        }
        ccutils::human_printf("-- end all-ranks: %s --\n", print_name ? print_name : "");
      }
    }
#endif
    if (any_human) {
      if (compact_human_json) {
        /* Already emitted once in schema plus per-rank value form. */
      } else if (ccutils::human_v2())
        ccutils::human_printf("-- all-ranks: %s --\n", print_name ? print_name : "");
      else if (print_name)
        ccutils::human_printf(CCUTILS_FMT_MPI_PRINT_ALL_NAMED_START, print_name);
    }
    for (int i = 0; i < size && !compact_human_json; ++i) {
      bool has_human = false;
      visit_frames(by_rank[i], [&](char kind, const char* bytes, std::size_t n) {
        if (kind == 'M') {
#ifndef CCUTILS_NO_JSON
          /* Machine frames are emitted below after trying MPI JSON compaction. */
#else
          ccutils::machine_write_line_for_rank(std::string(bytes, n), world_ranks[i]);
#endif
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
#ifndef CCUTILS_NO_JSON
    /* Parsed MPI JSON output is stored as structured rows, never as text. */
    if (ccutils::machine_enabled()) {
      typedef nlohmann::json Json;
      std::vector<Json> parsed(static_cast<std::size_t>(size));
      std::vector<bool> found(static_cast<std::size_t>(size), false);
      std::vector<std::vector<Json> > text_parts(static_cast<std::size_t>(size));
      std::vector<std::pair<int, std::string> > other_frames;
      bool complete_json = true;
      for (int i = 0; i < size; ++i) {
        visit_frames(by_rank[i], [&](char kind, const char* bytes, std::size_t n) {
          if (kind != 'M') return;
          Json rec;
          try { rec = Json::parse(bytes, bytes + n); }
          catch (...) {
            other_frames.push_back(std::make_pair(world_ranks[static_cast<std::size_t>(i)], std::string(bytes, n)));
            return;
          }
          if (rec.is_object() && (rec.value("type", std::string()) == "text" ||
                                  rec.value("type", std::string()) == "json"))
            text_parts[static_cast<std::size_t>(i)].push_back(rec);
          else
            other_frames.push_back(std::make_pair(world_ranks[static_cast<std::size_t>(i)], std::string(bytes, n)));
        });
      }
      for (int i = 0; i < size; ++i) {
        std::vector<Json>& parts = text_parts[static_cast<std::size_t>(i)];
        if (parts.empty()) { complete_json = false; continue; }
        std::sort(parts.begin(), parts.end(), [](const Json& a, const Json& b) {
          return a.value("part", 0) < b.value("part", 0);
        });
        Json meta = parts[0];
        std::string serialized;
        bool valid_parts = true;
        const bool direct_json = meta.value("type", std::string()) == "json";
        Json payload;
        if (direct_json) {
          valid_parts = parts.size() == 1 && parts[0].contains("value");
          if (valid_parts) payload = parts[0]["value"];
        }
        for (std::size_t p = 0; p < parts.size(); ++p) {
          if (direct_json) break;
          const Json& part = parts[p];
          if (part.value("part", -1) != static_cast<int>(p) ||
              part.value("last", false) != (p + 1 == parts.size()) ||
              part.value("print", std::string()) != meta.value("print", std::string()) ||
              part.value("call", 0LL) != meta.value("call", 0LL) ||
              part.value("trailing_newline", false) != meta.value("trailing_newline", false) ||
              part.value("comm", std::string()) != meta.value("comm", std::string())) {
            valid_parts = false;
            break;
          }
          if (part.contains("text") && part["text"].is_string())
            serialized += part["text"].get<std::string>();
          else if (part.value("encoding", std::string()) == "base64" &&
                   part.contains("data") && part["data"].is_string()) {
            if (!ccutils::mpi::decode_base64(part["data"].get<std::string>(), &serialized)) {
              valid_parts = false;
              break;
            }
          } else {
            valid_parts = false;
            break;
          }
        }
        if (!direct_json) {
          try {
            if (!valid_parts) throw std::runtime_error("invalid text chunks");
            payload = Json::parse(serialized);
          } catch (...) { complete_json = false; continue; }
        } else if (!valid_parts) { complete_json = false; continue; }
          Json schema = ccutils::mpi::json_value_schema(payload);
          parsed[static_cast<std::size_t>(i)] = Json::object();
          parsed[static_cast<std::size_t>(i)]["_meta"] = meta;
          parsed[static_cast<std::size_t>(i)]["_schema"] = schema;
          parsed[static_cast<std::size_t>(i)]["_payload"] = payload;
          found[static_cast<std::size_t>(i)] = true;
      }
      for (int i = 0; i < size; ++i) if (!found[static_cast<std::size_t>(i)]) complete_json = false;
      bool metadata_compatible = complete_json;
      if (metadata_compatible) {
        for (int i = 1; i < size; ++i) {
          if (parsed[static_cast<std::size_t>(i)]["_meta"].value("print", std::string()) !=
                  parsed[0]["_meta"].value("print", std::string()) ||
              parsed[static_cast<std::size_t>(i)]["_meta"].value("call", 0LL) !=
                  parsed[0]["_meta"].value("call", 0LL) ||
              parsed[static_cast<std::size_t>(i)]["_meta"].value("trailing_newline", false) !=
                  parsed[0]["_meta"].value("trailing_newline", false) ||
              parsed[static_cast<std::size_t>(i)]["_meta"].value("comm", std::string()) !=
                  parsed[0]["_meta"].value("comm", std::string())) metadata_compatible = false;
        }
      }
      bool same_schema = complete_json;
      Json schema;
      if (same_schema) {
        schema = parsed[0]["_schema"];
        for (int i = 1; i < size && same_schema; ++i)
          schema = ccutils::mpi::merge_json_schemas(
              schema, parsed[static_cast<std::size_t>(i)]["_schema"], &same_schema);
      }
      if (complete_json && metadata_compatible && same_schema) {
        Json rows = Json::array();
        for (int i = 0; i < size; ++i) {
          rows.push_back(Json{{"rank", world_ranks[static_cast<std::size_t>(i)]},
                              {"values", ccutils::mpi::compact_json_values(
                                  parsed[static_cast<std::size_t>(i)]["_payload"], schema)}});
        }
        const Json& meta = parsed[0]["_meta"];
        ccutils::machine::Record record("mpi_json_batch", ccutils::machine::Scope::Comm);
        record.str("print", meta.value("print", std::string()))
            .integer("call", meta.value("call", 0LL))
            .boolean("trailing_newline", meta.value("trailing_newline", true));
        if (meta.contains("comm") && meta["comm"].is_string() && !meta["comm"].get<std::string>().empty())
          record.str("comm", meta["comm"].get<std::string>());
        record.raw("schema", schema.dump()).raw("rows", rows.dump()).emit();
      } else if (complete_json) {
        Json rows = Json::array();
        for (int i = 0; i < size; ++i) {
          const Json& meta = parsed[static_cast<std::size_t>(i)]["_meta"];
          rows.push_back(Json{{"rank", world_ranks[static_cast<std::size_t>(i)]},
                              {"print", meta.value("print", std::string())},
                              {"call", meta.value("call", 0LL)},
                              {"trailing_newline", meta.value("trailing_newline", true)},
                              {"comm", meta.value("comm", std::string())},
                              {"value", parsed[static_cast<std::size_t>(i)]["_payload"]}});
        }
        ccutils::machine::Record("mpi_json_rows", ccutils::machine::Scope::Comm)
            .raw("rows", rows.dump()).emit();
      }
      if (complete_json) {
        for (std::size_t i = 0; i < other_frames.size(); ++i)
          ccutils::machine_write_line_for_rank(other_frames[i].second, other_frames[i].first);
      } else {
        for (int i = 0; i < size; ++i)
          visit_frames(by_rank[i], [&](char kind, const char* bytes, std::size_t n) {
            if (kind == 'M') ccutils::machine_write_line_for_rank(std::string(bytes, n), world_ranks[static_cast<std::size_t>(i)]);
          });
      }
    }
#endif
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
