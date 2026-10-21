// Unit test for the JSON text helpers and the section stack of ccutils/machine.hpp.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <ccutils/macros.hpp>
#include <ccutils/timers.hpp>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL line %d: %s\n", __LINE__, #cond); ++failures; } } while (0)

static void return_from_raii_section() {
  ccutils::Section section("early", "Early return");
  section.put("value", 2);
  return;
}

int main() {
  using namespace ccutils::machine;

  // ---- quote: escapes, control characters, UTF-8
  CHECK(quote("plain") == "\"plain\"");
  CHECK(quote("a\"b\\c") == "\"a\\\"b\\\\c\"");
  CHECK(quote("l1\nl2\r\t") == "\"l1\\nl2\\r\\t\"");
  CHECK(quote(std::string("x\x01y", 3)) == "\"x\\u0001y\"");
  CHECK(quote(std::string("nul\0end", 7)) == "\"nul\\u0000end\"");
  CHECK(quote("caf\xC3\xA9") == "\"caf\xC3\xA9\"");              // valid 2-byte sequence kept
  CHECK(quote("\xE2\x82\xAC") == "\"\xE2\x82\xAC\"");            // euro sign (3 bytes)
  CHECK(quote("bad\xFFz") == "\"bad\xEF\xBF\xBDz\"");            // invalid byte -> U+FFFD
  CHECK(quote("trunc\xE2\x82") == "\"trunc\xEF\xBF\xBD\xEF\xBF\xBD\"");  // truncated sequence
  CHECK(quote("\xC0\x80") == "\"\xEF\xBF\xBD\xEF\xBF\xBD\"");    // overlong lead byte rejected

  // ---- numbers: shortest text that reads back identically; nan/inf -> null
  CHECK(format_real(0.1) == "0.1");
  CHECK(format_real(1.5e-9) == "1.5e-09");
  CHECK(format_real(100.0) == "100");
  const double values[] = {1.0 / 3.0, 2.2250738585072014e-308, 1.7976931348623157e308, -0.0, 123456789.123456789};
  for (double v : values) CHECK(std::strtod(format_real(v).c_str(), nullptr) == v);
  CHECK(format_real(std::nan("")) == "null");
  CHECK(format_real(std::numeric_limits<double>::infinity()) == "null");
  CHECK(format_real32(0.1f) == "0.1");
  const float fvalues[] = {1.0f / 3.0f, 1.17549435e-38f, 3.4028235e38f, 0.0038145997f};
  for (float v : fvalues) CHECK(std::strtof(format_real32(v).c_str(), nullptr) == v);
  CHECK(format_real32(std::numeric_limits<float>::quiet_NaN()) == "null");

  // ---- CPU timer statistics retain low order bits over large sample sets
  std::vector<double> timer_samples(1000000, 0.1);
  const ccutils_timers::TimerStats timer_stats = ccutils_timers::compute_stats(timer_samples);
  CHECK(timer_stats.n == 1000000);
  CHECK(timer_stats.sum == 100000.0);  // same rounded result as math.fsum
  CHECK(timer_stats.avg == timer_stats.sum / timer_stats.n);

  // ---- section stack
  CHECK(section_path() == "");
  section_push("a");
  section_push("b");
  CHECK(section_path() == "a/b");
  section_pop("b");
  CHECK(section_path() == "a");
  section_push("b");
  section_push("c");
  section_pop("b");                // closes b and everything opened inside it
  CHECK(section_path() == "a");
  section_pop("does_not_exist");   // ignored
  CHECK(section_path() == "a");
  section_pop("a");
  CHECK(section_path() == "");

  // ---- init: explicit configuration emits run_start before other records
  ccutils::Config cfg;
  cfg.human_out = "none";
  cfg.machine_out = "file:run_start.jsonl";
  ccutils::init(cfg);
  CCUTILS_SECTION_DEF(outer, "Outer")
  CCUTILS_SUBSECTION_DEF(outer, inner, "Inner")
  CCUTILS_SECTION_END(outer)
  {
    ccutils::Section section("raii", "RAII");
    section.put("value", 1);
  }
  return_from_raii_section();
  try {
    ccutils::Section section("throws", "Exception");
    throw std::runtime_error("test unwind");
  } catch (const std::runtime_error&) {}
  cfg.machine_out = "none";
  ccutils::configure(cfg);  // closes the sink after flushing the record
  std::ifstream run_start_file("run_start.jsonl");
  const std::string records((std::istreambuf_iterator<char>(run_start_file)),
                            std::istreambuf_iterator<char>());
  const std::size_t first_newline = records.find('\n');
  const std::string run_start_line = records.substr(0, first_newline);
  CHECK(run_start_line.find("\"type\":\"run_start\"") != std::string::npos);
  CHECK(run_start_line.find("\"seq\":1") != std::string::npos);
  CHECK(records.find("\"type\":\"section_end\",\"section\":\"outer/inner\"") != std::string::npos);
  CHECK(records.find("\"section\":\"outer/inner\",\"scope\":\"global\",\"rank\":null,\"ok\":false") != std::string::npos);
  CHECK(records.find("\"type\":\"warning\",\"section\":\"outer/inner\"") != std::string::npos);
  CHECK(records.find("\"section\":\"outer\",\"scope\":\"global\",\"rank\":null,\"ok\":true") != std::string::npos);
  CHECK(records.find("\"type\":\"section_end\",\"section\":\"raii\",\"scope\":\"global\",\"rank\":null,\"ok\":true") != std::string::npos);
  CHECK(records.find("\"type\":\"section_end\",\"section\":\"early\",\"scope\":\"global\",\"rank\":null,\"ok\":true") != std::string::npos);
  CHECK(records.find("\"type\":\"section_end\",\"section\":\"throws\",\"scope\":\"global\",\"rank\":null,\"ok\":false") != std::string::npos);
  run_start_file.close();
  std::remove("run_start.jsonl");

  std::printf(failures == 0 ? "machine_unit: ok\n" : "machine_unit: %d failure(s)\n", failures);
  return failures == 0 ? 0 : 1;
}
