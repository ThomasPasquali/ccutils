// Unit test for ccutils/emitter.hpp (sink parsing, path templates, files, sidecars).
#include <cstdio>
#include <cstdlib>
#include <string>
#include <ccutils/emitter.hpp>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL line %d: %s\n", __LINE__, #cond); ++failures; } } while (0)

static std::string slurp(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return "<missing>";
  std::string s;
  char buf[256];
  std::size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
  std::fclose(f);
  return s;
}

int main() {
  using namespace ccutils;
  SinkSpec spec;
  std::string err;

  // ---- parse_sink
  CHECK(parse_sink("stdout", false, &spec, &err) && spec.kind == SinkKind::Stdout);
  CHECK(parse_sink("stderr", false, &spec, &err) && spec.kind == SinkKind::Stderr);
  CHECK(parse_sink("none", false, &spec, &err) && spec.kind == SinkKind::None);
  CHECK(parse_sink("file:a.txt", false, &spec, &err) && spec.kind == SinkKind::File && spec.path == "a.txt");
  CHECK(parse_sink("append:b:c.txt", false, &spec, &err) && spec.kind == SinkKind::Append && spec.path == "b:c.txt");
  CHECK(!parse_sink("inline", false, &spec, &err));
  CHECK(parse_sink("inline", true, &spec, &err) && spec.kind == SinkKind::Inline);
  CHECK(!parse_sink("file:", false, &spec, &err));
  CHECK(!parse_sink("append:", false, &spec, &err));
  CHECK(!parse_sink("bogus", false, &spec, &err) && !err.empty());
  CHECK(!parse_sink("", false, &spec, &err));

  // ---- expand_path
  bool uses_rank = false;
  CHECK(expand_path("o_%r.txt", 3, &uses_rank) == "o_3.txt" && uses_rank);
  CHECK(expand_path("o.txt", 3, &uses_rank) == "o.txt" && !uses_rank);
  CHECK(expand_path("100%%_%x", 0, &uses_rank) == "100%_%x");
  CHECK(expand_path("trailing%", 0, &uses_rank) == "trailing%");
  setenv("SLURM_JOB_ID", "4242", 1);
  CHECK(expand_path("job%j.txt", 0, &uses_rank) == "job4242.txt");
  unsetenv("SLURM_JOB_ID");
  CHECK(expand_path("p%p", 0, &uses_rank) == "p" + std::to_string(static_cast<long>(getpid())));

  // ---- file sink, rank 0: truncates, small and >512-byte writes
  std::remove("u_main.txt");
  Config cfg;
  cfg.human_out = "file:u_main.txt";
  configure(cfg);
  set_world_rank(0);
  human_printf("hello %d\n", 1);
  human_putc('x');
  human_flush();
  CHECK(slurp("u_main.txt") == "hello 1\nx");
  const std::string big(1000, 'a');
  human_printf("%s", big.c_str());
  human_flush();
  CHECK(slurp("u_main.txt") == "hello 1\nx" + big);
  configure(cfg);  // re-open with "file:" truncates again
  human_printf("again\n");
  human_flush();
  CHECK(slurp("u_main.txt") == "again\n");

  // ---- rank > 0 writes the sidecar, never the main file
  std::remove("u_side.txt");
  std::remove("u_side.txt.r2");
  cfg.human_out = "file:u_side.txt";
  configure(cfg);
  set_world_rank(2);
  human_printf("r2\n");
  human_flush();
  CHECK(slurp("u_side.txt.r2") == "r2\n");
  CHECK(slurp("u_side.txt") == "<missing>");

  // ---- %r: every rank has its own file, no ".r" suffix
  std::remove("u_5.txt");
  cfg.human_out = "file:u_%r.txt";
  configure(cfg);
  set_world_rank(5);
  human_printf("five\n");
  human_flush();
  CHECK(slurp("u_5.txt") == "five\n");

  // ---- append keeps earlier content across re-opens
  std::remove("u_app.txt");
  cfg.human_out = "append:u_app.txt";
  configure(cfg);
  set_world_rank(0);
  human_printf("1\n");
  configure(cfg);
  human_printf("2\n");
  human_flush();
  CHECK(slurp("u_app.txt") == "1\n2\n");

  // ---- machine stream: own file, and "inline" into the human stream
  std::remove("u_h.txt");
  std::remove("u_m.jsonl");
  cfg.human_out = "file:u_h.txt";
  cfg.machine_out = "file:u_m.jsonl";
  configure(cfg);
  CHECK(machine_enabled());
  machine_write_line("{\"a\":1}");
  human_printf("h\n");
  cfg.machine_out = "inline";
  configure(cfg);  // closes (flushes) the previous files
  CHECK(slurp("u_m.jsonl") == "{\"a\":1}\n");
  machine_write_line("{\"b\":2}");
  human_flush();
  CHECK(slurp("u_h.txt") == "@ccutils {\"b\":2}\n");  // truncated by the re-open, then one inline record
  cfg.machine_out = "none";
  configure(cfg);
  CHECK(!machine_enabled());

  // ---- none discards
  cfg.human_out = "none";
  configure(cfg);
  human_printf("discarded\n");
  human_flush();

  // ---- an unopenable path falls back to stdout (warning on stderr, silenced here)
  CHECK(std::freopen("/dev/null", "w", stderr) != nullptr);
  cfg.human_out = "file:/nonexistent_dir_for_ccutils/x.txt";
  configure(cfg);
  human_printf("fallback\n");
  human_flush();

  std::remove("u_main.txt"); std::remove("u_side.txt.r2"); std::remove("u_5.txt");
  std::remove("u_app.txt"); std::remove("u_h.txt"); std::remove("u_m.jsonl");
  cfg = Config();
  configure(cfg);
  std::printf(failures == 0 ? "emitter_unit: ok\n" : "emitter_unit: %d failure(s)\n", failures);
  return failures == 0 ? 0 : 1;
}
