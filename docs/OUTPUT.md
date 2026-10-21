# Output destinations

Everything ccutils prints for humans (sections, JSON blocks, timers, MPI all-print,
`*_ONCE` prints, buffered prints, MPI timer reports) goes through one output sink.
By default that sink is `stdout`, and the bytes are identical to previous versions.

| Variable | Values | Default |
|----------|--------|---------|
| `CCUTILS_HUMAN_OUT` | `stdout`, `stderr`, `none`, `file:<path>`, `append:<path>` | `stdout` |
| `CCUTILS_MACHINE_OUT` | `none`, `stdout`, `stderr`, `inline`, `file:<path>`, `append:<path>` | `none` |
| `CCUTILS_FORMAT` | `legacy`, `v2` | `v2` |
| `CCUTILS_COLOR` | `auto`, `always`, `never` | `auto` |
| `CCUTILS_FLUSH_WAIT_US` | non-negative integer microseconds | `200000` |

`CCUTILS_FLUSH_WAIT_US` sets the delay after each `CCUTILS_MPI_PRINT_ONCE` call; set it to
`0` to disable the delay. Invalid or out-of-range values use the default.

`CCUTILS_COLOR=auto` enables ANSI colors only when the sink is a terminal. `NO_COLOR` set to
a non-empty value disables colors regardless of `CCUTILS_COLOR`. Invalid `CCUTILS_COLOR`
values use `auto`. `-DCCUTILS_COLOR_LEVEL`
continues to select the palette at compile time.

`file:` truncates, `append:` appends. Path templates: `%r` rank, `%p` pid,
`%j` `SLURM_JOB_ID` (or pid), `%%` a literal `%`.

```bash
CCUTILS_HUMAN_OUT=file:run-%j.log mpirun -n 4 ./app      # log in a file, terminal stays clean
CCUTILS_HUMAN_OUT=file:run-%r.log mpirun -n 4 ./app      # one file per rank
```

The same can be set in code, before or after output has started:

```cpp
ccutils::Config cfg;
cfg.human_out = "file:run.log";
ccutils::init(cfg);               // overrides the environment; emits run_start if enabled
```

`ccutils::init(Config)` is optional and non-collective. It configures the sinks and
immediately emits `run_start` when machine output is enabled. `ccutils::configure()`
changes configuration without emitting a record; lazy machine output still starts
with `run_start` on the first record.

## Which rank writes a file

* A path **without** `%r` is opened by rank 0. Any other rank that prints something
  on its own (a CPU timer, an MPI timer report whose root is not rank 0) writes the
  **sidecar** `<path>.r<rank>` instead, so that processes never share a file.
  Gathered output (`CCUTILS_MPI_ALL_PRINT*`, section end) is always written by rank 0.
* A path **with** `%r` gives every rank its own file (no sidecars).
* If a file cannot be opened, ccutils warns once on stderr and falls back to stdout.

The rank is taken from the MPI macros (`CCUTILS_MPI_INIT`, all-print, MPI timers) or,
before they run, from the launcher environment (`OMPI_COMM_WORLD_RANK`, `PMI_RANK`,
`PMIX_RANK`, `MV2_COMM_WORLD_RANK`, `SLURM_PROCID`).

## Current limitations

* `CCUTILS_MPI_PROCESS_PRINT` routes its banner lines through the human sink, but text
  written by the block to `stdout` is not redirected.
* `auto` color mode uses terminal detection for each sink; files and pipes have colors removed.
* `CCUTILS_FORMAT=v2` is the default and enables path-based section banners, JSON key/value
  lines, timer tables and `[rN]` all-print text. `CCUTILS_FORMAT=legacy` preserves the
  existing output bytes. Define `CCUTILS_FORMAT_LEGACY` at compile time to select legacy
  as the build default.
