# Machine-readable output (JSON Lines, schema v2)

Enable with `CCUTILS_MACHINE_OUT` (see [OUTPUT.md](OUTPUT.md)):

```bash
CCUTILS_MACHINE_OUT=file:run.jsonl ./app        # human output on stdout, records in a file
CCUTILS_MACHINE_OUT=inline ./app > run.out      # one stream; records are lines starting with "@ccutils "
```

Human output retains its readable rendering and also carries structured side-channel
records. The 1.0 default is
`CCUTILS_FORMAT=v2`, which selects the path-based human renderer. Set
`CCUTILS_FORMAT=legacy` to retain the legacy renderer, or define
`CCUTILS_FORMAT_LEGACY` when compiling to make legacy the default for that build.

Human logs also include `@ccutils ` structured side-channel lines. These preserve the
complete machine record data, including values that the readable rendering rounds or
does not display. The parser uses those records as the authoritative input when present;
older human logs without them are parsed from their visible text and may be lossy.

## Human v2 output

Section starts and ends use full paths. Values are printed as one JSON value per line,
and timer reports use a table in milliseconds. Gathered MPI text has a rank prefix on each
line. For example:

```text
== solve :: Solve phase ==
residual = 1.5e-09
timers [ms]  (rank 0)
  name   n     avg     sd     min     max     sum
  kernel   5  0.004  0.000  0.004  0.004  0.019

-- all-ranks: status --
[r0] ready
[r1] ready
-- end all-ranks: status --
== end solve ==
```

The corresponding structural lines are `== <path> :: <title> ==` and
`== end <path> ==`. The `[rN] ` prefix is applied per line of gathered text; applications
that need byte-exact text should use the machine stream, which carries text encoding and
trailing-newline metadata.

When every rank in an all-print emits JSON with compatible structure, human output shows
one `schema:` template followed by compact positional values per rank. If payloads differ
in structure or include ordinary text, the rank-prefixed output is used.

## Records

Every record is **exactly one line** of JSON with this envelope:

| Field | Meaning |
|-------|---------|
| `v` | schema version, `2` |
| `type` | record type, see below |
| `section` | full path of the innermost open section (`"solve/spmv"`), or `null` |
| `scope` | `"global"` (one value for the run), `"rank"` (one process) or `"comm"` (a communicator) |
| `rank` | `MPI_COMM_WORLD` rank for `scope:"rank"`; `null` otherwise, and also `null` when the rank is unknown (plain non-MPI run) |
| ... | type-specific fields |
| `seq` | per-process counter, starts at 1 |

Readers must ignore unknown fields and unknown types.

| `type` | Fields | Emitted by |
|--------|--------|------------|
| `section_start` | `title` | `CCUTILS_SECTION_DEF`, `CCUTILS_SUBSECTION_DEF`, `CCUTILS_MPI_SECTION_DEF`, `CCUTILS_MPI_SUBSECTION_DEF` |
| `section_end` | `ok` | the matching `*_END` |
| `warning` | `message` | a section is closed because its matching end macro was skipped |
| `run_start` | `ccutils_version`, `format`, `host`, optional `slurm_job_id` and `mpi` (`library`, `world_size`, `wtime_is_global`, `wtick_s`) | first machine record in each process stream |
| `data_batch` | `values` (JSON object) | non-MPI section JSON and MPI global JSON, written as one record per object when the section ends |
| `data` | `key`, `value` (any JSON) | accepted for older streams; one record per top-level key |
| `json` | `print`, `call`, `value` (any JSON), `trailing_newline`, optional `comm` | a rank JSON print before MPI all-rank compaction |
| `mpi_json_batch` | `print`, `call`, recursive `schema`, `rows` (`rank` plus positional `values`) | MPI all-print when every rank prints JSON with the same object structure and compatible array elements; the structure is stored once |
| `mpi_json_rows` | `rows` (`rank`, `print`, `call`, `comm`, `value`) | MPI all-print when each rank prints JSON but the structures differ |
| `timer` | `name`, `group`, `unit` (`"ms"`), `clock`, `n`, `avg`, `sd`, `min`, `max`, `sum`; MPI summaries add `per_rank`, `rank_stats`, `imbalance`, `slowest_rank`, `ranks_reporting`, `errors` | CPU and MPI timer reports |
| `timer_sample` | `name`, `group`, `unit`, `clock`, `index`, `value`, `last` | `CCUTILS_TIMER_PRINT_LAST` (`last:true`), `CCUTILS_TIMER_PRINT_ALL` |
| `timer_samples` | `name`, `unit`, `part`, `values`, `last` | MPI timer reports when raw samples are requested |
| `timer_agg` | `name`, `op` (`sum`, `sum_avg`, `sum_last`), `unit`, `clock`, `value` | `CCUTILS_TIMER_SUM*_PRINT` |
| `text` | `print`, `call`, `text` or `encoding`/`data`, `trailing_newline`, `part`, `last`, optional `comm` | MPI all-print and `*_ONCE` human writes that are not compacted JSON |

Numbers use the shortest text that reads back as the same value; `nan` and `inf`
become `null`. Strings are valid UTF-8 (invalid bytes become U+FFFD).
UTF-8 text is stored as one JSON string per chunk. Invalid UTF-8 uses base64 in `data`.
Rank JSON print helpers store the parsed payload in `value`, rather than escaping it
inside a `text` field. During MPI all-print, matching nested object structure is compacted
into `mpi_json_batch`, which stores that structure once and one positional value tree per
rank. Array lengths may vary when their elements share a structure. Other MPI all-print
payloads remain `text` records.
The MPI metadata is supplied by the opt-in MPI headers; core headers remain MPI-free.
If an outer section ends while inner sections remain open, each inner section gets a
`section_end` with `ok:false` and a `warning`; the explicitly ended section gets
`ok:true`.

Example (a non-MPI run):

```jsonl
{"v":2,"type":"run_start","section":null,"scope":"global","rank":null,"ccutils_version":"1.0.0","format":"v2","host":"node","seq":1}
{"v":2,"type":"section_start","section":"solve","scope":"global","rank":null,"title":"Solve phase","seq":2}
{"v":2,"type":"timer","section":"solve","scope":"rank","rank":null,"name":"kernel","group":"Timer","unit":"ms","clock":"steady","n":5,"avg":0.0038146,"sd":1e-05,"min":0.003807,"max":0.003832,"sum":0.019073,"seq":3}
{"v":2,"type":"data_batch","section":"solve","scope":"global","rank":null,"values":{"residual":1.5e-09},"seq":4}
{"v":2,"type":"section_end","section":"solve","scope":"global","rank":null,"ok":true,"seq":5}
```

## Where records are written (MPI)

Section records and the global `data` come from rank 0. A timer report on a
communicator is written by that communicator's root: into the main file if the root
is world rank 0, otherwise into the sidecar `<path>.r<rank>` (see OUTPUT.md). Every
file has its own `seq` counter starting at 1.

## Deferred machine-stream feature

Optional batched MPI timer reporting at section end is deferred. Current MPI timer reports
are emitted individually; all records needed to interpret those reports, including
per-rank distributions and optional raw samples, are available in this schema.
