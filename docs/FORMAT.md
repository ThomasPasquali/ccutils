# Machine-readable output (JSON Lines, schema v2)

Enable with `CCUTILS_MACHINE_OUT` (see [OUTPUT.md](OUTPUT.md)):

```bash
CCUTILS_MACHINE_OUT=file:run.jsonl ./app        # human output on stdout, records in a file
CCUTILS_MACHINE_OUT=inline ./app > run.out      # one stream; records are lines starting with "@ccutils "
```

The human output is not affected by enabling the machine stream. `CCUTILS_FORMAT=v2`
selects the path-based human renderer; `legacy` remains the default.

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
| `data` | `key`, `value` (any JSON) | one record per top-level key of the section JSON (`CCUTILS_SECTION_JSON_PUT`, `CCUTILS_SUBSECTION_JSON_PUT`, `CCUTILS_MPI_GLOBAL_JSON_PUT`, `CCUTILS_JSON_PRINT`), written when the section ends |
| `timer` | `name`, `group`, `unit` (`"ms"`), `clock`, `n`, `avg`, `sd`, `min`, `max`, `sum`; MPI summaries add `per_rank`, `rank_stats`, `imbalance`, `slowest_rank`, `ranks_reporting`, `errors` | CPU and MPI timer reports |
| `timer_sample` | `name`, `group`, `unit`, `clock`, `index`, `value`, `last` | `CCUTILS_TIMER_PRINT_LAST` (`last:true`), `CCUTILS_TIMER_PRINT_ALL` |
| `timer_samples` | `name`, `unit`, `part`, `values`, `last` | MPI timer reports when raw samples are requested |
| `timer_agg` | `name`, `op` (`sum`, `sum_avg`, `sum_last`), `unit`, `clock`, `value` | `CCUTILS_TIMER_SUM*_PRINT` |
| `text` | `print`, `call`, `lines` or `encoding`/`data`, `trailing_newline`, `part`, `last`, optional `comm` | MPI all-print and `*_ONCE` human writes |

Numbers use the shortest text that reads back as the same value; `nan` and `inf`
become `null`. Strings are valid UTF-8 (invalid bytes become U+FFFD).
The MPI metadata is supplied by the opt-in MPI headers; core headers remain MPI-free.
If an outer section ends while inner sections remain open, each inner section gets a
`section_end` with `ok:false` and a `warning`; the explicitly ended section gets
`ok:true`.

Example (a non-MPI run):

```jsonl
{"v":2,"type":"run_start","section":null,"scope":"global","rank":null,"ccutils_version":"1.0.0","format":"legacy","host":"node","seq":1}
{"v":2,"type":"section_start","section":"solve","scope":"global","rank":null,"title":"Solve phase","seq":2}
{"v":2,"type":"timer","section":"solve","scope":"rank","rank":null,"name":"kernel","group":"Timer","unit":"ms","clock":"steady","n":5,"avg":0.0038146,"sd":1e-05,"min":0.003807,"max":0.003832,"sum":0.019073,"seq":3}
{"v":2,"type":"data","section":"solve","scope":"global","rank":null,"key":"residual","value":1.5e-09,"seq":4}
{"v":2,"type":"section_end","section":"solve","scope":"global","rank":null,"ok":true,"seq":5}
```

## Where records are written (MPI)

Section records and the global `data` come from rank 0. A timer report on a
communicator is written by that communicator's root: into the main file if the root
is world rank 0, otherwise into the sidecar `<path>.r<rank>` (see OUTPUT.md). Every
file has its own `seq` counter starting at 1.

## Remaining machine-stream work

The current records include buffered all-print text and MPI timer rank details. Batched
MPI timer reporting at section end remains deferred; see [TODO.md](TODO.md).
