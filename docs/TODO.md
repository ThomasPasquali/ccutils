# Remaining work

This file lists everything that is designed but not finished. It is the companion of
[DESIGN.md](DESIGN.md) (the design, with a status section at the top) and of the
`TODO(<ID>)` comments in the code: every active ID below appears at the place in the source
where the work belongs (`grep -rn "TODO(P" include parser tests`), and
`python3 tools/check_todos.py` fails if a code marker has no entry here or the other
way round. Remove the code marker and the entry together when you finish an item.

Legend: **Design** = section of DESIGN.md. **Where** = files to change.
**Done when** = what must be true (and tested) before you delete the item.

## Where things stand

| Phase (DESIGN.md §9) | State |
|----------------------|-------|
| 0 Safety net | Golden-output harness, bug fixes, reserved identifiers: **done**. CI, MPICH, CUDA, a few golden gaps: open (P0-*) |
| 1 Sinks and configuration | `emitter.hpp`, env vars, file/sidecar rules, colors, routed macros, `ccutils::init`, and the documented single-thread contract: **done** |
| 2 Context, sections, machine stream | Section registry, name-based macros, records for sections / global and rank data / text / CPU and MPI timers, `run_start`, double CPU timers, RAII sections: **done** |
| 3 Human renderer v2 | **done** for section, key/value, timer-table and rank-text output |
| 4 MPI integration | framed H/M all-print, chunked gather, MPI timer distributions, rank policies, sample records and warm-up exclusion: **implemented**; MPI runtime matrix was not run per request. Optional section timer batching is deferred. |
| 5 Parser v2 | model, JSONL / inline / legacy / human-v2 front ends, CLI, text chunk reassembly and MPI timer rank queries: **done** |
| 6 Docs and switch | partly (docs exist for output, format, multi-file, parser) (P6-*) |

## Ground rules (learned while implementing; read before touching anything)

1. **Legacy output is byte-identical until `CCUTILS_FORMAT=v2`.** The goldens in
   `tests/golden/*.txt` are the gate. A changed legacy golden is a behaviour change.
   Machine-stream goldens (`*.jsonl`) may change when you add records; review the diff.
2. **Never put a `//` comment inside a multi-line macro** (the backslash continuation
   makes the comment swallow the next line). Put comments above the macro.
3. **Keep the trailing `;` inside the legacy macros.** The docs call them without a
   semicolon; removing it breaks users. Multi-statement macros must still be one
   statement (`do { } while (0);`).
4. **No MPI (or any `#ifdef CCUTILS_ENABLE_MPI`) inside inline functions of core
   headers.** MPI and non-MPI translation units would see different definitions of the
   same inline function (ODR violation). MPI knowledge reaches the core through
   `ccutils::set_world_rank()` and the macros in `mpi/`.
5. **Do not add a collective to an existing macro** unless it is opt-in (DESIGN.md C4).
   A hidden collective deadlocks programs that call the macro from a subset of ranks.
6. **Macro parameters shadow names.** A parameter called `section` rewrote
   `ccutils::section(...)` inside the macro body. Use parameter names like
   `section_name`, and never name a parameter after something you call.
7. **Stdout is block-buffered when it is a pipe.** In multi-rank tests, anything printed
   by a rank other than 0 races with rank 0's buffered output unless both flush and wait
   (`CCUTILS_FLUSH_WAIT`) between them. This caused the only flaky golden so far.
8. **A mismatched `mpicxx`/`mpirun` pair silently starts N singleton worlds** (seen with
   Ubuntu's MPICH 4.2, which links PMIx but is launched by Hydra). The runner checks this
   before comparing anything; keep that check.
9. Goldens contain `__LINE__` (banners) and timings; `normalise()` in
   `tests/run_tests.py` replaces both. New volatile fields must be normalised there.
10. In `unittest`, do not call an attribute `run` (it shadows `TestCase.run`).
11. Tests must be able to fail. When you add one, break the code once on purpose and see
    it fail (this found a real mistake twice).

---

## Phase 0: safety net

### P0-ci
**Design:** §5.5, §8, §9 (Phase 0). **Where:** `.github/workflows/ci.yml` (a draft exists,
never executed), `tests/run_tests.py`, `tests/CMakeLists.txt`.
1. Run the draft workflow on GitHub and fix it. Matrix: {Open MPI, MPICH} x {g++, clang++}
   x C++ {11, 17, 20}. Install `libopenmpi-dev` / `libmpich-dev`, `nlohmann-json3-dev`.
2. Select the toolchain with `MPICXX` / `MPIRUN` (the runner already supports both and
   adds `--oversubscribe` for Open MPI).
3. Add a job with `-fsanitize=address,undefined` for the non-MPI cases, and one with
   `CCUTILS_TEST_WERROR=1` (see P0-warnings).
4. Run `python3 -m unittest tests/test_parser.py` in CI too (`ctest` already registers it).
**Done when:** CI is green on both MPI implementations and fails when a golden changes.

### P0-mpich
**Where:** `tests/run_tests.py` (launcher self-check), CI.
Run the whole suite with MPICH on a machine where `mpirun -n 2` really gives 2 ranks
(the sandbox this was written in could not; see ground rule 8). Goldens were generated
with Open MPI 4.1.6; if MPICH differs in anything ccutils prints, find out why before
adding per-implementation goldens (there should be no difference).
**Done when:** all cases pass on MPICH 4.x.

### P0-cuda
**Design:** §1 (reserved identifiers). **Where:** `include/ccutils/cuda/*.hpp`.
The CUDA headers got the same mechanical rename of `__...` identifiers as the others but
were never compiled (no `nvcc` available). Also bring them to parity with the CPU timers:
registration by name, a `CCUTILS_CUDA_TIMER_USE` macro, `ccutils::cuda_timer(name)`.
**Done when:** a small CUDA program with a timer defined in one file and used in another
builds and runs, and a golden for it exists (non-CUDA CI skips it).

### P0-golden-gaps
**Where:** `tests/programs/`, `tests/golden/`.
Add goldens for what has none: `CCUTILS_MPI_BUFFERED_PRINT_*` (writes temp files in the
cwd; the runner already checks for leftover files), `CCUTILS_MPI_PROCESS_PRINT` /
`CCUTILS_MPI_COMMUNICATOR_PRINT` with np > 1 (order across ranks is only deterministic with
`CCUTILS_FLUSH_WAIT` between prints), `CCUTILS_MPI_JSON_PRINT` with pretty JSON, and a rank
that prints nothing in every all-print. Add coverage for the framed transport when MPI tests can run.
**Done when:** each macro family above has a golden for np = 1, 2, 4.

## Phase 1: sinks and configuration

## Phase 2: context, sections and machine stream

## Phase 3: human renderer v2

## Phase 4: MPI integration (highest risk)

## Phase 5: parser v2

Optional deferred item: batched reporting of all MPI timers at section end, guarded by a
cross-rank registration consistency check. The current API reports timers individually.

## Phase 6: docs and switch

### P6-default-v2
**Design:** §7. **Where:** `include/ccutils/emitter.hpp` (`apply`), `README.md`,
`docs/OUTPUT.md`, `docs/FORMAT.md`.
After P3/P4/P5 are done: make `v2` the default in 1.0 (`-DCCUTILS_FORMAT_LEGACY` and
`CCUTILS_FORMAT=legacy` keep the old output), flip colors to `auto` (P1-color), deprecate the
global `TimerStats` alias, write the migration guide, and complete `docs/FORMAT.md` (the
"Not in the machine stream yet" list must be empty by then).
**Done when:** the migration guide exists and a legacy-mode CI job still passes.

---

## Open questions from the design (DESIGN.md §10) and a recommended default

1. Rank-local data in the human stream: rank x key table or one `[rN]` line per value?
   The current v2 renderer uses `[rN]` lines; a rank x key table remains a possible refinement.
2. pandas: optional (already implemented that way). Keep it optional.
3. Communicators other than WORLD for all-print: yes, via `..._COMM` variants.
4. Which MPI / launcher combinations matter: decides how much of the §5.5 matrix needs a real
   cluster instead of containers.
5. Per-rank text above ~1 GiB: H/M frame transport gathers in bounded chunks.
6. Gutter `| ` for rank-0 text: a different marker is a one-line change in the renderer; the
   parser must accept the same one.
7. `imbalance` uses the per-rank total (`sum`); the per-rank mean is in the record as well.
8. Sidecar files (OUTPUT.md) vs "rank 0 only" for file sinks: sidecars are implemented and
   tested; switching policy means changing `Sink::resolve()` and the scenario goldens.
9. `CCUTILS_TIMER_RANKS` default `summary`.
