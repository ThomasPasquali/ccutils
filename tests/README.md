# ccutils tests

Golden-output tests: small programs are compiled in several configurations,
run (under `mpirun` for MPI), their timing numbers are replaced by `<f>`, and
stdout is compared byte-for-byte with `tests/golden/*.txt`.

```bash
python3 tests/run_tests.py            # run all
python3 tests/run_tests.py -k mpi -v  # filter by id, show warnings and diffs
python3 tests/run_tests.py --update   # regenerate goldens (review the diff!)
python3 -m unittest tests/test_parser.py   # Python parser (fast, no compiler or MPI)
cmake -B build -DCCUTILS_BUILD_TESTS=ON && ctest --test-dir build   # everything via ctest
```

Requirements: a C++17 compiler, nlohmann/json, and for the MPI cases an MPI
installation. Select it with `MPICXX` / `MPIRUN` (e.g. `mpicxx.openmpi` /
`mpirun.openmpi`). Before comparing anything the runner checks that the
`mpicxx`/`mpirun` pair really launches N cooperating ranks (a mismatched pair
starts N independent singleton worlds and would silently produce garbage).

## Rules

* The goldens describe the **legacy** output format. Refactors must not change
  them; a diff in a golden file is a behaviour change and must be intentional.
* Goldens must not depend on rank-interleaving races: every line is printed by
  rank 0, or separated from rank 0's output by `CCUTILS_FLUSH_WAIT`.
* A case listed in `known_broken` is skipped and reported with the reason;
  remove the entry in the commit that fixes the bug.

## Helpers

* `python3 tools/check_todos.py` keeps the `TODO(<ID>)` markers in the code and the items of
  `docs/TODO.md` in sync (`--list` shows every marker with file:line).
* `tests/stress_golden.sh <program.cpp> <np> <runs> <golden>` runs one MPI program many times
  and compares each run with its golden; this is how the only flaky golden was found. It is
  incomplete (no machine-stream mode).
* `tests/fixtures/` holds real logs (human log + machine records of the same run) used by the
  parser tests.
* `.github/workflows/ci.yml` is a draft that has never been executed (P0-ci).

