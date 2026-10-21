#!/bin/bash
# Run one MPI test program many times and compare every run with its golden file.
# This is how the only flaky golden so far was found (rank 0's buffered stdout racing
# with another rank, see docs/TODO.md ground rule 7). INCOMPLETE: no machine-stream mode.
#
#   MPICXX=mpicxx.openmpi MPIRUN=mpirun.openmpi \
#     tests/stress_golden.sh tests/programs/golden_mpi.cpp 2 20 tests/golden/mpi.default.np2.txt
#
# Failing outputs are kept in $TMPDIR/ccutils_stress/fail_<n>.txt.
set -u
src=$1; np=$2; runs=$3; golden=$4
root=$(cd "$(dirname "$0")/.." && pwd)
work=${TMPDIR:-/tmp}/ccutils_stress; mkdir -p "$work"; rm -f "$work"/fail_*
${MPICXX:-mpicxx} -std=c++17 -O1 -I"$root/include" -DCCUTILS_ENABLE_MPI "$src" -o "$work/bin" || exit 2
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
extra=""; "${MPIRUN:-mpirun}" --version 2>&1 | grep -q -E "Open MPI|OpenRTE" && extra="--oversubscribe"
fails=0
for i in $(seq 1 "$runs"); do
  (cd "$work" && "${MPIRUN:-mpirun}" $extra -n "$np" ./bin 2>/dev/null) \
    | sed -E -e 's/((avg|stddev|min|max|sum|elapsed)_ms)=[-+0-9.eEinfa]+/\1=<f>/g' \
             -e 's/\bLine [0-9]+\b/Line <n>/g' > "$work/out.txt"
  if ! cmp -s "$work/out.txt" "$golden"; then fails=$((fails+1)); cp "$work/out.txt" "$work/fail_$i.txt"; fi
done
echo "np=$np: $fails failures / $runs"
[ "$fails" -eq 0 ]
