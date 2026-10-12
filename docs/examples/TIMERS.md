# Timers

Timer definitions retain all samples until the timer's vector goes out of scope. CPU and CUDA times are milliseconds. See [core timer API](../API.md#cpu-timers) for statistic fields and printing variants.

Timer print macros emit `CCUTILS_TIMER v=1` key/value records with colors controlled by `CCUTILS_COLOR_LEVEL`. They can be mixed with normal application output and parsed by the CCUTILS Python parser, which removes ANSI color codes before reading records.

Print calls emit nothing until the timer has at least one sample. The same applies to MPI timer summaries and to aggregate print macros when every input timer is empty.

From the `examples` directory:

```bash
# Install CCUTILS on the system
# If needed, change the branch name
wget -qO- https://raw.githubusercontent.com/ThomasPasquali/ccutils/main/install.sh | env bash

# Using CMake
cmake -B build
# If you want to include CUDA and MPI:
cmake -B build -DWITH_CUDA=ON -DWITH_MPI=ON
cmake --build build -t timers

# OR Using Makefile
make timers
# If you want to include CUDA and MPI:
make WITH_MPI=1 WITH_CUDA=1 timers
mkdir -p build
mv timers build

# Run (normal)
./build/timers
# Run (MPI)
mpirun -np 2 ./build/timers
```
