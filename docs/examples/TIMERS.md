# Timers

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
