### CPU Example

```bash
# Install ccutils in `./install`
cmake -B build -S .
cmake --build build
cmake --install build --prefix ./install

cd examples/cpu

# Using CMake
cmake -B build -S .
cmake --build build
./build/cpu_timers

# Using Makefile
make
./cpu_timers
```

### CUDA Example

```bash
# Install ccutils in `./install` enabling CUDA
cmake -B build -S . -DCCUTILS_ENABLE_CUDA=ON
cmake --build build
cmake --install build --prefix ./install

cd examples/cuda

# Using CMake
cmake -B build -S .
cmake --build build
./build/cuda_example

# Using Makefile
make
./cuda_example
```

### MPI Example

```bash
# Install ccutils in `./install` enabling CUDA
cmake -B build -S . -DCCUTILS_ENABLE_MPI=ON
cmake --build build
cmake --install build --prefix ./install

cd examples/mpi

# Using CMake
cmake -B build -S .
cmake --build build
mpirun -n 4 ./build/mpi_example

# Using Makefile
make
mpirun -n 4 ./build/mpi_example
```