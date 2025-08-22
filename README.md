# CC Utils

Header-only collection of C/C++/CUDA/MPI utilities (macros, timers, etc.).

## Features
- **CPU Utilities**: High-resolution timer macros.
- **CUDA Utilities**: CUDA event-based timer macros.
- **MPI Utilities**: Optional MPI helper macros.

Organized into optional "packages":
- `ccutils::ccutils` (always available)
- `ccutils::ccutils_cuda` (requires CUDA, enable with `-DCCUTILS_ENABLE_CUDA=ON`)
- `ccutils::ccutils_mpi` (requires MPI, enable with `-DCCUTILS_ENABLE_MPI=ON`)

---

## Build and Install

```bash
git clone https://github.com/ThomasPasquali/ccutils.git
cd ccutils
cmake -B build -S . -DCCUTILS_ENABLE_CUDA=ON -DCCUTILS_ENABLE_MPI=ON
cmake --build build
cmake --install build --prefix /your/install/path
```

You can omit `-DCCUTILS_ENABLE_XXX=ON` if you do not need to enable a specific package.

## Usage in CMake Projects

```cmake
find_package(ccutils REQUIRED)

# Link to your project
target_link_libraries(myapp PRIVATE ccutils::ccutils)        # Base macros
target_link_libraries(myapp PRIVATE ccutils::ccutils_cuda)   # CUDA (if enabled)
target_link_libraries(myapp PRIVATE ccutils::ccutils_mpi)    # MPI  (if enabled)

```

Or via `FetchContent` (no need to manually download, build and install):

```cmake
# Download automatically with CMake
include(FetchContent)
set(CCUTILS_ENABLE_CUDA ON  CACHE BOOL "")
set(CCUTILS_ENABLE_MPI  ON  CACHE BOOL "")
FetchContent_Declare(
  ccutils
  GIT_REPOSITORY https://github.com/ThomasPasquali/ccutils.git
  # GIT_TAG v1.0.0
)
FetchContent_MakeAvailable(ccutils)

# Link to your project
target_link_libraries(myapp PRIVATE ccutils::ccutils)        # Base macros
target_link_libraries(myapp PRIVATE ccutils::ccutils_cuda)   # CUDA (if enabled)
target_link_libraries(myapp PRIVATE ccutils::ccutils_mpi)    # MPI  (if enabled)
```

## Usage in Makefile Projects

This library is **header-only**. You just need to add the `include/` folder to your compiler flags.

```make
# Always
CXXFLAGS += -I/path/to/cc-utils/include

# If using CUDA utilities
CXXFLAGS += -I/path/to/cc-utils/include -DCCUTILS_ENABLE_CUDA

# If using MPI utilities
CXXFLAGS += -I/path/to/cc-utils/include $(shell mpicxx --showme:compile) -DCCUTILS_ENABLE_MPI
LDLIBS   += $(shell mpicxx --showme:link)
```

## Examples

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