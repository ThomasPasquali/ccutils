# Installation and integration

## Install from a checkout

```sh
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build
cmake --install build
```

CMake obtains nlohmann/json during configuration and installs its headers alongside CCUTILS. This is the only install-time download. MPI and CUDA are disabled by default:

To use the interactive installer with a local clone or fork, pass its checkout root:

```sh
./install.sh --repo /path/to/ccutils
```

The installer builds that checkout as-is and does not pull it. In local-clone mode it does not install the upstream `ccutils-upgrade` command; rerun `install.sh --repo /path/to/ccutils` after updating the clone.

To install a particular upstream branch or release tag, use `--branch` or `--tag` (the default branch is `main`):

```sh
./install.sh --branch feature/my-change
./install.sh --tag v1.0.0
```

These options are mutually exclusive and apply to the managed upstream clone. They cannot be combined with `--repo`.

```sh
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local" \
  -DCCUTILS_ENABLE_MPI=ON \
  -DCCUTILS_ENABLE_CUDA=ON
```

Enabling a module requires a usable MPI C++ implementation or CUDA toolkit. The install exports only targets enabled at configure time.

Install the Python package from the repository root with `python3 -m pip install .`.
This provides `import ccutils` and the `ccutils tree|timers|data|check` commands. Install
`.[pandas]` to enable the optional DataFrame helpers. The CMake install also keeps the
standalone `ccutils_parser.py` compatibility module in Python's site-packages directory;
it exposes the legacy `Document` parser API.

The installer also places the visualization CLI in the installer's `bin` directory. Run it as `ccutils-visualize my_output.log`.

## CMake consumer

```cmake
find_package(ccutils 1 CONFIG REQUIRED)
add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE ccutils::ccutils)
```

Use `ccutils::ccutils_mpi` and/or `ccutils::ccutils_cuda` for enabled optional modules. Point CMake at a nonstandard prefix using `-DCMAKE_PREFIX_PATH=/path/to/prefix` or `-Dccutils_DIR=/path/to/prefix/lib/cmake/ccutils`.

## FetchContent or subdirectory

```cmake
include(FetchContent)
FetchContent_Declare(ccutils GIT_REPOSITORY https://github.com/ThomasPasquali/ccutils.git GIT_TAG v1.0.0)
FetchContent_MakeAvailable(ccutils)
target_link_libraries(myapp PRIVATE ccutils::ccutils)
```

For `add_subdirectory`, add the repository to the parent project and link the same exported build target. Set `CCUTILS_ENABLE_MPI` or `CCUTILS_ENABLE_CUDA` before adding it.

## Make consumer

```make
CCUTILS_PREFIX ?= $(HOME)/.local
CXXFLAGS += -I$(CCUTILS_PREFIX)/include
```

Then include `<ccutils/ccutils.hpp>` or individual headers. Add `-DCCUTILS_ENABLE_MPI` and link through `mpicxx` for MPI. Compile CUDA sources with `nvcc` (or a CUDA enabled CMake target) and define `CCUTILS_ENABLE_CUDA`.

JSON output is enabled by default. To use core headers without JSON, define `CCUTILS_NO_JSON` before including CCUTILS; section JSON put macros then become no-ops. `macros.hpp` requires C++11 or newer.

Choose ANSI color intensity per consuming target with `CCUTILS_COLOR_LEVEL=0` (none) through `3` (bright foregrounds and colored CCUTILS markers). Level 2 is the default; `CCUTILS_NO_COLOR` remains an alias for level 0.
