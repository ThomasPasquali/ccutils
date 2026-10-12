# CCUTILS

CCUTILS is a header-only collection of small C++ utilities: structured output, timers, common macros, command-line argument parsing, and optional MPI and CUDA helpers. The core API requires C++11; examples use C++17. Optional modules require their respective toolkits.

## Features

- Named output sections with JSON records (or plain sections with `CCUTILS_NO_JSON`).
- CPU timers with per-run values and summary statistics; MPI and CUDA timer headers are optional.
- MPI rank output and MPI/CUDA error-checking helpers.
- ANSI color constants, assertions, and math helpers.
- The bundled single-header `easyargs` argument parser.
- Importable Python parser for sections, JSON, MPI blocks, and timer records (`import ccutils_parser` after CMake installation).

## Fast Install

To automatically download and install the library in `$HOME/.local`, run:

```bash
# Install the default main branch
wget -qO- https://raw.githubusercontent.com/ThomasPasquali/ccutils/main/install.sh | env bash

# Or select a branch or release tag
wget -qO- https://raw.githubusercontent.com/ThomasPasquali/ccutils/main/install.sh | env bash -s -- --branch specific_branch
wget -qO- https://raw.githubusercontent.com/ThomasPasquali/ccutils/main/install.sh | env bash -s -- --tag v1.0.0
```

> [!IMPORTANT]
> Currently, CMake is required for installation.

## Manual Install

CCUTILS uses CMake to install. CMake downloads the header-only JSON dependency while configuring and requires Python 3.8+ to install the importable output parser. After installation, C++ consumers need no separate JSON package.

```sh
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build
cmake --install build
```

The interactive installer can also use an existing local checkout: `./install.sh --repo /path/to/ccutils`. For the managed upstream clone, choose a branch or release tag with `./install.sh --branch feature/name` or `./install.sh --tag v1.0.0` (default branch: `main`).

Enable optional modules with `-DCCUTILS_ENABLE_MPI=ON` or `-DCCUTILS_ENABLE_CUDA=ON`. See [installation and integration](docs/INSTALL.md) for Make, CMake, and FetchContent examples.

## Minimal use

```cpp
#include <ccutils/ccutils.hpp>

int main() {
    CCUTILS_SECTION_DEF(run, "Run information")
    CCUTILS_SECTION_JSON_PUT(run, "status", "ok")
    CCUTILS_SECTION_END(run)
}
```

Link `ccutils::ccutils` in CMake, or add the install `include` directory to your Make compiler flags. Full macro and format behavior is documented in [the API guide](docs/API.md).

## Output and parsing

Section markers and JSON block markers are stable line-oriented delimiters. JSON records are valid JSON and emitted only for nonempty sections. Timer macros emit versioned key/value records only when there are samples; records can be mixed with free-form output and selected by their `CCUTILS_TIMER` prefix. Set `CCUTILS_COLOR_LEVEL` from 0 (no color) to 3 (bright foreground and background colors) to choose ANSI color intensity. See [the API guide](docs/API.md) and [parser documentation](docs/examples/PARSER.md).

## Examples

Build examples with CMake or Make from `examples/`. See [prints](docs/examples/PRINTS.md), [timers](docs/examples/TIMERS.md), [MPI](docs/examples/MPI_MACROS.md), and [CUDA](docs/examples/CUDA_MACROS.md).
