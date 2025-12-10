# CC Utils

Header-only collection of C/C++/CUDA/MPI utilities.

## Features

- **Structured Prints**. Organize output in sections and organize you output in Json-like style
- **Domain-specific Macros**. CUDA errors checks, MPI prints that do not overlap.
- **Timers**. Standardized prints and automatic statistics. Also specialized for CUDA and MPI.
- **Python Output Parser Library**. Using the print macros allows for automatic parsing
- **ANSI Colors**.
- **Common Macros**. Asserts, common math operations

Organized into optional packages:
- `ccutils` (always available)
- `ccutils_cuda` (requires CUDA, enable with `-DCCUTILS_ENABLE_CUDA=ON`)
- `ccutils_mpi` (requires MPI, enable with `-DCCUTILS_ENABLE_MPI=ON`)

## Quick Installation

```bash
# FIXME
wget .... | bash
```

If you prefer ot install manually, please refer to [docs/INSTALL.md](docs/INSTALL.md)

## Examples

* [Timers](docs/examples/TIMERS.md)
* [Prints](docs/examples/PRINTS.md) + [Python Parser](docs/examples/PARSER.md)
* [Cuda-Specific](docs/examples/CUDA_MACROS.md)
* [MPI-Specific](docs/examples/MPI_MACROS.md)