# Manual Installation Guide

```bash
git clone https://github.com/ThomasPasquali/ccutils.git
cd ccutils
cmake -B build -S . -DCCUTILS_ENABLE_CUDA=ON -DCCUTILS_ENABLE_MPI=ON
cmake --build build
cmake --install build --prefix /your/install/path
```

You can omit `-DCCUTILS_ENABLE_XXX=ON` if you do not need to enable a specific package.

## Usage in CMake Projects

### Fetch and Build Automatically (Recommended)

Via `FetchContent`, no need to manually download, build and install.

```cmake
# Enable CUDA and MPI (optional)
set(CCUTILS_ENABLE_CUDA ON  CACHE BOOL "")
set(CCUTILS_ENABLE_MPI  ON  CACHE BOOL "")

# Download automatically with CMake
include(FetchContent)
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

### Add CMake Subdirectory

```cmake
# Enable CUDA and MPI (optional)
set(CCUTILS_ENABLE_CUDA ON CACHE BOOL "Enable CUDA support in ccutils")
set(CCUTILS_ENABLE_MPI ON CACHE BOOL "Enable MPI support in ccutils")

add_subdirectory(ccutils)

# Link to your project
target_link_libraries(myapp PRIVATE ccutils::ccutils ccutils::ccutils_cuda ccutils::ccutils_mpi)
```

### Link Installed Library

```cmake
find_package(ccutils REQUIRED) # Here you need to ensure CMake knows the installation path

# Link to your project
target_link_libraries(myapp PRIVATE ccutils::ccutils)        # Base macros
target_link_libraries(myapp PRIVATE ccutils::ccutils_cuda)   # CUDA (if enabled)
target_link_libraries(myapp PRIVATE ccutils::ccutils_mpi)    # MPI  (if enabled)

```
> [!NOTE]  
> Ensure to pass `ccutils` installation path to CMake.  
> Option 1: add `-Dccutils_DIR=/your/install/path` to your build command.  
> Option 2: append to CMAKE PREFIX PATH by adding to your CMake file `list(APPEND CMAKE_PREFIX_PATH "/your/install/path")`  


## Usage in Makefile Projects

This library is **header-only**. You just need to add the `include/` folder to your compiler flags.

> [!NOTE]
> CMake is required to instal CCUTILS anyway

First, make sure to have CCUTILS code on your system. Then, add the following to your makefile:

```make
# Substitute `path/to` with your ccutils path
CCUTILS = path/to/ccutils/install/include
CXXFLAGS += -I$(CCUTILS)

$(CCUTILS):
	cd path/to/ccutils && \
	cmake -S . -B build -DCCUTILS_ENABLE_MPI=ON && \
	cmake --build build && \
	cmake --install build --prefix ./install

# Ensure targets that depend on CCUTILS will build it if needed
my_target: ... $(CCUTILS) ...
  ...
```