cmake_minimum_required(VERSION 3.18)
project(ccutils_examples LANGUAGES CXX)

option(WITH_CUDA "Enable CUDA utilities" OFF)
option(WITH_MPI  "Enable MPI utilities"  OFF)

# ---------------------------------------------------------------
# Find ccutils (required)
# ---------------------------------------------------------------
# If ccutils provides these variables, set them before find_package
set(CCUTILS_ENABLE_CUDA "${WITH_CUDA}")
set(CCUTILS_ENABLE_MPI  "${WITH_MPI}")

find_package(ccutils REQUIRED)

# ---------------------------------------------------------------
# Optional MPI
# ---------------------------------------------------------------
if(WITH_MPI)
    message(STATUS "MPI support enabled")
    find_package(MPI REQUIRED COMPONENTS CXX)
endif()

# ---------------------------------------------------------------
# Optional CUDA
# ---------------------------------------------------------------
if(WITH_CUDA)
    message(STATUS "CUDA support enabled")
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")
endif()

# ---------------------------------------------------------------
# Example builder helper
# ---------------------------------------------------------------
function(add_example name)
    add_executable(${name} ${name}.cpp)

    # Always link base ccutils
    target_link_libraries(${name} PRIVATE ccutils::ccutils)

    # If MPI support is enabled in this build
    if(WITH_MPI)
        target_link_libraries(${name} PRIVATE MPI::MPI_CXX ccutils::ccutils_mpi)
        target_compile_definitions(${name} PRIVATE WITH_MPI)
    endif()

    # If CUDA support is enabled in this build
    if(WITH_CUDA)
        target_link_libraries(${name} PRIVATE ccutils::ccutils_cuda CUDA::cudart)
        target_compile_definitions(${name} PRIVATE WITH_CUDA)
    endif()

    target_compile_options(${name} PRIVATE -O3 -Wall)
endfunction()

# ---------------------------------------------------------------
# Examples
# ---------------------------------------------------------------
add_example(prints)
add_example(timers)
