cmake_minimum_required(VERSION 3.18)
project(ccutils LANGUAGES C CXX)

# Options to toggle optional packages
option(CCUTILS_ENABLE_CUDA "Enable CUDA utilities" OFF)
option(CCUTILS_ENABLE_MPI  "Enable MPI utilities"  OFF)

# ---------------- Base package ----------------
add_library(ccutils INTERFACE)
target_include_directories(ccutils INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ---------------- CUDA package ----------------
if(CCUTILS_ENABLE_CUDA)
    message("Enabling CUDA module")
    enable_language(CUDA)
    add_library(ccutils_cuda INTERFACE)
    target_include_directories(ccutils_cuda INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    # target_link_libraries(ccutils_cuda INTERFACE CUDA::cudart)
    target_compile_definitions(ccutils_cuda INTERFACE CCUTILS_ENABLE_CUDA)
endif()

# ---------------- MPI package ----------------
if(CCUTILS_ENABLE_MPI)
    message("Enabling MPI module")
    find_package(MPI REQUIRED)  # Provides MPI::MPI_C and MPI::MPI_CXX
    add_library(ccutils_mpi INTERFACE)
    target_include_directories(ccutils_mpi INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(ccutils_mpi INTERFACE MPI::MPI_C MPI::MPI_CXX)
    target_compile_definitions(ccutils_mpi INTERFACE CCUTILS_ENABLE_MPI)
endif()

# ---------------- Installation ----------------
install(DIRECTORY include/ DESTINATION include)
install(TARGETS ccutils EXPORT ccutilsTargets)

if(CCUTILS_ENABLE_CUDA)
    install(TARGETS ccutils_cuda EXPORT ccutilsTargets)
endif()

if(CCUTILS_ENABLE_MPI)
    install(TARGETS ccutils_mpi EXPORT ccutilsTargets)
endif()

install(EXPORT ccutilsTargets
    FILE ccutilsConfig.cmake
    NAMESPACE ccutils::
    DESTINATION lib/cmake/ccutils
)
