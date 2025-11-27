cmake_minimum_required(VERSION 3.18)
project(ccutils LANGUAGES C CXX)

# Options to toggle optional packages
option(CCUTILS_ENABLE_CUDA "Enable CUDA utilities" OFF)
option(CCUTILS_ENABLE_MPI  "Enable MPI utilities"  OFF)

# ---------------- nlohmann/json dependency ----------------
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(json)

# ---------------- Base package ----------------
add_library(ccutils INTERFACE)
target_include_directories(ccutils INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(ccutils INTERFACE nlohmann_json::nlohmann_json)

# ---------------- CUDA package ----------------
if(CCUTILS_ENABLE_CUDA)
    message("CCUTILS: Enabling CUDA module")
    enable_language(CUDA)
    add_library(ccutils_cuda INTERFACE)
    target_include_directories(ccutils_cuda INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    # target_link_libraries(ccutils_cuda INTERFACE CUDA::cudart)
    target_compile_definitions(ccutils_cuda INTERFACE CCUTILS_ENABLE_CUDA)
    target_link_libraries(ccutils_cuda INTERFACE nlohmann_json::nlohmann_json)
endif()

# ---------------- MPI package ----------------
if(CCUTILS_ENABLE_MPI)
    message("CCUTILS: Enabling MPI module")
    find_package(MPI REQUIRED)  # Provides MPI::MPI_C and MPI::MPI_CXX
    add_library(ccutils_mpi INTERFACE)
    target_include_directories(ccutils_mpi INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(ccutils_mpi INTERFACE MPI::MPI_C MPI::MPI_CXX nlohmann_json::nlohmann_json)
    target_compile_definitions(ccutils_mpi INTERFACE CCUTILS_ENABLE_MPI)
endif()

# ---------------- Namespace Aliases ----------------
add_library(ccutils::ccutils ALIAS ccutils)

if(TARGET ccutils_mpi)
    add_library(ccutils::ccutils_mpi ALIAS ccutils_mpi)
endif()

if(TARGET ccutils_cuda)
    add_library(ccutils::ccutils_cuda ALIAS ccutils_cuda)
endif()

# ---------------- Installation ----------------
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
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
endif()