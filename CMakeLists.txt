cmake_minimum_required(VERSION 3.18)
project(ccutils VERSION 1.0.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FetchContent)

# Keep URL extraction timestamps current on CMake versions that support CMP0135.
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

option(CCUTILS_ENABLE_CUDA "Enable CUDA utilities" OFF)
option(CCUTILS_ENABLE_MPI "Enable MPI utilities" OFF)

# JSON output is part of the default API. Vendor the small header-only
# dependency into the install tree so installed Make and CMake users work alike.
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.10.5/json.tar.xz
  # The dependency is header-only here; don't add its legacy CMake project.
  SOURCE_SUBDIR single_include)
FetchContent_MakeAvailable(nlohmann_json)

add_library(ccutils INTERFACE)
add_library(ccutils::ccutils ALIAS ccutils)
target_include_directories(ccutils INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${nlohmann_json_SOURCE_DIR}/single_include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(CCUTILS_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  add_library(ccutils_mpi INTERFACE)
  add_library(ccutils::ccutils_mpi ALIAS ccutils_mpi)
  target_include_directories(ccutils_mpi INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_link_libraries(ccutils_mpi INTERFACE MPI::MPI_CXX ccutils)
  target_compile_definitions(ccutils_mpi INTERFACE CCUTILS_ENABLE_MPI)
endif()

if(CCUTILS_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(ccutils_cuda INTERFACE)
  add_library(ccutils::ccutils_cuda ALIAS ccutils_cuda)
  target_include_directories(ccutils_cuda INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_link_libraries(ccutils_cuda INTERFACE CUDA::cudart ccutils)
  target_compile_definitions(ccutils_cuda INTERFACE CCUTILS_ENABLE_CUDA)
endif()

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  find_package(Python3 3.8 COMPONENTS Interpreter REQUIRED)

  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(DIRECTORY ${nlohmann_json_SOURCE_DIR}/single_include/nlohmann
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(TARGETS ccutils EXPORT ccutilsTargets)
  if(CCUTILS_ENABLE_MPI)
    install(TARGETS ccutils_mpi EXPORT ccutilsTargets)
  endif()
  if(CCUTILS_ENABLE_CUDA)
    install(TARGETS ccutils_cuda EXPORT ccutilsTargets)
  endif()

  install(EXPORT ccutilsTargets FILE ccutilsTargets.cmake
    NAMESPACE ccutils:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ccutils)

  configure_package_config_file(cmake/ccutilsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ccutilsConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ccutils)
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ccutilsConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ccutilsConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ccutilsConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ccutils)

  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "import sysconfig; print(sysconfig.get_path('purelib', vars={'base': '_CCUTILS_PREFIX_', 'platbase': '_CCUTILS_PREFIX_'}).replace('_CCUTILS_PREFIX_', '', 1).lstrip('/\\\\'))"
    RESULT_VARIABLE CCUTILS_PYTHON_PATH_RESULT
    OUTPUT_VARIABLE CCUTILS_PYTHON_INSTALL_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(CCUTILS_PYTHON_PATH_RESULT EQUAL 0 AND NOT CCUTILS_PYTHON_INSTALL_DIR STREQUAL "")
    install(FILES parser/ccutils_parser.py DESTINATION "${CCUTILS_PYTHON_INSTALL_DIR}")
  else()
    message(FATAL_ERROR "Could not determine Python site-packages install path")
  endif()
endif()
