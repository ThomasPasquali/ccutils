cmake_minimum_required(VERSION 3.18)
project(mpi_example LANGUAGES CXX)

find_package(MPI REQUIRED)

if(NOT TARGET MPI::MPI_C)
  add_library(MPI::MPI_C INTERFACE IMPORTED)
  set_target_properties(MPI::MPI_C PROPERTIES INTERFACE_LINK_LIBRARIES "${MPI_C_LIBRARIES}")
endif()

# You could alternatively:
# a) Add this path to the env variable CMAKE_MODULE_PATH
# b) Use Cmake FetchContent 
set(ccutils_DIR "../../lib/cmake/ccutils")
find_package(ccutils REQUIRED)

add_executable(mpi_example mpi_example.cpp)
target_link_libraries(mpi_example PRIVATE ccutils::ccutils ccutils::ccutils_mpi)