cmake_minimum_required(VERSION 3.18)
project(mpi_examples LANGUAGES CXX)

find_package(MPI REQUIRED)

if(NOT TARGET MPI::MPI_C)
  add_library(MPI::MPI_C INTERFACE IMPORTED)
  set_target_properties(MPI::MPI_C PROPERTIES INTERFACE_LINK_LIBRARIES "${MPI_C_LIBRARIES}")
endif()

set(ccutils_DIR "../../lib/cmake/ccutils")
find_package(ccutils REQUIRED)

add_executable(mpi_example mpi_example.cpp)
target_link_libraries(mpi_example PRIVATE ccutils::ccutils ccutils::ccutils_mpi)

add_executable(mpi_mympicomm_example mpi_mympicomm_example.cpp)
target_link_libraries(mpi_mympicomm_example PRIVATE ccutils::ccutils ccutils::ccutils_mpi)

add_executable(mpi_shuffle_example mpi_shuffle_example.cpp)
target_link_libraries(mpi_shuffle_example PRIVATE ccutils::ccutils ccutils::ccutils_mpi)
