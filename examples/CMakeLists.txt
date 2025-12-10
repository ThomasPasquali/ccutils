cmake_minimum_required(VERSION 3.18)
project(ccutils_examples LANGUAGES CXX)

# FIXME
# find_package(MPI REQUIRED)

# This assumes ccutils has been installed on the system (check the README)
find_package(ccutils REQUIRED)

add_executable(prints prints.cpp)
target_link_libraries(prints PRIVATE ccutils::ccutils)