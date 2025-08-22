cmake_minimum_required(VERSION 3.18)
project(cpu_timers LANGUAGES CXX)

# You could alternatively:
# a) Add this path to the env variable CMAKE_MODULE_PATH
# b) Use Cmake FetchContent 
set(ccutils_DIR "../../install/lib/cmake/ccutils")
find_package(ccutils REQUIRED)

add_executable(cpu_timers cpu_timers.cpp)
target_link_libraries(cpu_timers PRIVATE ccutils::ccutils)
