cmake_minimum_required(VERSION 3.18)
project(cuda_example LANGUAGES CXX CUDA)

# You could alternatively:
# a) Add this path to the env variable CMAKE_MODULE_PATH
# b) Use Cmake FetchContent 
set(ccutils_DIR "../../install/lib/cmake/ccutils")
find_package(ccutils REQUIRED)

add_executable(cuda_example cuda_example.cu)
target_link_libraries(cuda_example PRIVATE ccutils::ccutils ccutils::ccutils_cuda)
