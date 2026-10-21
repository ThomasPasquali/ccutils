# Golden-output tests. The runner compiles the test programs itself (it needs
# several configurations of the same source), so CMake only registers it.
find_package(Python3 3.8 COMPONENTS Interpreter REQUIRED)

add_test(NAME ccutils_golden
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.py)

# Python parser (pure Python, a fraction of a second).
add_test(NAME ccutils_parser
  COMMAND ${Python3_EXECUTABLE} -m unittest tests/test_parser.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
