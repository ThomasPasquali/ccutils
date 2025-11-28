#ifndef __CCUTILS_CPU_MACROS__
#define __CCUTILS_CPU_MACROS__

#ifndef CCUTILS_NO_JSON
  #include <nlohmann/json.hpp>
  #include <iostream>
  #include <string>
#endif
#include "colors.h"

// Math
#define CEILING(x,y) (((x) + (y) - 1) / (y))

// Testing
#define ASSERT(cond, msg, ...) \
  if (!(cond)) { \
    fprintf(stderr, BRIGHT_RED "Assertion in %s on line %i failed\n" RESET, __FILE__, __LINE__);\
    fprintf(stderr, BRIGHT_RED msg RESET, ##__VA_ARGS__);\
    exit(EXIT_FAILURE); \
  }

// Prints
#define PRINT_SPLIT(s) \
  printf("--------------------  %s  --------------------\n", s); \
  fflush(stdout);

#define DEBUG_PRINT(fmt, ...) printf(BRIGHT_CYAN "[DEBUG] " fmt RESET, ##__VA_ARGS__);


//Section
#ifdef CCUTILS_NO_JSON
  #define SECTION_DEF(name, title)                        \
      printf(CCUTILS_FMT_SECTION_START, #name, title);
  #define SECTION_END(name)                               \
      printf(CCUTILS_FMT_SECTION_END, #name);
#else 
  // TODO fix and test
  #define SECTION_DEF(name, title)                                   \
    nlohmann::json __section_json_##name;                            \
    do {                                                             \
      printf("ccutils_section_%s_start\n", #name);                   \
      if (std::string(title) != "") printf("Title: %s\n", title);    \
      fflush(stdout);                                                \
    } while(0)

  #define SECTION_END(name)                                        \
    do {                                                           \
      if(!__section_json_##name.empty()) {                         \
        printf("ccutils_json\n");                                  \
        printf("%s\n", __section_json_##name.dump().c_str());      \
      }                                                            \
      printf("ccutils_section_%s_end\n", #name);                   \
      fflush(stdout);                                              \
    } while(0)

  #define SECTION_JSON_PUT(name, key, value) \
      __section_json_##name[key] = value

  // #define SECTION_DEF_GLOBAL(name)                                       \
  //     struct __section_##name##_t {                                      \
  //         nlohmann::json data;                                           \
  //         __section_##name##_t() {                                       \
  //             printf("ccutils_section_%s_start\n", #name);               \
  //             fflush(stdout);                                            \
  //         }                                                              \
  //         ~__section_##name##_t() {                                      \
  //             if(!data.empty()) {                                        \
  //                 printf("ccutils_json\n");                              \
  //                 printf("%s\n", data.dump().c_str());                   \
  //             }                                                          \
  //             printf("ccutils_section_%s_end\n", #name);                 \
  //             fflush(stdout);                                            \
  //         }                                                              \
  //     } __section_json_##name

#endif

//Section timer
#define SECTION_TIMER_DEF(section_name, timer_name) \
  CPU_TIMER_DEF(section_name##_##timer_name)

#define SECTION_TIMER_START(section_name, timer_name) \
  CPU_TIMER_START(section_name##_##timer_name)

#define SECTION_TIMER_STOP(section_name, timer_name) \
  CPU_TIMER_STOP(section_name##_##timer_name)

//TODO: pass an argument (optional) on how to print the stats 

enum Print {
  PRINT
}; 


#define SECTION_TIMER_PRINT(section_name, timer_name) \
  TIMER_PRINT(section_name##_##timer_name)


#endif
