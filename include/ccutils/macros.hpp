#ifndef __CCUTILS_CPU_MACROS__
#define __CCUTILS_CPU_MACROS__

#ifndef CCUTILS_NO_JSON
  #include <nlohmann/json.hpp>
  #include <iostream>
  #include <string>
#endif
#include "colors.h"
#include "formats.h"

/**********************************************************************/
/*                              MATH                                  */
/**********************************************************************/

#define CCUTILS_CEILING(x,y) (((x) + (y) - 1) / (y))


/**********************************************************************/
/*                          CORRECTNESS                               */
/**********************************************************************/

#define CCUTILS_ASSERT(cond, msg, ...) \
  if (!(cond)) { \
    fprintf(stderr, BRIGHT_RED "Assertion in %s on line %i failed\n" RESET, __FILE__, __LINE__);\
    fprintf(stderr, BRIGHT_RED msg RESET, ##__VA_ARGS__);\
    exit(EXIT_FAILURE); \
  }

/**********************************************************************/
/*                               PRINTS                                */
/**********************************************************************/

#define CCUTILS_PRINT_SPLIT(s) \
  printf("--------------------  %s  --------------------\n", s); \
  fflush(stdout);

#define CCUTILS_DEBUG_PRINT(fmt, ...) printf(BRIGHT_CYAN "[DEBUG] " fmt RESET, ##__VA_ARGS__);


/**********************************************************************/
/*                             SECTION                                 */
/**********************************************************************/

#ifdef CCUTILS_NO_JSON
  #define CCUTILS_SECTION_DEF(name, title)                        \
      printf(CCUTILS_FMT_SECTION_START, #name, title);
  #define CCUTILS_SECTION_END(name)                               \
      printf(CCUTILS_FMT_SECTION_END, #name);
#else 
  // TODO fix and test
  #define CCUTILS_SECTION_DEF(name, title)                                    \
    nlohmann::json __section_json_##name;                             \
    do {                                                              \
      printf(CCUTILS_FMT_SECTION_START, #name, title);                \
      fflush(stdout);                                                 \
    } while(0);

  #define CCUTILS_SECTION_END(name)                                         \
    do {                                                            \
      if(!__section_json_##name.empty()) {                          \
        printf(CCUTILS_FMT_GLOBAL_JSON_START, "ccutils_json");      \
        printf("%s\n", __section_json_##name.dump().c_str());       \
        printf(CCUTILS_FMT_GLOBAL_JSON_END, "ccutils_json");        \
      }                                                             \
      printf(CCUTILS_FMT_SECTION_END, #name);                       \
      fflush(stdout);                                               \
    } while(0);

  #define CCUTILS_SECTION_JSON_PUT(name, key, value) \
      __section_json_##name[key] = value;

  #define CCUTILS_SECTION_JSON_SUB_PUT(name, key1, key2, value) \
    __section_json_##name[key1][key2] = (value);

#endif


/**********************************************************************/
/*                           SECTION TIMERS                            */
/**********************************************************************/

#define CCUTILS_SECTION_TIMER_DEF(section_name, timer_name) \
  CCUTILS_CPU_TIMER_DEF(section_name##_##timer_name)

#define CCUTILS_SECTION_TIMER_START(section_name, timer_name) \
  CCUTILS_CPU_TIMER_START(section_name##_##timer_name)

#define CCUTILS_SECTION_TIMER_STOP(section_name, timer_name) \
  CCUTILS_CPU_TIMER_STOP(section_name##_##timer_name)

#define CCUTILS_SECTION_TIMER_PRINT(section_name, timer_name) \
  CCUTILS_TIMER_PRINT(section_name##_##timer_name)

#endif
