#ifndef CCUTILS_COLORS_H
#define CCUTILS_COLORS_H

/*
 * CCUTILS_COLOR_LEVEL controls the ANSI palette used by formatted output:
 *   0: no color
 *   1: basic (8 foreground colors; bright colors fall back to basic)
 *   2: bright (16 foreground colors)
 *   3: full (16 foreground and 16 background colors)
 *
 * Define this before including CCUTILS headers, or pass it as a compiler
 * definition (for example, -DCCUTILS_COLOR_LEVEL=1).
 */
#ifdef CCUTILS_NO_COLOR
#undef CCUTILS_COLOR_LEVEL
#define CCUTILS_COLOR_LEVEL 0
#elif !defined(CCUTILS_COLOR_LEVEL)
#define CCUTILS_COLOR_LEVEL 2
#endif

#if CCUTILS_COLOR_LEVEL < 0 || CCUTILS_COLOR_LEVEL > 3
#error "CCUTILS_COLOR_LEVEL must be an integer from 0 to 3"
#endif

#if CCUTILS_COLOR_LEVEL == 0
#define RESET ""
#define BLACK ""
#define RED ""
#define GREEN ""
#define YELLOW ""
#define BLUE ""
#define MAGENTA ""
#define CYAN ""
#define WHITE ""
#define BRIGHT_BLACK ""
#define BRIGHT_RED ""
#define BRIGHT_GREEN ""
#define BRIGHT_YELLOW ""
#define BRIGHT_BLUE ""
#define BRIGHT_MAGENTA ""
#define BRIGHT_CYAN ""
#define BRIGHT_WHITE ""
#define BG_BLACK ""
#define BG_RED ""
#define BG_GREEN ""
#define BG_YELLOW ""
#define BG_BLUE ""
#define BG_MAGENTA ""
#define BG_CYAN ""
#define BG_WHITE ""
#define BG_BRIGHT_BLACK ""
#define BG_BRIGHT_RED ""
#define BG_BRIGHT_GREEN ""
#define BG_BRIGHT_YELLOW ""
#define BG_BRIGHT_BLUE ""
#define BG_BRIGHT_MAGENTA ""
#define BG_BRIGHT_CYAN ""
#define BG_BRIGHT_WHITE ""
#elif CCUTILS_COLOR_LEVEL == 1
#define RESET "\033[0m"
#define BLACK "\033[30m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN "\033[36m"
#define WHITE "\033[37m"
#define BRIGHT_BLACK BLACK
#define BRIGHT_RED RED
#define BRIGHT_GREEN GREEN
#define BRIGHT_YELLOW YELLOW
#define BRIGHT_BLUE BLUE
#define BRIGHT_MAGENTA MAGENTA
#define BRIGHT_CYAN CYAN
#define BRIGHT_WHITE WHITE
#define BG_BLACK ""
#define BG_RED ""
#define BG_GREEN ""
#define BG_YELLOW ""
#define BG_BLUE ""
#define BG_MAGENTA ""
#define BG_CYAN ""
#define BG_WHITE ""
#define BG_BRIGHT_BLACK ""
#define BG_BRIGHT_RED ""
#define BG_BRIGHT_GREEN ""
#define BG_BRIGHT_YELLOW ""
#define BG_BRIGHT_BLUE ""
#define BG_BRIGHT_MAGENTA ""
#define BG_BRIGHT_CYAN ""
#define BG_BRIGHT_WHITE ""
#elif CCUTILS_COLOR_LEVEL == 2
#define RESET "\033[0m"
#define BLACK "\033[30m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN "\033[36m"
#define WHITE "\033[37m"
#define BRIGHT_BLACK "\033[90m"
#define BRIGHT_RED "\033[91m"
#define BRIGHT_GREEN "\033[92m"
#define BRIGHT_YELLOW "\033[93m"
#define BRIGHT_BLUE "\033[94m"
#define BRIGHT_MAGENTA "\033[95m"
#define BRIGHT_CYAN "\033[96m"
#define BRIGHT_WHITE "\033[97m"
#define BG_BLACK ""
#define BG_RED ""
#define BG_GREEN ""
#define BG_YELLOW ""
#define BG_BLUE ""
#define BG_MAGENTA ""
#define BG_CYAN ""
#define BG_WHITE ""
#define BG_BRIGHT_BLACK ""
#define BG_BRIGHT_RED ""
#define BG_BRIGHT_GREEN ""
#define BG_BRIGHT_YELLOW ""
#define BG_BRIGHT_BLUE ""
#define BG_BRIGHT_MAGENTA ""
#define BG_BRIGHT_CYAN ""
#define BG_BRIGHT_WHITE ""
#else
#define RESET "\033[0m"
#define BLACK "\033[30m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN "\033[36m"
#define WHITE "\033[37m"
#define BRIGHT_BLACK "\033[90m"
#define BRIGHT_RED "\033[91m"
#define BRIGHT_GREEN "\033[92m"
#define BRIGHT_YELLOW "\033[93m"
#define BRIGHT_BLUE "\033[94m"
#define BRIGHT_MAGENTA "\033[95m"
#define BRIGHT_CYAN "\033[96m"
#define BRIGHT_WHITE "\033[97m"
#define BG_BLACK "\033[40m"
#define BG_RED "\033[41m"
#define BG_GREEN "\033[42m"
#define BG_YELLOW "\033[43m"
#define BG_BLUE "\033[44m"
#define BG_MAGENTA "\033[45m"
#define BG_CYAN "\033[46m"
#define BG_WHITE "\033[47m"
#define BG_BRIGHT_BLACK "\033[100m"
#define BG_BRIGHT_RED "\033[101m"
#define BG_BRIGHT_GREEN "\033[102m"
#define BG_BRIGHT_YELLOW "\033[103m"
#define BG_BRIGHT_BLUE "\033[104m"
#define BG_BRIGHT_MAGENTA "\033[105m"
#define BG_BRIGHT_CYAN "\033[106m"
#define BG_BRIGHT_WHITE "\033[107m"
#endif

#endif
