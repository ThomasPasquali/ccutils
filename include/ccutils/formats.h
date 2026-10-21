#ifndef CCUTILS_FORMATS_H
#define CCUTILS_FORMATS_H

// These are the pinned LEGACY formats. v2 uses emitter render helpers below; keep these strings unchanged.
#include "colors.h"

/**********************************************************************/
/*                              TIMERS                                 */
/**********************************************************************/

#define CCUTILS_FMT_TIMER_MULTIPLE BG_CYAN BRIGHT_WHITE "CCUTILS_TIMER v=1 type=summary group=%s name=%s n=%zu avg_ms=%.6f stddev_ms=%.6f min_ms=%.6f max_ms=%.6f sum_ms=%.6f" RESET "\n"
#define CCUTILS_FMT_TIMER_ALL BG_CYAN BRIGHT_WHITE "CCUTILS_TIMER v=1 type=run name=%s index=%zu elapsed_ms=%.6f" RESET "\n"
#define CCUTILS_FMT_TIMER_SINGLE BG_CYAN BRIGHT_WHITE "CCUTILS_TIMER v=1 type=sample group=%s name=%s elapsed_ms=%.6f" RESET "\n"
#define CCUTILS_FMT_TIMER_SUM BG_MAGENTA BRIGHT_WHITE "CCUTILS_TIMER v=1 type=aggregate operation=%s name=%s elapsed_ms=%.6f" RESET "\n"

/**********************************************************************/
/*                               SECTIONS                              */
/**********************************************************************/

#define CCUTILS_FMT_SECTION_START BG_BLUE BRIGHT_WHITE "=+=+=+= %s :: %s =+=+=+=" RESET "\n"
#define CCUTILS_FMT_SECTION_END   BG_BLUE BRIGHT_WHITE "=+=+=+= %s END =+=+=+=" RESET "\n"

/**********************************************************************/
/*                             GLOBAL JSON                              */
/**********************************************************************/

#define CCUTILS_FMT_GLOBAL_JSON_START BG_MAGENTA BRIGHT_WHITE "=-=-= %s =-=-=" RESET "\n"
#define CCUTILS_FMT_GLOBAL_JSON_END   BG_MAGENTA BRIGHT_WHITE "=-=-= %s END =-=-=" RESET "\n"

/**********************************************************************/
/*                               MPI PRINTS                             */
/**********************************************************************/

#define CCUTILS_FMT_MPI_PRINT_ALL_NAMED_START BG_BLUE BRIGHT_WHITE "-+-+-+- %s -+-+-+-" RESET "\n"
#define CCUTILS_FMT_MPI_PRINT_ALL_NAMED_END   BG_BLUE BRIGHT_WHITE "-+-+-+- %s END -+-+-+-" RESET "\n"
#define CCUTILS_FMT_MPI_PRINT_ALL_START       BG_YELLOW BLACK "[[Rank %d]]" RESET "\n"
#define CCUTILS_FMT_MPI_PRINT_ALL_END         BG_YELLOW BLACK "[[END Rank %d]]" RESET "\n"
#define CCUTILS_FMT_MPI_PRINT_ALL_DBG         "[[Rank %d, File %s, Line %d]]\n"
#define CCUTILS_FMT_MPI_PRINT_BUFFERED_START BG_BLUE BRIGHT_WHITE "=+=+=+= Rank %d =+=+=+=" RESET "\n"
#define CCUTILS_FMT_MPI_PRINT_BUFFERED_END   BG_BLUE BRIGHT_WHITE "=+=+=+= Rank %d END =+=+=+=" RESET "\n"

/**********************************************************************/
/*                               CUDA PRINTS                            */
/**********************************************************************/
// TODO

/**********************************************************************/
/*                                ERRORS                                */
/**********************************************************************/

#define CCUTILS_FMT_ERROR BRIGHT_RED "CCUTILS. Error at line %d of file '%s':\n\t%s" RESET "\n"

#endif
