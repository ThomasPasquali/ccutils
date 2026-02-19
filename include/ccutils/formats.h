#ifndef __CCUTILS_FORMATS__
#define __CCUTILS_FORMATS__

#include "colors.h"

/**********************************************************************/
/*                              TIMERS                                 */
/**********************************************************************/

#define CCUTILS_FMT_TIMER_MULTIPLE  BRIGHT_CYAN     "<%s>[%s] n=%zu,avg=%f,stddev=%f,min=%f,max=%f,sum=%f" RESET "\n"
#define CCUTILS_FMT_TIMER_ALL       BRIGHT_CYAN     "<Timer>[%s] All times (ms):\n"
#define CCUTILS_FMT_TIMER_SINGLE    BRIGHT_CYAN     "<%s>[%s] %f ms" RESET "\n"
#define CCUTILS_FMT_TIMER_WARN      BRIGHT_YELLOW   "<%s>[%s] No recorded runs" RESET "\n"
#define CCUTILS_FMT_TIMER_SUM       BRIGHT_MAGENTA  "<%s>[%s] %f ms" RESET "\n"

/**********************************************************************/
/*                               SECTIONS                              */
/**********************************************************************/

#define CCUTILS_FMT_SECTION_START "=+=+=+= %s :: %s =+=+=+=\n"
#define CCUTILS_FMT_SECTION_END   "=+=+=+= %s END =+=+=+=\n"

/**********************************************************************/
/*                             GLOBAL JSON                              */
/**********************************************************************/

#define CCUTILS_FMT_GLOBAL_JSON_START "=-=-= %s =-=-=\n"
#define CCUTILS_FMT_GLOBAL_JSON_END   "=-=-= %s END =-=-=\n"

/**********************************************************************/
/*                               MPI PRINTS                             */
/**********************************************************************/

#define CCUTILS_FMT_MPI_PRINT_ALL_NAMED_START "-+-+-+- %s -+-+-+-\n"
#define CCUTILS_FMT_MPI_PRINT_ALL_NAMED_END   "-+-+-+- %s END -+-+-+-\n"
#define CCUTILS_FMT_MPI_PRINT_ALL_START       "[[Rank %d]]\n"
#define CCUTILS_FMT_MPI_PRINT_ALL_END         "[[END Rank %d]]\n"
#define CCUTILS_FMT_MPI_PRINT_ALL_DBG         "[[Rank %d, File %s, Line %d]]\n"
#define CCUTILS_FMT_MPI_PRINT_BUFFERED_START "=+=+=+= Rank %d =+=+=+=\n"
#define CCUTILS_FMT_MPI_PRINT_BUFFERED_END   "=+=+=+= Rank %d END =+=+=+=\n"

/**********************************************************************/
/*                               CUDA PRINTS                            */
/**********************************************************************/
// TODO

/**********************************************************************/
/*                                ERRORS                                */
/**********************************************************************/

#define CCUTILS_FMT_ERROR BRIGHT_RED "CCUTILS. Error at line %d of file '%s':\n\t%s" RESET "\n"

#endif
