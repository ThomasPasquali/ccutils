#ifndef __CCUTILS_MPI_TIMERS__
#define __CCUTILS_MPI_TIMERS__

#ifndef CCUTILS_ENABLE_MPI
#error "ccutils MPI headers require -DCCUTILS_ENABLE_MPI"
#endif

#include <ccutils/timers.h>
#include <mpi.h>
#include <vector>
#include <iostream>
#include <string>

/**********************************************************************/
/*                           MPI TIMER DEF                              */
/**********************************************************************/

#define CCUTILS_MPI_TIMER_DEF(name) \
    double __timer_start_##name = 0.0; \
    double __timer_stop_##name = 0.0; \
    std::vector<float> __timer_vals_##name;

/**********************************************************************/
/*                          MPI TIMER OPERATIONS                        */
/**********************************************************************/

#define CCUTILS_MPI_TIMER_START(name) \
    do { \
        __timer_start_##name = MPI_Wtime(); \
    } while(0);

#define CCUTILS_MPI_TIMER_STOP(name) \
    do { \
        __timer_stop_##name = MPI_Wtime(); \
        __timer_vals_##name.push_back(__timer_stop_##name - __timer_start_##name); \
    } while(0);

#define CCUTILS_MPI_TIMER_DESTROY(name) \
    __timer_vals_##name.clear();

/**********************************************************************/
/*                          MPI TIMER SHORTCUTS                         */
/**********************************************************************/

#define CCUTILS_MPI_TIMER_INIT(name) \
    CCUTILS_MPI_TIMER_DEF(name) CCUTILS_MPI_TIMER_START(name)

#define CCUTILS_MPI_TIMER_CLOSE(name) {                         \
    int inmacro_myid;                                           \
    MPI_Comm_rank(MPI_COMM_WORLD, &inmacro_myid);               \
    if (inmacro_myid == 0){                                     \
        TIMER_PRINT(name)                                       \
    }                                                           \
    CCUTILS_MPI_TIMER_STOP(name);                               \
    CCUTILS_MPI_TIMER_DESTROY(name)                             \
}

#endif
