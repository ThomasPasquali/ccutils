#ifndef __CCUTILS_MPI_MACROS__
#define __CCUTILS_MPI_MACROS__

#ifndef CCUTILS_ENABLE_MPI
#error "ccutils MPI headers require -DCCUTILS_ENABLE_MPI"
#endif

#include <unistd.h>
#include <limits.h>
#include "../formats.h"
#include "../macros.hpp"

/**********************************************************************/
/*                           INIT                                     */
/**********************************************************************/

#define CCUTILS_MPI_INIT                                  \
    int ccutils_inmacro_myid;                             \
    int ccutils_inmacro_ntask;                            \
    MPI_Comm_rank(MPI_COMM_WORLD, &ccutils_inmacro_myid); \
    MPI_Comm_size(MPI_COMM_WORLD, &ccutils_inmacro_ntask);

/**********************************************************************/
/*                           FLUSH UTILITY                            */
/**********************************************************************/

#define CCUTILS_FLUSH_WAIT(useconds) \
    fflush(stdout);                  \
    fflush(stderr);                  \
    usleep(useconds);

/**********************************************************************/
/*                        MPI ONCE MACROS                              */
/**********************************************************************/

#define CCUTILS_MPI_ONCE(statement) \
    if (ccutils_inmacro_myid == 0)  \
    {                               \
        statement;                  \
    }

#define CCUTILS_MPI_PRINT_ONCE(print_statement) \
    if (ccutils_inmacro_myid == 0)              \
    {                                           \
        print_statement;                        \
    }                                           \
    CCUTILS_FLUSH_WAIT(200000)

#define CCUTILS_MPI_PRINTF_ONCE(fmt, ...) \
    CCUTILS_MPI_PRINT_ONCE(printf(fmt, ##__VA_ARGS__))

/**********************************************************************/
/*                        MPI ALL PRINT MACROS                        */
/**********************************************************************/

#define CCUTILS_MPI_ALL_PRINT(PRINTS_CODE_BLOCK)                                      \
do {                                                                                  \
    int _ccutils_myid, _ccutils_numprocs;                                             \
    MPI_Comm_rank(MPI_COMM_WORLD, &_ccutils_myid);                                    \
    MPI_Comm_size(MPI_COMM_WORLD, &_ccutils_numprocs);                                \
                                                                                      \
    /* STEP 1: Each rank writes output to local memory */                             \
    char  *_ccutils_buf = NULL;                                                       \
    size_t _ccutils_bsz = 0;                                                          \
    FILE  *fp = open_memstream(&_ccutils_buf, &_ccutils_bsz);                         \
    if (fp != NULL) {                                                                 \
        fprintf(fp, CCUTILS_FMT_MPI_PRINT_ALL_START, _ccutils_myid);                  \
        PRINTS_CODE_BLOCK;                                                            \
        fprintf(fp, CCUTILS_FMT_MPI_PRINT_ALL_END, _ccutils_myid);                    \
        fclose(fp);                                                                   \
    } else {                                                                          \
        fprintf(stderr, "[Rank %d] WARNING: open_memstream failed, "                  \
                        "output will be missing.\n", _ccutils_myid);                  \
        _ccutils_buf = (char *)calloc(1, 1);                                          \
        _ccutils_bsz = 0;                                                             \
    }                                                                                 \
                                                                                      \
    /* Guard: abort if any rank's output exceeds INT_MAX */                           \
    if (_ccutils_bsz > (size_t)INT_MAX) {                                             \
        fprintf(stderr, "[Rank %d] FATAL: per-rank output exceeds INT_MAX "           \
                        "(%zu bytes), aborting.\n", _ccutils_myid, _ccutils_bsz);     \
        MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
    }                                                                                 \
    int _ccutils_local_count = (int)_ccutils_bsz;                                     \
                                                                                      \
    int *_ccutils_recvcounts = NULL;                                                  \
    int *_ccutils_displs     = NULL;                                                  \
    char *_ccutils_recvbuf   = NULL;                                                  \
                                                                                      \
    if (_ccutils_myid == 0) {                                                         \
        _ccutils_recvcounts = (int *)malloc(_ccutils_numprocs * sizeof(int));         \
        _ccutils_displs     = (int *)malloc(_ccutils_numprocs * sizeof(int));         \
        if (!_ccutils_recvcounts || !_ccutils_displs) {                               \
            fprintf(stderr, "[Rank 0] FATAL: metadata malloc failed.\n");             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    }                                                                                 \
                                                                                      \
    /* STEP 2: Gather sizes */                                                        \
    MPI_Gather(&_ccutils_local_count, 1, MPI_INT,                                     \
               _ccutils_recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);                   \
                                                                                      \
    /* Rank 0: compute displacements and guard against total overflow */              \
    if (_ccutils_myid == 0) {                                                         \
        long long _ccutils_total = 0;                                                 \
        for (int i = 0; i < _ccutils_numprocs; i++) {                                 \
            _ccutils_displs[i] = (int)_ccutils_total;                                 \
            _ccutils_total += _ccutils_recvcounts[i];                                 \
            if (_ccutils_total > (long long)INT_MAX) {                                \
                fprintf(stderr, "[Rank 0] FATAL: combined output exceeds INT_MAX "    \
                                "after rank %d, aborting.\n", i);                     \
                MPI_Abort(MPI_COMM_WORLD, 1);                                         \
            }                                                                         \
        }                                                                             \
        int _ccutils_total_chars = (int)_ccutils_total;                               \
        _ccutils_recvbuf = (char *)malloc(_ccutils_total_chars + 1);                  \
        if (!_ccutils_recvbuf) {                                                      \
            fprintf(stderr, "[Rank 0] FATAL: combined buffer malloc failed.\n");      \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
        _ccutils_recvbuf[_ccutils_total_chars] = '\0';                                \
    }                                                                                 \
                                                                                      \
    /* STEP 3: Gather actual strings */                                               \
    MPI_Gatherv(_ccutils_buf, _ccutils_local_count, MPI_CHAR,                         \
                _ccutils_recvbuf, _ccutils_recvcounts, _ccutils_displs, MPI_CHAR,     \
                0, MPI_COMM_WORLD);                                                   \
                                                                                      \
    /* STEP 4: Rank 0 prints and cleans up */                                         \
    if (_ccutils_myid == 0) {                                                         \
        int _ccutils_total_size = _ccutils_displs[_ccutils_numprocs - 1]              \
                                + _ccutils_recvcounts[_ccutils_numprocs - 1];         \
        fwrite(_ccutils_recvbuf, 1, _ccutils_total_size, stdout);                     \
        fflush(stdout);                                                               \
        free(_ccutils_recvcounts);                                                    \
        free(_ccutils_displs);                                                        \
        free(_ccutils_recvbuf);                                                       \
    }                                                                                 \
                                                                                      \
    free(_ccutils_buf);                                                               \
    MPI_Barrier(MPI_COMM_WORLD);                                                      \
} while (0); 


#define CCUTILS_MPI_ALL_PRINT_NAMED(print_name, PRINTS_CODE_BLOCK) {                         \
    CCUTILS_MPI_PRINT_ONCE(printf(CCUTILS_FMT_MPI_PRINT_ALL_NAMED_START, #print_name))       \
        CCUTILS_MPI_ALL_PRINT(PRINTS_CODE_BLOCK)                                             \
            CCUTILS_MPI_PRINT_ONCE(printf(CCUTILS_FMT_MPI_PRINT_ALL_NAMED_END, #print_name)) \
                CCUTILS_FLUSH_WAIT(200000)}

/**********************************************************************/
/*                       MPI PROCESS PRINT MACROS                         */
/**********************************************************************/

#define CCUTILS_MPI_COMMUNICATOR_PRINT(CM, X)                                                                                           \
    {                                                                                                                                   \
        int inmacro_myid, inmacro_ntask;                                                                                                \
        MPI_Comm_rank(CM, &inmacro_myid);                                                                                               \
        MPI_Comm_size(CM, &inmacro_ntask);                                                                                              \
        char name[MPI_MAX_OBJECT_NAME];                                                                                                 \
        int name_length;                                                                                                                \
        MPI_Comm_get_name(CM, name, &name_length);                                                                                      \
        FILE *fp;                                                                                                                       \
        char s[50], s1[50];                                                                                                             \
        sprintf(s, "temp_%s_%d_%d.txt", name, inmacro_myid, ccutils_inmacro_myid);                                                      \
        fp = fopen(s, "w");                                                                                                             \
        fclose(fp);                                                                                                                     \
        fp = fopen(s, "a+");                                                                                                            \
        fprintf(fp, "------------------------- Proc %d File %s Line %d -------------------------\n", inmacro_myid, __FILE__, __LINE__); \
        X;                                                                                                                              \
        if (inmacro_myid == inmacro_ntask - 1)                                                                                          \
            fprintf(fp, "--------------------------------------------------------------------------\n");                                \
        fclose(fp);                                                                                                                     \
        for (int i = 0; i < inmacro_ntask; i++)                                                                                         \
        {                                                                                                                               \
            if (inmacro_myid == i)                                                                                                      \
            {                                                                                                                           \
                int error;                                                                                                              \
                sprintf(s1, "cat %s", s);                                                                                               \
                error = system(s1);                                                                                                     \
                if (error == -1)                                                                                                        \
                    fprintf(stderr, "Error at line %d of file %s", __LINE__, __FILE__);                                                 \
                sprintf(s1, "rm %s", s);                                                                                                \
                error = system(s1);                                                                                                     \
                if (error == -1)                                                                                                        \
                    fprintf(stderr, "Error at line %d of file %s", __LINE__, __FILE__);                                                 \
            }                                                                                                                           \
            MPI_Barrier(CM);                                                                                                            \
        }                                                                                                                               \
    }

#define CCUTILS_MPI_PROCESS_PRINT(CM, P, X)                                                                                                   \
    {                                                                                                                                         \
        int myid, ntask;                                                                                                                      \
        MPI_Comm_rank(CM, &myid);                                                                                                             \
        MPI_Comm_size(CM, &ntask);                                                                                                            \
        if (myid == P)                                                                                                                        \
        {                                                                                                                                     \
            fprintf(stdout, "--------------------- Proc %d of %d. File %s Line %d ---------------------\n", myid, ntask, __FILE__, __LINE__); \
            X;                                                                                                                                \
            fprintf(stdout, "--------------------------------------------------------------------------\n");                                  \
        }                                                                                                                                     \
    }

#define CCUTILS_MPI_BUFFERED_PRINT_ADD(PRINTS_CODE_BLOCK)                   \
    {                                                                       \
        FILE *fp;                                                           \
        char fname[256];                                                    \
        char job_id[64];                                                    \
        char *slurm_job_id = getenv("SLURM_JOB_ID");                        \
                                                                            \
        if (slurm_job_id == NULL)                                           \
            sprintf(job_id, "%d", getpid());                                \
        else                                                                \
            sprintf(job_id, "%s", slurm_job_id);                            \
                                                                            \
        sprintf(fname, "ccutils_bufprint_%s_rank%d.txt",                    \
                job_id, ccutils_inmacro_myid);                              \
                                                                            \
        fp = fopen(fname, "a"); /* creates file if not existing */          \
        if (fp != NULL)                                                     \
        {                                                                   \
            PRINTS_CODE_BLOCK;                                              \
            fclose(fp);                                                     \
        }                                                                   \
        else                                                                \
        {                                                                   \
            fprintf(stderr, CCUTILS_FMT_ERROR,                              \
                    __LINE__, __FILE__,                                     \
                    "MPI_BUFFERED_PRINT_ADD: could not open buffer file."); \
        }                                                                   \
    }

/**
 * This macro will flush only the calling rank buffer
 */
#define CCUTILS_MPI_BUFFERED_PRINT_FLUSH                                     \
    {                                                                        \
        FILE *fp;                                                            \
        char fname[256];                                                     \
        char job_id[64];                                                     \
        char *slurm_job_id = getenv("SLURM_JOB_ID");                         \
                                                                             \
        if (slurm_job_id == NULL)                                            \
            sprintf(job_id, "%d", getpid());                                 \
        else                                                                 \
            sprintf(job_id, "%s", slurm_job_id);                             \
                                                                             \
        sprintf(fname, "ccutils_bufprint_%s_rank%d.txt",                     \
                job_id, ccutils_inmacro_myid);                               \
                                                                             \
        fp = fopen(fname, "r");                                              \
        if (fp != NULL)                                                      \
        {                                                                    \
            printf(CCUTILS_FMT_MPI_PRINT_BUFFERED_START,                     \
                   ccutils_inmacro_myid);                                    \
                                                                             \
            int c;                                                           \
            while ((c = fgetc(fp)) != EOF)                                   \
                putchar(c);                                                  \
                                                                             \
            fclose(fp);                                                      \
                                                                             \
            if (remove(fname) != 0)                                          \
            {                                                                \
                fprintf(stderr, CCUTILS_FMT_ERROR,                           \
                        __LINE__, __FILE__,                                  \
                        "MPI_BUFFERED_PRINT_FLUSH: could not remove file."); \
            }                                                                \
                                                                             \
            printf(CCUTILS_FMT_MPI_PRINT_BUFFERED_END,                       \
                   ccutils_inmacro_myid);                                    \
        }                                                                    \
    }

/**
 * This macro MUST be called by all ranks
 */
#define CCUTILS_MPI_BUFFERED_PRINT_ALL_FLUSH                                    \
    {                                                                           \
        char fname[256];                                                        \
        char job_id[64];                                                        \
        char *slurm_job_id = getenv("SLURM_JOB_ID");                            \
                                                                                \
        if (slurm_job_id == NULL)                                               \
            sprintf(job_id, "%d", getpid());                                    \
        else                                                                    \
            sprintf(job_id, "%s", slurm_job_id);                                \
                                                                                \
        for (int __ccutils_i = 0; __ccutils_i < ccutils_inmacro_ntask;          \
             ++__ccutils_i)                                                     \
        {                                                                       \
            if (ccutils_inmacro_myid == __ccutils_i)                            \
            {                                                                   \
                sprintf(fname, "ccutils_bufprint_%s_rank%d.txt",                \
                        job_id, ccutils_inmacro_myid);                          \
                                                                                \
                FILE *fp = fopen(fname, "r");                                   \
                if (fp != NULL)                                                 \
                {                                                               \
                    printf(CCUTILS_FMT_MPI_PRINT_BUFFERED_START,                \
                           ccutils_inmacro_myid);                               \
                                                                                \
                    int c;                                                      \
                    while ((c = fgetc(fp)) != EOF)                              \
                        putchar(c);                                             \
                                                                                \
                    fclose(fp);                                                 \
                                                                                \
                    if (remove(fname) != 0)                                     \
                    {                                                           \
                        fprintf(stderr, CCUTILS_FMT_ERROR,                      \
                                __LINE__, __FILE__,                             \
                                "MPI_BUFFERED_PRINT_ALL_FLUSH: could not rm."); \
                    }                                                           \
                                                                                \
                    printf(CCUTILS_FMT_MPI_PRINT_BUFFERED_END,                  \
                           ccutils_inmacro_myid);                               \
                    fflush(stdout);                                             \
                }                                                               \
            }                                                                   \
                                                                                \
            MPI_Barrier(MPI_COMM_WORLD);                                        \
        }                                                                       \
    }

/**********************************************************************/
/*                            MPI JSON MACROS                         */
/**********************************************************************/

#ifndef CCUTILS_NO_JSON
#define CCUTILS_DECLARE_GLOBAL_JSON(name)                    \
    nlohmann::json __section_json_##name;                    \
    nlohmann::json *__section_json_global_##name = nullptr;  \
    if (ccutils_inmacro_myid == 0)                           \
    {                                                        \
        __section_json_global_##name = new nlohmann::json(); \
    }

#define CCUTILS_MPI_GLOBAL_JSON_PUT(name, key, value) \
    if (ccutils_inmacro_myid == 0)                    \
    {                                                 \
        (*__section_json_global_##name)[key] = value; \
    }

#define CCUTILS_MPI_LOCAL_JSON_PUT(name, key, value) \
    __section_json_##name[key] = value;
#else
#define CCUTILS_DECLARE_GLOBAL_JSON(name)
#define CCUTILS_DECLARE_LOCAL_JSON(name)
#endif

/**********************************************************************/
/*                        MPI SECTION MACROS                            */
/**********************************************************************/

#define CCUTILS_MPI_SECTION_DEF(name, title) \
    CCUTILS_DECLARE_GLOBAL_JSON(name)        \
    CCUTILS_MPI_PRINT_ONCE(CCUTILS_SECTION_DEF(name, title));

#ifndef CCUTILS_NO_JSON
#define CCUTILS_MPI_SECTION_END(name)                                                                                 \
    do                                                                                                                \
    {                                                                                                                 \
        if (!__section_json_##name.empty())                                                                           \
            CCUTILS_MPI_ALL_PRINT_NAMED(ccutils_rank_json, fprintf(fp, "%s\n", __section_json_##name.dump().c_str())) \
        if (ccutils_inmacro_myid == 0)                                                                                \
            __section_json_##name.clear();                                                                            \
        if (__section_json_global_##name && !__section_json_global_##name->empty())                                   \
        {                                                                                                             \
            printf(CCUTILS_FMT_GLOBAL_JSON_START, "ccutils_global_json");                                             \
            printf("%s\n", __section_json_global_##name->dump().c_str());                                             \
            printf(CCUTILS_FMT_GLOBAL_JSON_END, "ccutils_global_json");                                               \
            fflush(stdout);                                                                                           \
            delete __section_json_global_##name;                                                                      \
            __section_json_global_##name = nullptr;                                                                   \
        }                                                                                                             \
        CCUTILS_MPI_PRINT_ONCE(CCUTILS_SECTION_END(name));                                                            \
    } while (0);
#else
#define CCUTILS_MPI_SECTION_END(name) \
    CCUTILS_MPI_PRINT_ONCE(CCUTILS_SECTION_END(name));
#endif

/**********************************************************************/
/*                       MPI + CUDA PRINT MACROS                        */
/**********************************************************************/

#ifdef CCUTILS_ENABLE_CUDA
#define CCUTILS_MPI_CUDA_PRINT_DEVICE                                           \
    {                                                                           \
        int dev;                                                                \
        cudaError_t err = cudaGetDevice(&dev);                                  \
        if (err == cudaSuccess)                                                 \
        {                                                                       \
            printf("[[Rank %d]] CUDA device: %d\n", ccutils_inmacro_myid, dev); \
        }                                                                       \
        else                                                                    \
        {                                                                       \
            printf("cudaGetDevice failed: %s\n", cudaGetErrorString(err));      \
        }                                                                       \
    }
#endif

/**********************************************************************/
/*                             MPI STATUS CHECK                       */
/**********************************************************************/

#define CCUTILS_MPI_STATUS_CHECK(NREQ, STATV, COMM)                      \
    for (int i = 0; i < NREQ; i++)                                       \
    {                                                                    \
        if (STATV[i].MPI_ERROR != MPI_SUCCESS)                           \
        {                                                                \
            char errstr[MPI_MAX_ERROR_STRING];                           \
            int len;                                                     \
            MPI_Error_string(STATV[i].MPI_ERROR, errstr, &len);          \
            fprintf(stderr, "MPI error in request %d: %s\n", i, errstr); \
            MPI_Abort(COMM, STATV[i].MPI_ERROR);                         \
        }                                                                \
    }

#endif

/**********************************************************************/
/*                       MPI FINALIZE                                 */
/**********************************************************************/

#define CCUTILS_MPI_CLEANUP                                            \
    {                                                                  \
        char fname[256];                                               \
        char job_id[64];                                               \
        char *slurm_job_id = getenv("SLURM_JOB_ID");                   \
                                                                       \
        if (slurm_job_id == NULL)                                      \
            sprintf(job_id, "%d", getpid());                           \
        else                                                           \
            sprintf(job_id, "%s", slurm_job_id);                       \
                                                                       \
        /* ---- Buffered print files ---- */                           \
        sprintf(fname, "ccutils_bufprint_%s_rank%d.txt",               \
                job_id, ccutils_inmacro_myid);                         \
                                                                       \
        remove(fname); /* safe even if file doesn't exist */           \
                                                                       \
        /* ---- Extend here with additional tmp resources ---- */      \
        /* Example future extension:                                   \
           sprintf(fname, "ccutils_otherprefix_%s_rank%d.tmp", job_id, \
                   ccutils_inmacro_myid);                              \
           remove(fname);                                              \
        */                                                             \
                                                                       \
        MPI_Barrier(MPI_COMM_WORLD);                                   \
    }
