#ifndef __CCUTILS_MPI_MACROS__
#define __CCUTILS_MPI_MACROS__

#ifndef CCUTILS_ENABLE_MPI
#error "ccutils MPI headers require -DCCUTILS_ENABLE_MPI"
#endif

#include <unistd.h>
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

#define CCUTILS_MPI_ALL_PRINT(PRINTS_CODE_BLOCK)                                                \
    do {                                                                                        \
        FILE *fp;                                                                               \
        char s[256];                                                                            \
        char job_id[128] = "local";                                                             \
                                                                                                \
        /* 1. Get Job ID for uniqueness across multiple jobs */                                 \
        char *slurm_id = getenv("SLURM_JOB_ID");                                                \
        if (slurm_id != NULL) snprintf(job_id, sizeof(job_id), "%s", slurm_id);                 \
                                                                                                \
        /* 2. Create the filename using YOUR PID logic */                                       \
        snprintf(s, sizeof(s), "ccutils_tmp_%s_%d_rank%d.txt",                                  \
                 job_id, (int)getpid(), ccutils_inmacro_myid);                                  \
                                                                                                \
        /* 3. Everyone writes their own code block to their own file */                         \
        fp = fopen(s, "w");                                                                     \
        if (fp != NULL) {                                                                       \
            fprintf(fp, CCUTILS_FMT_MPI_PRINT_ALL_START, ccutils_inmacro_myid);                 \
            PRINTS_CODE_BLOCK;                                                                  \
            fprintf(fp, CCUTILS_FMT_MPI_PRINT_ALL_END, ccutils_inmacro_myid);                   \
            fclose(fp);                                                                         \
        }                                                                                       \
                                                                                                \
        /* 4. Barrier: Wait for everyone to finish writing to the disk */                       \
        MPI_Barrier(MPI_COMM_WORLD);                                                            \
                                                                                                \
        /* 5. YOUR ROUND-ROBIN LOGIC (Rank by Rank) */                                          \
        for (int i = 0; i < ccutils_inmacro_ntask; i++) {                                       \
            if (ccutils_inmacro_myid == i) {                                                    \
                FILE *tfp = fopen(s, "r");                                                      \
                if (tfp != NULL) {                                                              \
                    char buffer[4096];                                                          \
                    size_t bytes;                                                               \
                    while ((bytes = fread(buffer, 1, sizeof(buffer), tfp)) > 0) {               \
                        fwrite(buffer, 1, bytes, stdout);                                       \
                    }                                                                           \
                    fclose(tfp);                                                                \
                    /* CRITICAL: Tell the OS to push this text out NOW */                       \
                    fflush(stdout);                                                             \
                }                                                                               \
                remove(s); /* Rank cleans up its own file */                                    \
            }                                                                                   \
            /* No one moves to the next rank until Rank 'i' is done flushing */                 \
            MPI_Barrier(MPI_COMM_WORLD);                                                        \
        }                                                                                       \
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
