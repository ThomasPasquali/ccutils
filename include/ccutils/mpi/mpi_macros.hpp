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
    int ccutils_macro_myid;                               \
    MPI_Comm_rank(MPI_COMM_WORLD, &ccutils_macro_myid);

/**********************************************************************/
/*                           FLUSH UTILITY                            */
/**********************************************************************/

#define CCUTILS_FLUSH_WAIT(useconds) \
    fflush(stdout);        \
    fflush(stderr);        \
    usleep(useconds);

/**********************************************************************/
/*                        MPI ONCE MACROS                              */
/**********************************************************************/

#define CCUTILS_MPI_ONCE(statement) \
    if (ccutils_macro_myid == 0) {  \
        statement;                  \
    }

#define CCUTILS_MPI_PRINT_ONCE(print_statement) \
    if (ccutils_macro_myid == 0) {              \
        print_statement;                        \
    }                                           \
    CCUTILS_FLUSH_WAIT(200000)

#define CCUTILS_MPI_PRINTF_ONCE(fmt, ...) \
    CCUTILS_MPI_PRINT_ONCE(printf(fmt, ##__VA_ARGS__))

/**********************************************************************/
/*                        MPI ALL PRINT MACROS                        */
/**********************************************************************/

#define CCUTILS_MPI_ALL_PRINT(PRINTS_CODE_BLOCK) {                                   \
    int inmacro_ntask;                                                               \
    MPI_Comm_size(MPI_COMM_WORLD, &inmacro_ntask);                                   \
    FILE *fp;                                                                        \
    char s[100], s1[100];                                                            \
    char job_id[50];                                                                 \
    char *slurm_job_id = getenv("SLURM_JOB_ID");                                     \
    if (slurm_job_id == NULL) sprintf(job_id, "%d", getpid());                       \
    else sprintf(job_id, "%s", slurm_job_id);                                        \
    sprintf(s, "ccutils_temp_%s_%d.txt", job_id, ccutils_macro_myid);                \
    fp = fopen (s, "w");                                                             \
    fclose(fp);                                                                      \
    fp = fopen (s, "a+");                                                            \
    fprintf(fp, CCUTILS_FMT_MPI_PRINT_ALL_START, ccutils_macro_myid);                \
    PRINTS_CODE_BLOCK;                                                               \
    fprintf(fp, CCUTILS_FMT_MPI_PRINT_ALL_END, ccutils_macro_myid);                  \
    fclose(fp);                                                                      \
    for (int i=0; i<inmacro_ntask; i++) {                                            \
        if (ccutils_macro_myid == i) {                                               \
            int error;                                                               \
            sprintf(s1, "cat ccutils_temp_%s_%d.txt", job_id, ccutils_macro_myid);  \
            error = system(s1);                                                      \
            if (error != 0) fprintf(stderr, CCUTILS_FMT_ERROR,                       \
                __LINE__, __FILE__, "MPI_ALL_PRINT: could not cat tmp file.");       \
            sprintf(s1, "rm ccutils_temp_%s_%d.txt", job_id, ccutils_macro_myid);   \
            error = system(s1);                                                      \
            if (error != 0) fprintf(stderr, CCUTILS_FMT_ERROR,                       \
                __LINE__, __FILE__, "MPI_ALL_PRINT: could not rm tmp file.");        \
        }                                                                            \
        MPI_Barrier(MPI_COMM_WORLD);                                                 \
    }                                                                                \
  }

#define CCUTILS_MPI_ALL_PRINT_NAMED(print_name, PRINTS_CODE_BLOCK) {                      \
    CCUTILS_MPI_PRINT_ONCE(printf(CCUTILS_FMT_MPI_PRINT_ALL_NAMED_START, #print_name))    \
    CCUTILS_MPI_ALL_PRINT(PRINTS_CODE_BLOCK)                                              \
    CCUTILS_MPI_PRINT_ONCE(printf(CCUTILS_FMT_MPI_PRINT_ALL_NAMED_END,   #print_name))    \
    CCUTILS_FLUSH_WAIT(200000)                                                            \
  }

/**********************************************************************/
/*                       MPI PROCESS PRINT MACROS                         */
/**********************************************************************/

#define CCUTILS_MPI_COMMUNICATOR_PRINT(CM, X) {             \
    int inmacro_myid, inmacro_ntask;                        \
    MPI_Comm_rank(CM, &inmacro_myid);                       \
    MPI_Comm_size(CM, &inmacro_ntask);                      \
    MPI_Comm_rank(MPI_COMM_WORLD, &ccutils_macro_myid);     \
    char name[MPI_MAX_OBJECT_NAME];                         \
    int name_length;                                        \
    MPI_Comm_get_name(CM, name, &name_length);              \
    FILE *fp;                                               \
    char s[50], s1[50];                                     \
    sprintf(s, "temp_%s_%d_%d.txt", name, inmacro_myid, ccutils_macro_myid); \
    fp = fopen ( s, "w" );\
    fclose(fp);\
    fp = fopen ( s, "a+" );\
    fprintf(fp, "\t------------------------- Proc %d File %s Line %d -------------------------\n\n", inmacro_myid, __FILE__, __LINE__);\
    X;\
    if (inmacro_myid==inmacro_ntask-1) \
        fprintf(fp, "\t--------------------------------------------------------------------------\n\n");\
    fclose(fp);\
    for (int i=0; i<inmacro_ntask; i++) {\
        if (inmacro_myid == i) {\
            int error; \
            sprintf(s1, "cat %s", s);\
            error = system(s1);\
            if (error == -1) fprintf(stderr, "Error at line %d of file %s", __LINE__, __FILE__); \
            sprintf(s1, "rm %s", s);\
            error = system(s1);\
            if (error == -1) fprintf(stderr, "Error at line %d of file %s", __LINE__, __FILE__); \
        }\
        MPI_Barrier(CM);\
    }\
  }

#define CCUTILS_MPI_PROCESS_PRINT(CM, P, X) { \
    int myid, ntask;  \
    MPI_Comm_rank(CM, &myid);  \
    MPI_Comm_size(CM, &ntask);  \
    if (myid == P) {  \
      fprintf(stdout, "\t--------------------- Proc %d of %d. File %s Line %d ---------------------\n\n", myid, ntask, __FILE__, __LINE__);\
      X;\
      fprintf(stdout, "\t--------------------------------------------------------------------------\n\n");\
    }  \
  }

/**********************************************************************/
/*                            MPI JSON MACROS                         */
/**********************************************************************/

#ifndef CCUTILS_NO_JSON
  #define CCUTILS_DECLARE_GLOBAL_JSON(name)                     \
      nlohmann::json __section_json_##name;                     \
      nlohmann::json* __section_json_global_##name = nullptr;   \
      if (ccutils_macro_myid == 0) {                            \
        __section_json_global_##name = new nlohmann::json();    \
      }

  #define CCUTILS_MPI_GLOBAL_JSON_PUT(name, key, value)       \
      if (ccutils_macro_myid == 0) {                          \
        (*__section_json_global_##name)[key] = value;         \
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

#define CCUTILS_MPI_SECTION_DEF(name, title)           \
    CCUTILS_DECLARE_GLOBAL_JSON(name)                  \
    CCUTILS_MPI_PRINT_ONCE(CCUTILS_SECTION_DEF(name, title));

#ifndef CCUTILS_NO_JSON
#define CCUTILS_MPI_SECTION_END(name)                                                                                   \
    do {                                                                                                        \
        int inmacro_myid;                                                                                       \
        MPI_Comm_rank(MPI_COMM_WORLD, &inmacro_myid);                                                           \
        if (!__section_json_##name.empty())                                                                     \
            CCUTILS_MPI_ALL_PRINT_NAMED(ccutils_rank_json, fprintf(fp, "%s\n", __section_json_##name.dump().c_str()))   \
        if (inmacro_myid == 0) __section_json_##name.clear();                                                   \
        if (__section_json_global_##name && !__section_json_global_##name->empty()) {                           \
            printf(CCUTILS_FMT_GLOBAL_JSON_START, "ccutils_global_json");                                       \
            printf("%s\n", __section_json_global_##name->dump().c_str());                                       \
            printf(CCUTILS_FMT_GLOBAL_JSON_END, "ccutils_global_json");                                         \
            fflush(stdout);                                                                                     \
            delete __section_json_global_##name;                                                                \
            __section_json_global_##name = nullptr;                                                             \
        }                                                                                                       \
        CCUTILS_MPI_PRINT_ONCE(CCUTILS_SECTION_END(name));                                                                      \
    } while(0);
#else
#define CCUTILS_MPI_SECTION_END(name)                                               \
    do {                                                                    \
        int inmacro_myid;                                                   \
        MPI_Comm_rank(MPI_COMM_WORLD, &inmacro_myid);                       \
        CCUTILS_MPI_PRINT_ONCE(CCUTILS_SECTION_END(name));                                  \
    } while(0);
#endif

/**********************************************************************/
/*                       MPI + CUDA PRINT MACROS                        */
/**********************************************************************/

#ifdef CCUTILS_ENABLE_CUDA
    #define CCUTILS_MPI_CUDA_PRINT_DEVICE {                                     \
        int dev;                                                                \
        cudaError_t err = cudaGetDevice(&dev);                                  \
        if (err == cudaSuccess) {                                               \
            printf("[[Rank %d]] CUDA device: %d\n", ccutils_macro_myid, dev);   \
        } else {                                                                \
            printf("cudaGetDevice failed: %s\n", cudaGetErrorString(err));      \
        }                                                                       \
    }
#endif

/**********************************************************************/
/*                             MPI STATUS CHECK                           */
/**********************************************************************/

#define CCUTILS_MPI_STATUS_CHECK(NREQ, STATV, COMM) \
    for (int i = 0; i < NREQ; i++) { \
        if (STATV[i].MPI_ERROR != MPI_SUCCESS) { \
            char errstr[MPI_MAX_ERROR_STRING]; \
            int len; \
            MPI_Error_string(STATV[i].MPI_ERROR, errstr, &len); \
            fprintf(stderr, "MPI error in request %d: %s\n", i, errstr); \
            MPI_Abort(COMM, STATV[i].MPI_ERROR); \
        } \
    }

#endif
