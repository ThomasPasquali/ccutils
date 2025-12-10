#ifndef __CCUTILS_MPI_MACROS__
#define __CCUTILS_MPI_MACROS__
#ifndef CCUTILS_ENABLE_MPI
#error "ccutils MPI headers require -DCCUTILS_ENABLE_MPI"
#endif

#include "../formats.h"
#include "../macros.h"

#define MPI_ONCE(statement)                             \
    do {                                                \
        int inmacro_myid;                               \
        MPI_Comm_rank(MPI_COMM_WORLD, &inmacro_myid);   \
        if (inmacro_myid == 0) {                        \
            statement;                                  \
        }                                               \
    } while(0);

#define MPI_PRINT_ONCE(print_statement)                 \
    do {                                                \
        int inmacro_myid;                               \
        MPI_Comm_rank(MPI_COMM_WORLD, &inmacro_myid);   \
        if (inmacro_myid == 0) {                        \
            print_statement;                            \
        }                                               \
        FLUSH_WAIT(200000)                              \
    } while(0);

#define MPI_PRINTF_ONCE(fmt, ...)                   \
    MPI_PRINT_ONCE(printf(fmt, ##__VA_ARGS__))      

// @param print_name The name that will be printed the prefix and suffix of the print block
// @param PRINTS_CODE_BLOCK A user-defined code block that prints per-rank information.
//
// IMPORTANT: for your prints use `fprintf(fp, ...);`. This will ensure the print is redirected to the correct file descriptor.
#define MPI_ALL_PRINT_NAMED(print_name, PRINTS_CODE_BLOCK) {                      \
    MPI_PRINT_ONCE(printf(CCUTILS_FMT_MPI_PRINT_ALL_NAMED_START, #print_name))    \
    MPI_ALL_PRINT(PRINTS_CODE_BLOCK)                                              \
    MPI_PRINT_ONCE(printf(CCUTILS_FMT_MPI_PRINT_ALL_NAMED_END,   #print_name))    \
    FLUSH_WAIT(200000)                                                            \
  }

// @param PRINTS_CODE_BLOCK A user-defined code block that prints per-rank information.
//
// IMPORTANT: for your prints use `fprintf(fp, ...);`. This will ensure the print is redirected to the correct file descriptor.
#define MPI_ALL_PRINT(PRINTS_CODE_BLOCK) {                                           \
    int inmacro_myid, inmacro_ntask;                                                 \
    MPI_Comm_rank(MPI_COMM_WORLD, &inmacro_myid);                                    \
	MPI_Comm_size(MPI_COMM_WORLD, &inmacro_ntask);                                   \
    FILE *fp;                                                                        \
    char s[50], s1[50];                                                              \
    sprintf(s, "ccutils_temp_%d.txt", inmacro_myid);                                 \
    fp = fopen (s, "w");                                                             \
    fclose(fp);                                                                      \
    fp = fopen (s, "a+");                                                            \
    fprintf(fp, CCUTILS_FMT_MPI_PRINT_ALL_START, inmacro_myid);                      \
    PRINTS_CODE_BLOCK;                                                               \
    fprintf(fp, CCUTILS_FMT_MPI_PRINT_ALL_END, inmacro_myid);                        \
    fclose(fp);                                                                      \
    for (int i=0; i<inmacro_ntask; i++) {                                            \
        if (inmacro_myid == i) {                                                     \
            int error;                                                               \
            sprintf(s1, "cat ccutils_temp_%d.txt", inmacro_myid);                    \
            error = system(s1);                                                      \
            if (error != 0) fprintf(stderr, CCUTILS_FMT_ERROR,                       \
                __LINE__, __FILE__, "MPI_ALL_PRINT: could not cat tmp file.");       \
            sprintf(s1, "rm ccutils_temp_%d.txt", inmacro_myid);                     \
            error = system(s1);                                                      \
            if (error != 0) fprintf(stderr, CCUTILS_FMT_ERROR,                       \
                __LINE__, __FILE__, "MPI_ALL_PRINT: could not rm tmp file.");        \
        }                                                                            \
        MPI_Barrier(MPI_COMM_WORLD);                                                 \
    }                                                                                \
  }

#define MPI_COMMUNICATOR_PRINT(CM, X)  \
  {\
    int global_rank; \
    int inmacro_myid, inmacro_ntask;  \
    MPI_Comm_rank(CM, &inmacro_myid);  \
	MPI_Comm_size(CM, &inmacro_ntask);  \
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);  \
    char name[MPI_MAX_OBJECT_NAME]; \
    int name_length; \
    MPI_Comm_get_name(CM, name, &name_length); \
    FILE *fp;\
    char s[50], s1[50];\
    sprintf(s, "temp_%s_%d_%d.txt", name, inmacro_myid, global_rank);\
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

#define MPI_PROCESS_PRINT(CM, P, X)  \
  {\
    int myid, ntask;  \
    MPI_Comm_rank(CM, &myid);  \
	MPI_Comm_size(CM, &ntask);  \
	if (myid == P) {  \
      fprintf(stdout, "\t--------------------- Proc %d of %d. File %s Line %d ---------------------\n\n", myid, ntask, __FILE__, __LINE__);\
      X;\
      fprintf(stdout, "\t--------------------------------------------------------------------------\n\n");\
    }  \
  }

// Flushes stdout and sleeps for `useconds` microseconds. 1000000 == 1 second
#define FLUSH_WAIT(useconds) \
  do {                       \
      fflush(stdout);        \
      fflush(stderr);        \
      usleep(useconds);      \
  } while(0);


#ifndef CCUTILS_NO_JSON
  // Declare a global JSON only on rank 0
  #define DECLARE_GLOBAL_JSON(name)                             \
      int inmacro_myid;                                         \
      MPI_Comm_rank(MPI_COMM_WORLD, &inmacro_myid);             \
      nlohmann::json __section_json_##name;                     \
      nlohmann::json* __section_json_global_##name = nullptr;   \
      if (inmacro_myid == 0) {                                  \
          __section_json_global_##name = new nlohmann::json();  \
      }

    #define MPI_GLOBAL_JSON_PUT(name, key, value)               \
        if (inmacro_myid == 0) {                                \
            (*__section_json_global_##name)[key] = value;       \
        }

    #define MPI_LOCAL_JSON_PUT(name, key, value) \
        __section_json_##name[key] = value; 
#else
    #define DECLARE_GLOBAL_JSON(name)
    #define DECLARE_LOCAL_JSON(name)
#endif


#define MPI_SECTION_DEF(name, title)           \
    DECLARE_GLOBAL_JSON(name)                  \
    MPI_PRINT_ONCE(SECTION_DEF(name, title));

#ifndef CCUTILS_NO_JSON
#define MPI_SECTION_END(name)                                                                                   \
    do {                                                                                                        \
        int inmacro_myid;                                                                                       \
        MPI_Comm_rank(MPI_COMM_WORLD, &inmacro_myid);                                                           \
        if (!__section_json_##name.empty())                                                                     \
            MPI_ALL_PRINT_NAMED(ccutils_rank_json, fprintf(fp, "%s\n", __section_json_##name.dump().c_str()))   \
        /* Only rank 0 prints SECTION_END */                                                                    \
        if (inmacro_myid == 0) __section_json_##name.clear();                                                   \
        if (__section_json_global_##name && !__section_json_global_##name->empty()) {                           \
            printf(CCUTILS_FMT_GLOBAL_JSON_START, "ccutils_global_json");                                       \
            printf("%s\n", __section_json_global_##name->dump().c_str());                                       \
            printf(CCUTILS_FMT_GLOBAL_JSON_END, "ccutils_global_json");                                         \
            fflush(stdout);                                                                                     \
            delete __section_json_global_##name;                                                                \
            __section_json_global_##name = nullptr;                                                             \
        }                                                                                                       \
        MPI_PRINT_ONCE(SECTION_END(name));                                                                      \
    } while(0);
#else
#define MPI_SECTION_END(name)                                               \
    do {                                                                    \
        int inmacro_myid;                                                   \
        MPI_Comm_rank(MPI_COMM_WORLD, &inmacro_myid);                       \
        /* Only rank 0 prints SECTION_END */                                \
        MPI_PRINT_ONCE(SECTION_END(name));                                  \
    } while(0);
#endif

// MPI + CUDA
#ifdef CCUTILS_ENABLE_CUDA
    #define CUDA_PRINT_DEVICE {                                             \
        int inmacro_myid;                                                   \
        MPI_Comm_rank(MPI_COMM_WORLD, &inmacro_myid);                       \
        int dev;                                                            \
        cudaError_t err = cudaGetDevice(&dev);                              \
        if (err == cudaSuccess) {                                           \
            printf("[[Rank %d]] CUDA device: %d\n", inmacro_myid, dev);     \
        } else {                                                            \
            printf("cudaGetDevice failed: %s\n", cudaGetErrorString(err));  \
        }                                                                   \
    }
#endif


// Misc
#define MPI_STATUS_CHECK(NREQ, STATV, COMM) \
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
