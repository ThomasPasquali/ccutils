#ifndef __CCUTILS_MPI_MACROS__
#define __CCUTILS_MPI_MACROS__
#ifndef CCUTILS_ENABLE_MPI
#error "ccutils MPI headers require -DCCUTILS_ENABLE_MPI"
#endif

#define MPI_ALL_PRINT(X) {                                                                                                              \
    int inmacro_myid, inmacro_ntask;                                                                                                    \
    MPI_Comm_rank(MPI_COMM_WORLD, &inmacro_myid);                                                                                       \
	  MPI_Comm_size(MPI_COMM_WORLD, &inmacro_ntask);                                                                                      \
	\
    FILE *fp;                                                                                                                           \
    char s[50], s1[50];                                                                                                                 \
    sprintf(s, "temp_%d.txt", inmacro_myid);                                                                                            \
    fp = fopen ( s, "w" );                                                                                                              \
    fclose(fp);                                                                                                                         \
    fp = fopen ( s, "a+" );                                                                                                             \
    fprintf(fp, "\t------------------------- Proc %d File %s Line %d -------------------------\n\n", inmacro_myid, __FILE__, __LINE__); \
    X;                                                                                                                                  \
    if (inmacro_myid==inmacro_ntask-1)                                                                                                  \
        fprintf(fp, "\t--------------------------------------------------------------------------\n\n");                                \
    fclose(fp);                                                                                                                         \
    \
    for (int i=0; i<inmacro_ntask; i++) {                                                                                               \
        if (inmacro_myid == i) {                                                                                                        \
            int error;                                                                                                                  \
            sprintf(s1, "cat temp_%d.txt", inmacro_myid);                                                                               \
            error = system(s1);                                                                                                         \
            if (error == -1) fprintf(stderr, "Error at line %d of file %s\n", __LINE__, __FILE__);                                      \
            sprintf(s1, "rm temp_%d.txt", inmacro_myid);                                                                                \
            error = system(s1);                                                                                                         \
            if (error == -1) fprintf(stderr, "Error at line %d of file %s\n", __LINE__, __FILE__);                                      \
        }                                                                                                                               \
        MPI_Barrier(MPI_COMM_WORLD);                                                                                                    \
    }                                                                                                                                   \
  }

#define MPI_COMMUNICATOR_PRINT(CM, X)  \
  {\
    int inmacro_myid, inmacro_ntask;  \
    MPI_Comm_rank(CM, &inmacro_myid);  \
	MPI_Comm_size(CM, &inmacro_ntask);  \
    char name[MPI_MAX_OBJECT_NAME]; \
    int name_length; \
    MPI_Comm_get_name(CM, name, &name_length); \
    FILE *fp;\
    char s[50], s1[50];\
    sprintf(s, "temp_%s_%d.txt", name, inmacro_myid);\
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
  do { \
      fflush(stdout); \
      usleep(useconds); \
  } while(0);

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
