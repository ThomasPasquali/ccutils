#ifndef __CCUTILS_MPI_MYMPICOMM__
#define __CCUTILS_MPI_MYMPICOMM__

#ifndef CCUTILS_ENABLE_MPI
#error "ccutils MPI headers require -DCCUTILS_ENABLE_MPI"
#endif

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// NOTE TODO TEMPORARY JUST TO MAKE IT WORKS TODO NOTE
struct MyMpiComm {
    MPI_Comm comm;
    int rank, size;

    void init(MPI_Comm in_comm) {

        if (in_comm == MPI_COMM_NULL) {
            comm = MPI_COMM_NULL;
            size = -1;
            rank = -1;
            return;
        }

        comm = in_comm;
        MPI_Comm_size(in_comm, &size);
        MPI_Comm_rank(in_comm, &rank);
    }

    void print_info(const char *label, FILE *fp) {
        char name[MPI_MAX_OBJECT_NAME];
        int name_len = 0;

        MPI_Comm_get_name(comm, name, &name_len);

        if (name_len == 0) {
            snprintf(name, MPI_MAX_OBJECT_NAME, "<unnamed>");
        }

        int hostname_len = 0;
        char host_name[MPI_MAX_PROCESSOR_NAME];
        MPI_Get_processor_name(host_name, &hostname_len);

        if (hostname_len == 0) {
            snprintf(host_name, MPI_MAX_PROCESSOR_NAME, "<unnamed>");
        }

        int wrank, wsize;
        MPI_Comm_size(MPI_COMM_WORLD, &wsize);
        MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
        fprintf(fp, "[%-12s] hostname=%-20s commname=%-20s rank=%4d size=%4d [wrank=%4d wsize=%4d]\n",
                label, host_name, name, rank, size, wrank, wsize);
    }
};

#endif
