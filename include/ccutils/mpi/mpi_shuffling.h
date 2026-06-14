#ifndef __CCUTILS_MPI_MPISHUFFLING__
#define __CCUTILS_MPI_MPISHUFFLING__

#ifndef CCUTILS_ENABLE_MPI
#error "ccutils MPI headers require -DCCUTILS_ENABLE_MPI"
#endif

#include <unistd.h>
#include <ccutils/mpi/mpi_mympicomm.h>

// ==============================================================================
//                        Distribuited random suffling
// ==============================================================================

struct MpiShufflingHelper {
private:
  MyMpiComm comm;
  int *myindices;
  int *allindices;
  int indices_per_process;

  void shuffle_indices(void) {
    int *local_v  = (int *)malloc(sizeof(int) * indices_per_process);
    int *global_v = (int *)malloc(sizeof(int) * indices_per_process * comm.size);

    for (int i = 0; i < indices_per_process; i++)
      local_v[i] = rand();
    MPI_Allgather(local_v, indices_per_process, MPI_INT, global_v,
                  indices_per_process, MPI_INT, comm.comm);

    for (int j = 0; j < indices_per_process; j++) {
      myindices[j] = 0;
      int x = local_v[j];
      for (int i = 0; i < comm.size * indices_per_process; i++) {
        if (global_v[i] < x)
          myindices[j]++;
        else if (global_v[i] == x && i < comm.rank * indices_per_process + j)
          myindices[j]++;
      }
    }

    free(local_v);
    free(global_v);
    MPI_Allgather(myindices, indices_per_process, MPI_INT, allindices,
                  indices_per_process, MPI_INT, comm.comm);
  }

public:
  MpiShufflingHelper(MPI_Comm input_comm, int indperproc) {
    comm.init(input_comm);
    indices_per_process = indperproc;

    myindices = (int *)malloc(sizeof(int) * indices_per_process);
    allindices = (int *)malloc(sizeof(int) * comm.size * indices_per_process);

    srand(316837 * (comm.rank + 1));
    for (int i=0; i<10; i++) rand();
    shuffle_indices();
  }

  int get_index(int i) {
    if (i >= indices_per_process) {
      fprintf(stderr,
              "[%d] Error: function %s required and index %d > the "
              "indices_per_process %d!\n",
              comm.rank, __func__, i, indices_per_process);
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }
    return (myindices[i]);
  }

  void print(void) const {
    MPI_Barrier(comm.comm);
    fflush(stdout);
    sleep(1);

    if (comm.rank == 0) {
      fprintf(stdout, "----------------------------------- MpiShufflingHelper "
                      "-----------------------------------\n");
      fprintf(stdout, "Allindices: ");
      for (int i = 0; i < comm.size * indices_per_process; i++)
        fprintf(stdout, "%d ", allindices[i]);
      fprintf(stdout, "\n");

      fflush(stdout);
    }

    MPI_Barrier(comm.comm);
    fflush(stdout);
    sleep(1);

    for (int r = 0; r < comm.size; r++) {
      if (r == comm.rank) {
        fprintf(stdout, "[%d]   myindices: ", comm.rank);
        for (int i = 0; i < indices_per_process; i++)
          fprintf(stdout, "%d ", myindices[i]);
        fprintf(stdout, "\n");
      }
      MPI_Barrier(comm.comm);
    }
  }

  const int *get_shuffling_vector(void) {
    return (static_cast<const int *>(allindices));
  }

  ~MpiShufflingHelper(void) {
    free(myindices);
    free(allindices);
  }
};

#endif
