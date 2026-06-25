#ifndef __CCUTILS_MPI_MPIVCOLLAUXBUFF__
#define __CCUTILS_MPI_MPIVCOLLAUXBUFF__

#ifndef CCUTILS_ENABLE_MPI
#error "ccutils MPI headers require -DCCUTILS_ENABLE_MPI"
#endif

#include <unistd.h>
#include <ccutils/mpi/mpi_mympicomm.h>

typedef enum {
    CCUTILS_ALLREDUCE,
    CCUTILS_ALL2ALL,
    CCUTILS_ALL2ALLV,
    CCUTILS_ALLGATHER,
    CCUTILS_ALLGATHERV,
    CCUTILS_SCATTER,  // scatter
    CCUTILS_SCATTERV, // scatter-variable
    CCUTILS_GATHER,
    CCUTILS_GATHERV,
    CCUTILS_PEER2PEER,
    CCUTILS_PINGPONG,
    CCUTILS_UNDEF
} CcutilsCommunicatioType;

struct CcutilsVcollectiveAuxiliaryBuffers {

private:
  CcutilsMpiComm mycomm;
  CcutilsCommunicatioType commtype;

  void compute_displacement_buffs(int* input, int* output) {
    output[0] = 0;
    for (int i = 1; i < mycomm.size; i++)
      output[i] = output[i - 1] + input[i - 1];
  }

  void print_int_array(FILE *fp, const char *label, int *arr, int n) const {
    if (arr == nullptr) return;
    fprintf(fp, "[rank %d]   %-20s: ", mycomm.rank, label);
    for (int i = 0; i < n; i++) fprintf(fp, "%d ", arr[i]);
    fprintf(fp, "\n");
  }

public:
  int root;
  int *send_count;
  int *recv_count;
  int *send_displacement;
  int *recv_displacement;

  int nelements_send;
  int nelements_recv;

  CcutilsVcollectiveAuxiliaryBuffers(CcutilsMpiComm &comm_input,
                                        CcutilsCommunicatioType commtype_input,
                                        int root_input){
    root     = root_input;
    mycomm   = comm_input;
    commtype = commtype_input;

    send_count        = nullptr;
    recv_count        = nullptr;
    send_displacement = nullptr;
    recv_displacement = nullptr;
    nelements_send    = 0;
    nelements_recv    = 0;
  }

  void alltoallv_init(int* input_sendbuffsize) {
    if (commtype != CCUTILS_ALL2ALLV) {
      fprintf(stderr, "Error: call of %s with commtype %d\n", __func__, commtype);
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }

    send_count        = (int *)malloc(sizeof(int) * mycomm.size);
    recv_count        = (int *)malloc(sizeof(int) * mycomm.size);
    send_displacement = (int *)malloc(sizeof(int) * mycomm.size);
    recv_displacement = (int *)malloc(sizeof(int) * mycomm.size);

    nelements_send = 0;
    for (int i = 0; i < mycomm.size; i++) {
      send_count[i]   = input_sendbuffsize[i];
      nelements_send += input_sendbuffsize[i];
    }

    MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, mycomm.comm);

    nelements_recv = 0;
    for (int i = 0; i < mycomm.size; i++)
      nelements_recv += recv_count[i];

    compute_displacement_buffs(send_count, send_displacement);
    compute_displacement_buffs(recv_count, recv_displacement);
  }

  /* Each rank provides the number of elements it sends.
     recv_count and recv_displacement are filled on all ranks. */
  void allgatherv_init(int input_sendbuffsize) {
    if (commtype != CCUTILS_ALLGATHERV) {
      fprintf(stderr, "Error: call of %s with commtype %d\n", __func__, commtype);
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }

    send_count        = (int *)malloc(sizeof(int) * 1);
    recv_count        = (int *)malloc(sizeof(int) * mycomm.size);
    recv_displacement = (int *)malloc(sizeof(int) * mycomm.size);

    send_count[0]  = input_sendbuffsize;
    nelements_send = input_sendbuffsize;

    MPI_Allgather(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, mycomm.comm);

    nelements_recv = 0;
    for (int i = 0; i < mycomm.size; i++)
      nelements_recv += recv_count[i];

    compute_displacement_buffs(recv_count, recv_displacement);
  }

  /* Each rank provides the number of elements it sends.
     Only root fills recv_count and recv_displacement. */
  void gatherv_init(int input_sendbuffsize) {
    if (commtype != CCUTILS_GATHERV) {
      fprintf(stderr, "Error: call of %s with commtype %d\n", __func__, commtype);
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }

    send_count = (int *)malloc(sizeof(int) * 1);
    if (mycomm.rank == root) {
      recv_count        = (int *)malloc(sizeof(int) * mycomm.size);
      recv_displacement = (int *)malloc(sizeof(int) * mycomm.size);
    }

    send_count[0]  = input_sendbuffsize;
    nelements_send = input_sendbuffsize;

    MPI_Gather(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, root, mycomm.comm);

    if (mycomm.rank == root) {
      nelements_recv = 0;
      for (int i = 0; i < mycomm.size; i++)
        nelements_recv += recv_count[i];
      compute_displacement_buffs(recv_count, recv_displacement);
    } else {
      nelements_recv = 0;
    }
  }

  /* Each rank provides the number of elements it expects to receive.
     Root gathers them all to build send_count and send_displacement. */
  void scatterv_init(int input_recvbuffsize) {
    if (commtype != CCUTILS_SCATTERV) {
      fprintf(stderr, "Error: call of %s with commtype %d\n", __func__, commtype);
      MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }

    recv_count        = (int *)malloc(sizeof(int) * 1);
    recv_count[0]     = input_recvbuffsize;
    nelements_recv    = input_recvbuffsize;

    if (mycomm.rank == root) {
      send_count        = (int *)malloc(sizeof(int) * mycomm.size);
      send_displacement = (int *)malloc(sizeof(int) * mycomm.size);
    }

    MPI_Gather(recv_count, 1, MPI_INT, send_count, 1, MPI_INT, root, mycomm.comm);

    if (mycomm.rank == root) {
      nelements_send = 0;
      for (int i = 0; i < mycomm.size; i++)
        nelements_send += send_count[i];
      compute_displacement_buffs(send_count, send_displacement);
    } else {
      nelements_send = 0;
    }
  }

  void print(FILE *fp = stdout) const {
    static const char *commtype_names[] = {
      "ALLREDUCE", "ALL2ALL", "ALL2ALLV", "ALLGATHER", "ALLGATHERV",
      "SCATTER", "SCATTERV", "GATHER", "GATHERV",
      "PEER2PEER", "PINGPONG", "UNDEF"
    };

    int send_count_n = (commtype == CCUTILS_ALL2ALLV ||
                        commtype == CCUTILS_SCATTERV) ? mycomm.size : 1;
    int recv_count_n = (commtype == CCUTILS_ALL2ALLV  ||
                        commtype == CCUTILS_ALLGATHERV ||
                        commtype == CCUTILS_GATHERV)   ? mycomm.size : 1;

    MPI_Barrier(mycomm.comm);
    if (mycomm.rank == 0) {
      fprintf(fp, "--- CcutilsVcollectiveAuxiliaryBuffers [%s] root=%d size=%d ---\n",
              commtype_names[commtype], root, mycomm.size);
      fflush(fp);
    }
    MPI_Barrier(mycomm.comm);

    for (int r = 0; r < mycomm.size; r++) {
      if (r == mycomm.rank) {
        fprintf(fp, "[rank %d] nelements_send=%-6d nelements_recv=%-6d\n",
                mycomm.rank, nelements_send, nelements_recv);
        print_int_array(fp, "send_count",        send_count,        send_count_n);
        print_int_array(fp, "send_displacement", send_displacement, mycomm.size);
        print_int_array(fp, "recv_count",        recv_count,        recv_count_n);
        print_int_array(fp, "recv_displacement", recv_displacement, mycomm.size);
        fflush(fp);
      }
      MPI_Barrier(mycomm.comm);
    }
  }
};

#endif
