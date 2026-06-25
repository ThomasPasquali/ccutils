MPICXX      = mpicxx
MPICXXFLAGS = -O3 -std=c++17 -I../../include -DCCUTILS_ENABLE_MPI

TARGETS = mpi_example mpi_mympicomm_example mpi_shuffle_example mpi_vcollauxbuff_example

all: $(TARGETS)

mpi_example: mpi_example.cpp
	$(MPICXX) $(MPICXXFLAGS) $< -o $@

mpi_mympicomm_example: mpi_mympicomm_example.cpp
	$(MPICXX) $(MPICXXFLAGS) $< -o $@

mpi_shuffle_example: mpi_shuffle_example.cpp
	$(MPICXX) $(MPICXXFLAGS) $< -o $@

mpi_vcollauxbuff_example: mpi_vcollauxbuff_example.cpp
	$(MPICXX) $(MPICXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
