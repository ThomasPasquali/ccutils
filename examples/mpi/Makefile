MPICXX      = mpicxx
MPICXXFLAGS = -O3 -std=c++17 -I../../include -DCCUTILS_ENABLE_MPI

TARGET    = mpi_example
SRC       = mpi_example.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(MPICXX) $(MPICXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
