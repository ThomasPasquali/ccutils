NVCC      = nvcc
NVCCFLAGS = -O2 -std=c++17 -I../../include -DCCUTILS_ENABLE_CUDA

TARGET    = cuda_example
SRC       = cuda_example.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
