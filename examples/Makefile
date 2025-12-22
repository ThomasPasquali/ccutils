# ================================================================
# User options
# ================================================================
WITH_MPI  ?= 0
WITH_CUDA ?= 0
WITH_HIP  ?= 0

CXX      := g++
CXXFLAGS := -O3 -Wall -std=c++17

# ================================================================
# ccutils include path (mandatory)
# ================================================================
ifndef CCUTILS_INCLUDE
$(warning CCUTILS_INCLUDE is not set!)
$(warning Please set CCUTILS_INCLUDE to the include directory of ccutils)
$(error Example, install by running: "wget -qO- https://raw.githubusercontent.com/ThomasPasquali/ccutils/main/install.sh | env bash")
endif

CXXFLAGS += -I$(CCUTILS_INCLUDE)

# ================================================================
# Optional MPI
# ================================================================
ifeq ($(WITH_MPI),1)
    CXX := mpicxx
    MPI_CFLAGS := $(shell mpicxx --showme:compile)
    MPI_LIBS   := $(shell mpicxx --showme:link)
    CXXFLAGS  += $(MPI_CFLAGS) -DWITH_MPI -DCCUTILS_ENABLE_MPI
endif

# ================================================================
# Optional CUDA
# ================================================================
ifeq ($(WITH_CUDA),1)
    CUDA_HOME ?= $(shell dirname `which nvcc`)/..
    CUDA_CFLAGS := -I$(CUDA_HOME)/include
    CUDA_LIBS   := -L$(CUDA_HOME)/lib64 -lcudart
    CXXFLAGS    += $(CUDA_CFLAGS) -DWITH_CUDA -DCCUTILS_ENABLE_CUDA
endif

# ================================================================
# Optional HIP
# ================================================================
ifeq ($(WITH_HIP),1)
    HIP_PATH ?= $(shell hipconfig --prefix 2>/dev/null)
    ifeq ($(HIP_PATH),)
        $(error HIP not found! Please install ROCm or set HIP_PATH)
    endif
    HIP_CFLAGS := -I$(HIP_PATH)/include
    HIP_LIBS   := -L$(HIP_PATH)/lib -lhiprtc -lhip_hcc
    CXXFLAGS  += $(HIP_CFLAGS) -DWITH_HIP -DCCUTILS_ENABLE_HIP
endif

# ================================================================
# Examples
# ================================================================
EXAMPLES = prints timers
OBJS     = $(EXAMPLES:%=%.o)

# ================================================================
# Build rules
# ================================================================
all: $(EXAMPLES)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%: %.o
	$(CXX) $< -o $@ $(MPI_LIBS) $(CUDA_LIBS)

# ================================================================
# Clean
# ================================================================
clean:
	rm -f $(EXAMPLES) $(OBJS)

.PHONY: all clean
