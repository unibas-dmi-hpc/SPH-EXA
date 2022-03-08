CXX ?= g++ # This is the main compiler
# CXX := clang --analyze # and comment out the linker last line for sanity

CC ?= gcc

MPICXX ?= mpic++ -DOMPI_SKIP_MPICXX

ENV ?= gnu

NVCC ?= $(CUDA_PATH)/bin/nvcc

CUDA_PATH ?= /usr/local/cuda

SRCDIR := src
BINDIR := bin
BUILDDIR := build

THIS_FILE := $(lastword $(MAKEFILE_LIST))

CUDA_OBJS := $(BUILDDIR)/gather.o                     \
             $(BUILDDIR)/findneighbors.o              \
             $(BUILDDIR)/cudaDensity.o                \
             $(BUILDDIR)/cudaIAD.o                    \
             $(BUILDDIR)/cudaMomentumAndEnergyIAD.o

SEDOV_TEST    := src/sedov/sedov.cpp
SEDOV_FLAGS   := 
SEDOV_SOL_DIR := src/analytical_solutions/sedov_solution
SEDOV_SOL_CPP := $(SEDOV_SOL_DIR)/main.cpp

NOH_TEST    := src/noh/noh.cpp
NOH_FLAGS   := 
NOH_SOL_DIR := src/analytical_solutions/noh_solution
NOH_SOL_CPP := $(NOH_SOL_DIR)/main.cpp

EVRARD_TEST  := src/evrard/evrard.cpp
EVRARD_FLAGS := -DGRAVITY

TEST_CUDA_FLAGS := $(SEDOV_FLAGS)  \
                   $(NOH_FLAGS)    \
                   $(EVRARD_FLAGS) \

RELEASE := -DNDEBUG
DEBUG := -D__DEBUG -D_GLIBCXX_DEBUG

# cuda architecture targets
SMS ?= 35 60 70 75
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
GENCODE_FLAGS += -Wno-deprecated-gpu-targets

INC += -Isrc -Iryoanji/src -Iinclude -Idomain/include -I$(CUDA_PATH)/include
CXXFLAGS += $(RELEASE)
NVCCFLAGS := -std=c++17 -O3 --expt-relaxed-constexpr -rdc=true $(GENCODE_FLAGS)
NVCCLDFLAGS := $(GENCODE_FLAGS) -rdc=true

CXXFLAGS += -O3 -Wall -Wextra -Wno-unknown-pragmas

ifeq ($(ENV),gnu)
	CXXFLAGS += -std=c++17 -fopenmp -march=native -mtune=native
endif

ifeq ($(ENV),intel)
	CXXFLAGS += -std=c++17 -qopenmp -dynamic
endif

ifeq ($(ENV),clang)
	CXXFLAGS += -march=native -std=c++17 -fopenmp
endif

all: mpi cuda solution

mpi:
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI $(SEDOV_FLAGS)  $(SEDOV_TEST)  -o $(BINDIR)/sedov  $(LIB)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI $(NOH_FLAGS)    $(NOH_TEST)    -o $(BINDIR)/noh    $(LIB)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI $(EVRARD_FLAGS) $(EVRARD_TEST) -o $(BINDIR)/evrard $(LIB)

cuda:
	make sedov-cuda
	make noh-cuda

sedov-cuda: $(CUDA_OBJS)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_CUDA $(SEDOV_FLAGS) -o $(BUILDDIR)/cuda_mpi.o -c $(SEDOV_TEST)
	$(NVCC) $(NVCCLDFLAGS) -dlink -o $(BUILDDIR)/cudalinked.o $(CUDA_OBJS) -lcudadevrt -lcudart
	$(MPICXX) $(CXXFLAGS) -o $(BINDIR)/$@ $(BUILDDIR)/cudalinked.o $(BUILDDIR)/cuda_mpi.o $+ -L$(CUDA_PATH)/lib64 -lcudadevrt -lcudart
	$(RM) -rf $(BUILDDIR)

noh-cuda: $(CUDA_OBJS)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_CUDA $(NOH_FLAGS) -o $(BUILDDIR)/cuda_mpi.o -c $(NOH_TEST)
	$(NVCC) $(NVCCLDFLAGS) -dlink -o $(BUILDDIR)/cudalinked.o $(CUDA_OBJS) -lcudadevrt -lcudart
	$(MPICXX) $(CXXFLAGS) -o $(BINDIR)/$@ $(BUILDDIR)/cudalinked.o $(BUILDDIR)/cuda_mpi.o $+ -L$(CUDA_PATH)/lib64 -lcudadevrt -lcudart
	$(RM) -rf $(BUILDDIR)

$(BUILDDIR)/%.o: sph/include
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA $(TEST_CUDA_FLAGS) $(INC) -c -o $@ $<

$(BUILDDIR)/%.o: domain/include/cstone/cuda/%.cu
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA $(TEST_CUDA_FLAGS) $(INC) -c -o $@ $<

solution:
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) $(SEDOV_SOL_CPP) -o $(BINDIR)/sedov_$@ $(LIB)
	$(MPICXX) $(CXXFLAGS) $(INC) $(NOH_SOL_CPP)   -o $(BINDIR)/noh_$@   $(LIB)

#run_test:
#	cd test/ && ./test_correctness.sh;

test:
	make clean
	make -j mpi
	make -j solution
	bin/sedov -n 50  -s 200  -w 200  --outDir ./bin/
	bin/noh   -n 100 -s 1000 -w 1000 --outDir ./bin/
	make compare

test-cuda:
	make clean
	make -j cuda
	make -j solution
	bin/sedov-cuda -n 50  -s 200  -w 200  --outDir ./bin/
	bin/noh-cuda   -n 100 -s 1000 -w 1000 --outDir ./bin/
	make compare

compare:
	python src/analytical_solutions/compare_solutions.py sedov --binary_file \
    bin/sedov_solution --constants_file ./bin/constants_sedov.txt \
    --iteration 200 --nparts 125000 --snapshot_file ./bin/dump_sedov200.dat \
    --out_dir bin/ --error_rho --error_p --error_vel
	python src/analytical_solutions/compare_solutions.py noh --binary_file \
    bin/noh_solution --constants_file ./bin/constants_noh.txt \
    --iteration 1000 --nparts 1000000 --snapshot_file ./bin/dump_noh1000.dat \
    --out_dir bin/ --error_u --error_vel --error_cs
	ls -alF bin/

clean:
	$(info Cleaning...)
	$(RM) -rf $(BUILDDIR) $(BINDIR)

.PHONY: all clean
