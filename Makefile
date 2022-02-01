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
SEDOV_SOL_CPP := $(SEDOV_SOL_DIR)/sedov_io.cpp        \
                 $(SEDOV_SOL_DIR)/sedov_solution.cpp  \
                 $(SEDOV_SOL_DIR)/main.cpp

NOH_TEST    := src/noh/noh.cpp
NOH_FLAGS   := 
NOH_SOL_DIR := src/analytical_solutions/noh_solution
NOH_SOL_CPP := $(NOH_SOL_DIR)/noh_io.cpp              \
               $(NOH_SOL_DIR)/noh_solution.cpp        \
               $(NOH_SOL_DIR)/main.cpp

EVRARD_TEST  := src/evrard/evrard.cpp
EVRARD_FLAGS := -DGRAVITY

TEST_CASE_FLAGS := $(SEDOV_FLAGS)   \
                   $(NOH_FLAGS)     \
                   $(EVRARD_FLAGS)
 

RELEASE := -DNDEBUG
DEBUG := -D__DEBUG -D_GLIBCXX_DEBUG

# cuda architecture targets
SMS ?= 35 60 70 75
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
GENCODE_FLAGS += -Wno-deprecated-gpu-targets

INC += -Isrc -Iinclude -Idomain/include -I$(CUDA_PATH)/include -I$(PGI_PATH)/include
CXXFLAGS += $(RELEASE)
NVCCFLAGS := -std=c++17 -O3 --expt-relaxed-constexpr -rdc=true $(GENCODE_FLAGS)
NVCCLDFLAGS := $(GENCODE_FLAGS) -rdc=true

CXXFLAGS += -O3 -Wall -Wextra -Wno-unknown-pragmas

ifeq ($(ENV),gnu)
	CXXFLAGS += -std=c++17 -fopenmp -fopenacc -march=native -mtune=native
endif

ifeq ($(ENV),pgi)
	CXXFLAGS += -std=c++17 -mp -dynamic -acc -ta=tesla,cc60 -mp=nonuma -Mcuda -g # -Minfo=accel # prints generated accel functions
endif

ifeq ($(ENV),cray)
	CXXFLAGS += -hstd=c++17 -homp -hacc -dynamic
endif

ifeq ($(ENV),intel)
	CXXFLAGS += -std=c++17 -qopenmp -dynamic
endif

ifeq ($(ENV),clang)
	CXXFLAGS += -march=native -std=c++17 -fopenmp
endif

 
#omp:
#	@mkdir -p $(BINDIR)
#	$(info Linking the executable:)
#	$(CXX) $(CXXFLAGS) $(INC) $(SEDOV_FLAGS)  $(SEDOV_TEST)  -o $(BINDIR)/sedov_$@.app  $(LIB)
#	$(CXX) $(CXXFLAGS) $(INC) $(NOH_FLAGS)    $(NOH_TEST)    -o $(BINDIR)/noh_$@.app    $(LIB)
#	$(CXX) $(CXXFLAGS) $(INC) $(EVRARD_FLAGS) $(EVRARD_TEST) -o $(BINDIR)/evrard_$@.app $(LIB)
#	make solution

mpi+omp:
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI $(SEDOV_FLAGS)  $(SEDOV_TEST)  -o $(BINDIR)/sedov_$@.app  $(LIB)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI $(NOH_FLAGS)    $(NOH_TEST)    -o $(BINDIR)/noh_$@.app    $(LIB)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI $(EVRARD_FLAGS) $(EVRARD_TEST) -o $(BINDIR)/evrard_$@.app $(LIB)
	make solution
    
#omp+cuda: $(BUILDDIR)/cuda_no_mpi.o $(CUDA_OBJS)
#	@mkdir -p $(BINDIR)
#	$(info Linking the executable:)
#	$(NVCC) $(NVCCLDFLAGS) -DUSE_CUDA $(TESTCASE_FLAGS) -dlink -o $(BUILDDIR)/cudalinked.o $(CUDA_OBJS) -lcudadevrt -lcudart
#	$(CXX) $(CXXFLAGS) -o $(BINDIR)/sedov_$@.app $(BUILDDIR)/cudalinked.o $+ -L$(CUDA_PATH)/lib64 -lcudart -lcudadevrt
##	$(CXX) -o $(BINDIR)/sedov_$@.app $+ -L$(CUDA_PATH)/lib64 -lcudart -fopenmp
#	make solution

#omp+target:
#	@mkdir -p $(BINDIR)
#	$(info Linking the executable:)
#	$(CXX) $(CXXFLAGS) $(INC) -DUSE_OMP_TARGET $(SEDOV_FLAGS)  $(SEDOV_TEST)  -o $(BINDIR)/sedov_$@.app  $(LIB)
#	$(CXX) $(CXXFLAGS) $(INC) -DUSE_OMP_TARGET $(NOH_FLAGS)    $(NOH_TEST)    -o $(BINDIR)/noh_$@.app    $(LIB)
#	$(CXX) $(CXXFLAGS) $(INC) -DUSE_OMP_TARGET $(EVRARD_FLAGS) $(EVRARD_TEST) -o $(BINDIR)/evrard_$@.app $(LIB)
#	make solution

mpi+omp+target:
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_OMP_TARGET $(SEDOV_FLAGS)  $(SEDOV_TEST)  -o $(BINDIR)/sedov_$@.app  $(LIB)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_OMP_TARGET $(NOH_FLAGS)    $(NOH_TEST)    -o $(BINDIR)/noh_$@.app    $(LIB)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_OMP_TARGET $(EVRARD_FLAGS) $(EVRARD_TEST) -o $(BINDIR)/evrard_$@.app $(LIB)
	make solution

mpi+omp+acc:
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_STD_MATH_IN_KERNELS $(SEDOV_FLAGS)  -DUSE_ACC $(SEDOV_TEST)  -o $(BINDIR)/sedov_$@.app  $(LIB)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_STD_MATH_IN_KERNELS $(NOH_FLAGS)    -DUSE_ACC $(NOH_TEST)    -o $(BINDIR)/noh_$@.app    $(LIB)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_STD_MATH_IN_KERNELS $(EVRARD_FLAGS) -DUSE_ACC $(EVRARD_TEST) -o $(BINDIR)/evrard_$@.app $(LIB)
	make solution

mpi+omp+cuda: $(BUILDDIR)/cuda_mpi.o $(CUDA_OBJS)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(NVCC) $(NVCCLDFLAGS) -dlink -o $(BUILDDIR)/sedov_cudalinked.o $(CUDA_OBJS) -lcudadevrt -lcudart
	$(MPICXX) $(CXXFLAGS) -o $(BINDIR)/sedov_$@.app $(BUILDDIR)/sedov_cudalinked.o $+ -L$(CUDA_PATH)/lib64 -lcudadevrt -lcudart
	make solution

solution:
	$(MPICXX) $(CXXFLAGS) $(INC) $(SEDOV_SOL_CPP) -o $(BINDIR)/sedov_$@ $(LIB)
	$(MPICXX) $(CXXFLAGS) $(INC) $(NOH_SOL_CPP)   -o $(BINDIR)/noh_$@   $(LIB)

#all: omp mpi+omp omp+cuda mpi+omp+cuda omp+target mpi+omp+target mpi+omp+acc
all: mpi+omp mpi+omp+cuda mpi+omp+target mpi+omp+acc

$(BUILDDIR)/cuda_mpi.o: $(SEDOV_TEST)
	@mkdir -p $(BUILDDIR)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_CUDA $(TEST_CASE_FLAGS) -o $@ -c $<

$(BUILDDIR)/cuda_no_mpi.o: $(SEDOV_TEST)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INC) -DUSE_CUDA $(TEST_CASE_FLAGS) -o $@ -c $<

$(BUILDDIR)/%.o: include/sph/cuda/%.cu
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA $(TEST_CASE_FLAGS) $(INC) -c -o $@ $<
#	$(NVCC) $(NVCCFLAGS) $(INC) -DUSE_STD_MATH_IN_KERNELS -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -c -o $@ $<

$(BUILDDIR)/%.o: domain/include/cstone/cuda/%.cu
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA $(TEST_CASE_FLAGS) $(INC) -c -o $@ $<

#run_test:
#	@$(MAKE) -f $(THIS_FILE) omp
#	@$(MAKE) -f $(THIS_FILE) omp+cuda
#	@$(MAKE) -f $(THIS_FILE) mpi+omp
#	@$(MAKE) -f $(THIS_FILE) mpi+omp+cuda
#	cd test/ && ./test_correctness.sh;

clean:
	$(info Cleaning...)
	$(RM) -rf $(BUILDDIR) $(BINDIR)

.PHONY: all clean
