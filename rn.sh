#!/bin/bash

export CRAYPE_LINK_TYPE=dynamic
export OMP_NUM_THREADS=4

# -> Add -o nvprof.%h.%p.%q{SLURM_NODEID}.%q{SLURM_PROCID}.nvvp
# to create 1 file per MPI rank for the Visual Profiler:
# nvvp -i nvprof.nid*
# -> Add --kernels "::put_kernel_name_here:" --analysis-metrics
# to collect advanced nvvp performance data, for example:
# "::computeMomentumAndEnergyIAD:" 

rm -f report-*
rm -f dump_Sedov*
#sudo nvprof --profile-child-processes -f -o report-%p ./bin/mpi+omp+cuda.app -n 30 -s 2 2>&1
#mpirun -n 2 ./bin/mpi+omp+cuda.app -n 30 -s 2
#mpirun -n 2 ./bin/app -n 50 -s 50 -w 50
mpirun -n 2 ./bin/app --input bigfiles/Test3DEvrardRel.bin

