#!/bin/bash
#SBATCH --job-name="rfm_sphexa_nvprofcuda_sqpatch_002mpi_001omp_40n_3steps_job"
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --account=usup
#SBATCH --output=rfm_sphexa_nvprofcuda_sqpatch_002mpi_001omp_40n_3steps_job.out
#SBATCH --error=rfm_sphexa_nvprofcuda_sqpatch_002mpi_001omp_40n_3steps_job.err
#SBATCH --time=0:15:0
#SBATCH --account=csstaff
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
module load daint-gpu
module unload PrgEnv-cray
module load PrgEnv-gnu
module load CrayGNU/.20.08
module load craype-accel-nvidia60
module load nvhpc
export CRAYPE_LINK_TYPE=dynamic
export OMP_NUM_THREADS=1
module rm xalt
#nvprof --version &> version.rpt
#which nvprof &> which.rpt
# -> Add -o nvprof.%h.%p.%q{SLURM_NODEID}.%q{SLURM_PROCID}.nvvp
# to create 1 file per MPI rank for the Visual Profiler:
# nvvp -i nvprof.nid*
# -> Add --kernels "::put_kernel_name_here:" --analysis-metrics
# to collect advanced nvvp performance data, for example:
# "::computeMomentumAndEnergyIAD:" 
echo starttime=`date +%s`
rm -f report
#srun nvprof -o report --metrics achieved_occupancy ./bin/mpi+omp+cuda -n 100 -s 1 2>&1
srun ./bin/mpi+omp+cuda -n 100 -s 200 -w 200 2>&1
#cat /etc/modprobe.d/nvidia.conf
echo stoptime=`date +%s`

