#!/bin/bash -l
#SBATCH --job-name="sphexa"
#SBATCH --time=00:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-core=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=multithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_ACCEL_TARGET=nvidia60

ENVS="cray pgi gnu intel" #cray pgi" #mpi+omp mpi+omp+target mpi+omp+acc m
MODELS="mpi+omp mpi+omp+target mpi+omp+acc mpi+omp+cuda"

make clean

module load daint-gpu
module load cudatoolkit

lastenv="cray"
for env in $ENVS; do
	module switch PrgEnv-$lastenv PrgEnv-$env
	if [ $env == "pgi" ]; then
		module unload cudatoolkit
		module load cudatoolkit/8.0.61_2.4.9-6.0.7.0_17.1__g899857c
	fi
	lastenv=$env
	for model in $MODELS; do
		echo make ENV=$env MPICXX=CC $model
		make ENV=$env MPICXX=CC $model
		rm -rf build
	done
	for model in $MODELS; do
		srun bin/$model.app -n 100 -s 10 > benchmark/$env-$model.txt
	done
	if [ $env == "pgi" ]; then
		module unload cudatoolkit/8.0.61_2.4.9-6.0.7.0_17.1__g899857c
		module load cudatoolkit
	fi
done

module switch PrgEnv-$lastenv PrgEnv-cray

for env in $ENVS; do
	for model in $MODELS; do
		cat benchmark/$env-$model.txt | grep "Total time for" | cut -d' ' -f 6 | cut -d's' -f 1 > benchmark/$env-$model.time
		awk '{ total += $1; count++ } END { print total/count }' benchmark/$env-$model.time > benchmark/$env-$model.avg
	done
done

printf '%-6s ' "ENV"
for model in $MODELS; do
	printf '%-16s ' $model
done

for env in $ENVS; do
	printf '\n%-6s ' $env
	for model in $MODELS; do
		printf '%-16f ' $(cat benchmark/$env-$model.avg) 
	done
done

# make ENV=cray MPICXX=CC mpi+omp
# make ENV=cray MPICXX=CC mpi+omp+target
# make CUDA_PATH=/opt/nvidia/cudatoolkit9.1/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52/ ENV=cray MPICXX=CC mpi+omp+cuda

# mkdir -p benchmark

# srun --job-name="sphexa" --nodes=4 --ntasks-per-core=2 --ntasks-per-node=1 --cpus-per-task=24 --constraint=gpu --hint=multithread bin/mpi+omp+target.app -n 100 -s 10 > benchmark/mpi+omp+target.txt
# srun --job-name="sphexa" --nodes=4 --ntasks-per-core=2 --ntasks-per-node=1 --cpus-per-task=24 --constraint=gpu --hint=multithread bin/mpi+omp+cuda.app -n 100 -s 10 > benchmark/mpi+omp+cuda.txt
# srun --job-name="sphexa" --nodes=4 --ntasks-per-core=2 --ntasks-per-node=1 --cpus-per-task=24 --constraint=gpu --hint=multithread bin/mpi+omp.app -n 100 -s 10 > benchmark/mpi+omp.txt

# cat benchmark/mpi+omp+target.txt | grep "Total time for" | cut -d' ' -f 6 | cut -d's' -f 1 > benchmark/mpi+omp+target.time
# cat benchmark/mpi+omp+cuda.txt | grep "Total time for" | cut -d' ' -f 6 | cut -d's' -f 1 > benchmark/mpi+omp+cuda.time
# cat benchmark/mpi+omp.txt | grep "Total time for" | cut -d' ' -f 6 | cut -d's' -f 1 > benchmark/mpi+omp.time

# MPI_OMP_TARGET=$(awk '{ total += $1; count++ } END { print total/count }' benchmark/mpi+omp+target.time)
# MPI_OMP_CUDA=$(awk '{ total += $1; count++ } END { print total/count }' benchmark/mpi+omp+cuda.time)
# MPI_OMP=$(awk '{ total += $1; count++ } END { print total/count }' benchmark/mpi+omp.time)

# echo $CC 