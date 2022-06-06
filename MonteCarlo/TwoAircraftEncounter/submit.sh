#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=2:ncpus=24:mem=64G:mpiprocs=24:ompthreads=1
#PBS -l walltime=00:06:00
#PBS -P 12001778
#PBS -N MonteCarloTestProgram
#PBS -M zhihao.quek@ntu.edu.sg
#PBS -m abe

# Commands start here
module load anaconda/3
module load openmpi/gcc493/1.10.2
cd ${PBS_O_WORKDIR}
source activate testenv
mpirun python -m mpi4py.futures Test_mpi4py.py
