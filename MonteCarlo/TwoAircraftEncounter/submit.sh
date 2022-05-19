#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=24:mem=96G
#PBS -l walltime=00:30:00
#PBS -P 12001778
#PBS -N MonteCarloTestProgram

# Commands start here
module load anaconda/3
cd ${PBS_O_WORKDIR}
source activate mypyenv
python Test_Run.py
