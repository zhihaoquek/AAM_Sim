#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=1:mem=4G
#PBS -l walltime=00:01:00
#PBS -P Personal
#PBS -N Check_Config_Python

# Commands start here
module load anaconda/3
cd ${PBS_O_WORKDIR}
source activate mypyenv
python check_scipy.py