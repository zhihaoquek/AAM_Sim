#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=1:mem=4G
#PBS -l walltime=00:20:00
#PBS -P Personal
#PBS -N conda_install

# Commands start here
module load anaconda/3
cd ${PBS_O_WORKDIR}
source activate mypyenv
conda install -c conda-forge tqdm -y
