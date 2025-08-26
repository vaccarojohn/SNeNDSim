#!/bin/bash

#SBATCH --partition=milano
#SBATCH --account=mli:gampix
#SBATCH --job-name=gampixpy
#SBATCH --output=logs/output-%j.txt
#SBATCH --error=logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10g
#SBATCH --cpus-per-task=8

SINGULARITY_IMAGE_PATH=/sdf/group/neutrino/images/larcv2_ub22.04-cuda12.1-pytorch2.2.1-larndsim.sif

HOME_DIR=/sdf/home/j/jvaccaro
FILE_PATH=$1

COMMAND="python3 ${HOME_DIR}/${FILE_PATH}"
echo $COMMAND
singularity exec --env PYTHONPATH="\$PYTHONPATH:/sdf/home/j/jvaccaro" --nv -B /sdf,/lscratch ${SINGULARITY_IMAGE_PATH} ${COMMAND}