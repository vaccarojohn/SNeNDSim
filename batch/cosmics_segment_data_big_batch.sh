#!/bin/bash

#SBATCH --partition=turing
#SBATCH --account=mli:gampix
#SBATCH --job-name=gampixpy
#SBATCH --output=logs/output-%j.txt
#SBATCH --error=logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10g
#SBATCH --cpus-per-task=8
#SBATCH --array=0-99

SINGULARITY_IMAGE_PATH=/sdf/group/neutrino/images/larcv2_ub22.04-cuda12.1-pytorch2.4.0-larndsim.sif

EDEPSIM_ROOT=/sdf/home/j/jvaccaro/SNeNDSim/edepsim/cosmics

COMMAND="python3 -u ${EDEPSIM_ROOT}/cosmics_segment_data_big_batch.py $((SLURM_ARRAY_TASK_ID*100)) $(((SLURM_ARRAY_TASK_ID+1)*100))"

echo $COMMAND
singularity exec --env PYTHONPATH=\\$PYTHONPATH:/sdf/home/j/jvaccaro/SNeNDSim --nv -B /sdf,/lscratch ${SINGULARITY_IMAGE_PATH} ${COMMAND}
