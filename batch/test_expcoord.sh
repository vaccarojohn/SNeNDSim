#!/bin/bash

#SBATCH --partition=turing
#SBATCH --account=mli:gampix
#SBATCH --job-name=gampixpy
#SBATCH --output=logs/output-%j.txt
#SBATCH --error=logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10g
#SBATCH --gpus=1

SINGULARITY_IMAGE_PATH=/sdf/group/neutrino/images/larcv2_ub22.04-cuda12.1-pytorch2.4.0-larndsim.sif

# INPUT_EDEPSIM=$1
# GAMPIXPY_OUTPUT=$2
# ROTATED_OUTPUT=$3

ID="test"
ROTATED_OUTPUT=/sdf/data/neutrino/jvaccaro/SNeNDSens/temp
TMP_GAMPIXPY_OUTPUT=/sdf/data/neutrino/jvaccaro/SNeNDSens/temp/${ID}.h5

# READOUT_CONFIG=$3
# DETECTOR_CONFIG=$4

GAMPIXROOT=/sdf/home/j/jvaccaro/SNeNDSim/GAMPixPy

COMMAND="python3 ${GAMPIXROOT}/rotate_to_experimental_coordinates.py ${TMP_GAMPIXPY_OUTPUT} -o ${ROTATED_OUTPUT}/temp.h5"
# COMMAND="python3 ${GAMPIXROOT}/examples/rotate_to_experimental_coordinates.py ${TMP_FILE} -o ${OUTPUT_HDF5}"

echo $COMMAND
singularity exec --nv -B /sdf,/lscratch ${SINGULARITY_IMAGE_PATH} ${COMMAND}