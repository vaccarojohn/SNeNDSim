#!/bin/bash

#SBATCH --partition=turing
#SBATCH --account=mli:gampix
#SBATCH --job-name=gampixpy
#SBATCH --output=logs/output-%j.txt
#SBATCH --error=logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10g
#SBATCH --gpus=1
#SBATCH --array=0-9

SINGULARITY_IMAGE_PATH=/sdf/group/neutrino/images/larcv2_ub22.04-cuda12.1-pytorch2.2.1-larndsim.sif

# INPUT_EDEPSIM=$1
# GAMPIXPY_OUTPUT=$2
# ROTATED_OUTPUT=$3

INPUT_START=$1
INPUT_EDEPSIM=/sdf/data/neutrino/jvaccaro/SNeNDSens/edepsim/Cosmics_processed
ROTATED_OUTPUT=/sdf/data/neutrino/jvaccaro/SNeNDSens/gampixpy/Cosmics
FORMATTED=$(printf -v formatted_num "%04d" "$((SLURM_ARRAY_TASK_ID + INPUT_START))"; echo "$formatted_num")

ID=$(cat /proc/sys/kernel/random/uuid) 
TMP_GAMPIXPY_OUTPUT=/sdf/data/neutrino/jvaccaro/SNeNDSens/temp/${ID}.h5

# READOUT_CONFIG=$3
# DETECTOR_CONFIG=$4

GAMPIXROOT=/sdf/home/j/jvaccaro/GAMPixPy

COMMAND="python3 ${GAMPIXROOT}/batch_sim.py "${INPUT_EDEPSIM}/CosmicFlux_g4_${FORMATTED}-processed.h5" -o ${TMP_GAMPIXPY_OUTPUT} -r ${GAMPIXROOT}/gampixpy/readout_config/GAMPixD.yaml -d ${GAMPIXROOT}/gampixpy/detector_config/coh_250.yaml"
# COMMAND="python3 ${GAMPIXROOT}/examples/batch_sim.py ${INPUT_EDEPSIM} -o ${TMP_FILE} -r ${READOUT_CONFIG} -d ${DETECTOR_CONFIG}"

echo $COMMAND
singularity exec --nv -B /sdf,/lscratch ${SINGULARITY_IMAGE_PATH} ${COMMAND}

COMMAND="python3 ${GAMPIXROOT}/rotate_to_experimental_coordinates.py ${TMP_GAMPIXPY_OUTPUT} -o ${ROTATED_OUTPUT}/CosmicFlux_g4_gampixpy_${FORMATTED}.h5 -d ${GAMPIXROOT}/gampixpy/detector_config/coh_250.yaml"
# COMMAND="python3 ${GAMPIXROOT}/examples/rotate_to_experimental_coordinates.py ${TMP_FILE} -o ${OUTPUT_HDF5} -d ${DETECTOR_CONFIG}"

echo $COMMAND
singularity exec --nv -B /sdf,/lscratch ${SINGULARITY_IMAGE_PATH} ${COMMAND}

rm $TMP_GAMPIXPY_OUTPUT
