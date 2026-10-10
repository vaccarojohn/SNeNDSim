#!/bin/bash

#SBATCH --partition=turing
#SBATCH --account=mli:gampix
#SBATCH --job-name=gampixpy
#SBATCH --output=logs/output-%j.txt
#SBATCH --error=logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10g
#SBATCH --gpus=1
#SBATCH --array=0-39

SINGULARITY_IMAGE_PATH=/sdf/group/neutrino/images/larcv2_ub22.04-cuda12.1-pytorch2.4.0-larndsim.sif
GAMPIXPY_VERSION="0.3.5"

# INPUT_EDEPSIM=$1
# GAMPIXPY_OUTPUT=$2
# ROTATED_OUTPUT=$3

INPUT_START=$1
INPUT_EDEPSIM=/sdf/data/neutrino/jvaccaro/SNeNDSens/edepsim/NueArCCoutside_processed
ROTATED_OUTPUT=/sdf/data/neutrino/jvaccaro/SNeNDSens/gampixpy/NueArCCoutside
FORMATTED=$(printf -v formatted_num "%04d" "$((SLURM_ARRAY_TASK_ID + INPUT_START))"; echo "$formatted_num")

ID=$(cat /proc/sys/kernel/random/uuid) 
TMP_GAMPIXPY_OUTPUT=/sdf/data/neutrino/jvaccaro/SNeNDSens/temp/${ID}.h5

# READOUT_CONFIG=$3
# DETECTOR_CONFIG=$4

COMMAND="pip3 install gampixpy==${GAMPIXPY_VERSION}"

echo $COMMAND
singularity exec --nv -B /sdf,/lscratch ${SINGULARITY_IMAGE_PATH} ${COMMAND}

GAMPIXROOT=/sdf/home/j/jvaccaro/SNeNDSim/GAMPixPy

COMMAND="python3 ${GAMPIXROOT}/batch_sim.py "${INPUT_EDEPSIM}/nueArCC_sns_g4_${FORMATTED}-processed.h5" -o ${ROTATED_OUTPUT}/nueArCC_sns_g4_gampixpy_${FORMATTED}.h5 -r ${GAMPIXROOT}/readout_config/GAMPixD_cosmics.yaml -d ${GAMPIXROOT}/detector_config/coh_250.yaml"
# COMMAND="python3 ${GAMPIXROOT}/examples/batch_sim.py ${INPUT_EDEPSIM} -o ${TMP_FILE} -r ${READOUT_CONFIG} -d ${DETECTOR_CONFIG}"

echo $COMMAND
singularity exec --nv -B /sdf,/lscratch ${SINGULARITY_IMAGE_PATH} ${COMMAND}

#COMMAND="python3 ${GAMPIXROOT}/rotate_to_experimental_coordinates.py ${TMP_GAMPIXPY_OUTPUT} -o ${ROTATED_OUTPUT}/CosmicFlux_g4_gampixpy_${FORMATTED}.h5"
# COMMAND="python3 ${GAMPIXROOT}/examples/rotate_to_experimental_coordinates.py ${TMP_FILE} -o ${OUTPUT_HDF5}"

#echo $COMMAND
#singularity exec --nv -B /sdf,/lscratch ${SINGULARITY_IMAGE_PATH} ${COMMAND}

#rm $TMP_GAMPIXPY_OUTPUT
