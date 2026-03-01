#!/bin/bash

SEQUENCES=(
    #"0010_03" # Lower
    #"0024_05" # all
    #"0019_09" # Upper
    #"0037_02" # Upper
    #"0343_01" # Lower
    #"0673_04" # Lower
    "0683_04" # Upper
)
# -----------------------------------------------------------------

GPU_ID=0

SCRIPT_FILE="./inference.sh"
MODEL_NAME="LHM-500M"

OUTPUT_DIR="/data/qw00n/LHM/in_the_wild3/"
BASE_DATA_DIR="/data/jane/DNA_Rendering/DNA_Rendering_inference_seqs"

LAST_ARG="True"

for SEQ_ID in "${SEQUENCES[@]}"
do
    SMPLX_PATH="${BASE_DATA_DIR}/${SEQ_ID}/smplx/smplx_params_smooth"
    
    echo "================================================="
    echo "Starting inference for sequence: ${SEQ_ID}"
    echo "================================================="
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} bash ${SCRIPT_FILE} \
        "${MODEL_NAME}" \
        "${OUTPUT_DIR}" \
        "${SMPLX_PATH}" \
        "${LAST_ARG}"
        
    echo "Finished sequence: ${SEQ_ID}"
    echo "" #
done

echo "All inference tasks completed."