#!/bin/bash
# Multi-person face-blend inference.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash inference_face_multi.sh \
#       LHM-1B \
#       /path/to/avatar_imgs_dir \        # N images (sorted) = N persons
#       /path/to/motion_root/person_00/smplx_params \   # ONE person dir; siblings auto-detected
#       <motion_size|None> <input_fps> <interp> <blend_hops>

MODEL_NAME=${1:-LHM-1B}
IMAGE_INPUT=${2:?image dir required}
MOTION_ROOT=${3:?motion root required}
MOTION_SIZE=${4:-None}
INPUT_FPS=${5:-15}
INTERP=${6:-soft}
BLEND_HOPS=${7:-10}
BASE_MODEL_NAME=LHM-1B

echo "MODEL_NAME:   $MODEL_NAME"
echo "IMAGE_INPUT:  $IMAGE_INPUT"
echo "MOTION_ROOT:  $MOTION_ROOT"
echo "MOTION_SIZE:  $MOTION_SIZE  INPUT_FPS: $INPUT_FPS  INTERP: $INTERP"

python -m LHM.launch infer.human_lrm_face_blend_multi \
        model_name=$MODEL_NAME \
        base_model_name=$BASE_MODEL_NAME \
        image_input=$IMAGE_INPUT \
        export_video=True \
        motion_seqs_dir=$MOTION_ROOT motion_img_dir=None \
        vis_motion=false motion_img_need_mask=true \
        render_fps=15 motion_video_read_fps=15 \
        start_frame_idx=None \
        cam_idx=None \
        motion_size=$MOTION_SIZE \
        input_fps=$INPUT_FPS \
        interp=$INTERP \
        blend_hops=$BLEND_HOPS
