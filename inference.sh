#!/bin/bash
# given pose sequence, generating animation video .

MODEL_NAME=LHM-1B
IMAGE_INPUT="./train_data/example_imgs/"
MOTION_SEQS_DIR="./train_data/motion_video/mimo6/smplx_params/"
RETURN_GS=false

MODEL_NAME=${1:-$MODEL_NAME}
IMAGE_INPUT=${2:-$IMAGE_INPUT}
MOTION_SEQS_DIR=${3:-$MOTION_SEQS_DIR}

CAM_IDX=${4:-${CAM_IDX:-None}}
START_FRAME_IDX=${5:-${START_FRAME_IDX:-None}}
MOTION_SIZE=${6:-${MOTION_SIZE:-None}}
INPUT_FPS=${7:-${INPUT_FPS:-30}}

RETURN_GS=${8:-$RETURN_GS}


echo "IMAGE_INPUT: $IMAGE_INPUT"
echo "MODEL_NAME: $MODEL_NAME"
echo "MOTION_SEQS_DIR: $MOTION_SEQS_DIR"
echo "RETURN_GS: $RETURN_GS"

if [ "$START_FRAME_IDX" = "None" ]; then
    START_FRAME_TXT="First Frame"
else
    START_FRAME_TXT="$START_FRAME_IDX"
fi

if [ "$CAM_IDX" = "None" ]; then
    CAM_IDX_TXT="Front view"
else
    CAM_IDX_TXT="$CAM_IDX"
fi

if [ "$MOTION_SIZE" = "None" ]; then
    MOTION_SIZE_TXT="Whole frames"
else
    MOTION_SIZE_TXT="$MOTION_SIZE"
fi

echo "START_FRAME_IDX: $START_FRAME_TXT"
echo "CAM_IDX: $CAM_IDX_TXT"
echo "MOTION_SIZE: $MOTION_SIZE_TXT"
echo "INPUT_FPS: $INPUT_FPS"

echo "INFERENCE VIDEO"

MOTION_IMG_DIR=None
VIS_MOTION=true
MOTION_IMG_NEED_MASK=true
RENDER_FPS=15
MOTION_VIDEO_READ_FPS=15
EXPORT_VIDEO=True

python -m LHM.launch infer.human_lrm model_name=$MODEL_NAME \
        image_input=$IMAGE_INPUT \
        export_video=$EXPORT_VIDEO \
        motion_seqs_dir=$MOTION_SEQS_DIR motion_img_dir=$MOTION_IMG_DIR  \
        vis_motion=$VIS_MOTION motion_img_need_mask=$MOTION_IMG_NEED_MASK \
        render_fps=$RENDER_FPS motion_video_read_fps=$MOTION_VIDEO_READ_FPS \
        return_gs=$RETURN_GS \
        start_frame_idx=$START_FRAME_IDX \
        cam_idx=$CAM_IDX \
        motion_size=$MOTION_SIZE \
        input_fps=$INPUT_FPS