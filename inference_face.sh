#!/bin/bash
# Face-blend inference:
#   Phase 1  – base pretrained LHM renders all frames (face quality)
#   Phase 2  – finetuned dynamic model renders all frames, face region is
#              replaced with Phase 1 output (body quality preserved)
#
# Usage (same interface as inference.sh):
#   CUDA_VISIBLE_DEVICES=7 bash inference_face.sh \
#       LHM-500M \
#       /path/to/image_input \
#       /path/to/smplx_params_smooth \
#       <cam_idx|None> <start_frame_idx|None> <motion_size|None> <input_fps>

MODEL_NAME=LHM-500M
BASE_MODEL_NAME=LHM-1B   # static (is_dynamic=false) model for Phase 1 face extraction
IMAGE_INPUT="./train_data/example_imgs/"
MOTION_SEQS_DIR="./train_data/motion_video/mimo6/smplx_params/"

MODEL_NAME=${1:-$MODEL_NAME}
IMAGE_INPUT=${2:-$IMAGE_INPUT}
MOTION_SEQS_DIR=${3:-$MOTION_SEQS_DIR}

CAM_IDX=${4:-${CAM_IDX:-None}}
START_FRAME_IDX=${5:-${START_FRAME_IDX:-None}}
MOTION_SIZE=${6:-${MOTION_SIZE:-None}}
INPUT_FPS=${7:-${INPUT_FPS:-30}}
INTERP=${8:-${INTERP:-hard}}
BLEND_HOPS=${9:-${BLEND_HOPS:-10}}

echo "IMAGE_INPUT:    $IMAGE_INPUT"
echo "MODEL_NAME:     $MODEL_NAME"
echo "MOTION_SEQS_DIR: $MOTION_SEQS_DIR"

if [ "$START_FRAME_IDX" = "None" ]; then START_FRAME_TXT="First Frame"; else START_FRAME_TXT="$START_FRAME_IDX"; fi
if [ "$CAM_IDX"         = "None" ]; then CAM_IDX_TXT="Front view";  else CAM_IDX_TXT="$CAM_IDX";           fi
if [ "$MOTION_SIZE"     = "None" ]; then MOTION_SIZE_TXT="All frames"; else MOTION_SIZE_TXT="$MOTION_SIZE"; fi

echo "START_FRAME_IDX: $START_FRAME_TXT"
echo "CAM_IDX:         $CAM_IDX_TXT"
echo "MOTION_SIZE:     $MOTION_SIZE_TXT"
echo "INPUT_FPS:       $INPUT_FPS"
echo "INTERP:          $INTERP"
echo "BLEND_HOPS:      $BLEND_HOPS"
echo ""
echo "INFERENCE (face-blend)"

MOTION_IMG_DIR=None
VIS_MOTION=true
MOTION_IMG_NEED_MASK=true
RENDER_FPS=15
MOTION_VIDEO_READ_FPS=15
EXPORT_VIDEO=True

python -m LHM.launch infer.human_lrm_face_blend \
        model_name=$MODEL_NAME \
        base_model_name=$BASE_MODEL_NAME \
        image_input=$IMAGE_INPUT \
        export_video=$EXPORT_VIDEO \
        motion_seqs_dir=$MOTION_SEQS_DIR motion_img_dir=$MOTION_IMG_DIR \
        vis_motion=$VIS_MOTION motion_img_need_mask=$MOTION_IMG_NEED_MASK \
        render_fps=$RENDER_FPS motion_video_read_fps=$MOTION_VIDEO_READ_FPS \
        start_frame_idx=$START_FRAME_IDX \
        cam_idx=$CAM_IDX \
        motion_size=$MOTION_SIZE \
        input_fps=$INPUT_FPS \
        interp=$INTERP \
        blend_hops=$BLEND_HOPS
