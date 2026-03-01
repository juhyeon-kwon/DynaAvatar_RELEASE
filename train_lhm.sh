# bash train_lhm.sh LHM-500M-HF
export CUDA_VISIBLE_DEVICES=4,5
MODEL_NAME=LHM-500M-HF # just default

ACC_CONFIG="./configs/accelerate-train.yaml"
TRAIN_CONFIG="./configs/train/human-lrm-500M-dynamic.yaml" # just default
MODEL_NAME=${1:-$MODEL_NAME}
echo "MODEL_NAME: $MODEL_NAME"
TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --config_file $ACC_CONFIG -m LHM.launch train.human_lrm model_name=$MODEL_NAME --config $TRAIN_CONFIG
