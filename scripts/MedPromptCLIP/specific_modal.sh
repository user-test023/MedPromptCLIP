DATASET=$1
EXP=$2
DATAPATH='./DATA'
export CUDA_VISIBLE_DEVICES=1
export CUDA_LAUNCH_BLOCKING=1

# --seed is only a place holder
python train.py --root ${DATAPATH} --seed 1 --trainer MedPromptCLIP --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/specific_modal/${DATASET}.yaml --output-dir ${EXP} DATASET.SUBSAMPLE_CLASSES all
