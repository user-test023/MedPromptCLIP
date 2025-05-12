
DATASET=$1
EXP=zeroshot
DATAPATH="./DATA"
export CUDA_VISIBLE_DEVICES=1
export CUDA_LAUNCH_BLOCKING=1
# Zeroshot CLIP evaluation
# --seed is only a place holder
python train.py --root ${DATAPATH} --seed 1 --trainer ZeroshotCLIP2 --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/zeroshot/${DATASET}.yaml --output-dir ./output/${EXP}/${DATASET} --eval-only DATASET.SUBSAMPLE_CLASSES all

