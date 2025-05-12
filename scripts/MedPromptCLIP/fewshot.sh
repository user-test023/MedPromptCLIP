DATASET=$1
MODELPATH=$2
EXP=$3
DATAPATH='./DATA'
export CUDA_VISIBLE_DEVICES=1
export CUDA_LAUNCH_BLOCKING=1

if [ -d "${EXP}/VLPromptLearner" ]; then
    python train.py --root ${DATAPATH} --seed 1 --trainer Fewshot --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/fewshot/${DATASET}.yaml --output-dir ${EXP} --eval-only --model-dir ${EXP} DATASET.SUBSAMPLE_CLASSES all
else
    python train.py --root ${DATAPATH} --seed 1 --trainer Fewshot --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/fewshot/${DATASET}.yaml --model-dir ${MODELPATH} --output-dir ${EXP} DATASET.SUBSAMPLE_CLASSES all
fi
