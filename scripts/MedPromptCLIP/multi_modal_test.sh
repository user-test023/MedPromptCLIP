DATASET=$1
MODELPATH=$2

DATAPATH='./DATA'
export CUDA_VISIBLE_DEVICES=1
# Evaluate on cross-dataset
python train.py --root ${DATAPATH} --seed 1 --trainer MedPromptCLIP --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/multi_modal/ODIR.yaml --output-dir ./${DATASET}_multi_modal --eval-only --model-dir ${MODELPATH}