DATASET=$1
MODELPATH=$2
EXP=output_domain_generalization
DATAPATH='./DATA'
export CUDA_VISIBLE_DEVICES=1
# Evaluate on cross-dataset
python train.py --root ${DATAPATH} --seed 1 --trainer MedPromptCLIP --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/text_only_supervised/imagenet.yaml --output-dir ./${EXP}/${DATASET}_domain_generalization --eval-only --model-dir ${MODELPATH}