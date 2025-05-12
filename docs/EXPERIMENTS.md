# Evaluating and Reproducing MedPromptCLIP Results

Below we provide instructions to reproduce MedPromptCLIP main experimental results using our pre-trained models. We use bash scripts in [scripts/](../scripts/) directory for evaluating ModalCLIP using the PMC-clip model checkpoints. 

Make sure to update the `DATAPATH` variable with dataset path in the script file and run the commands from the main directory `MedPromptCLIP/`. 

## MedPromptCLIP

#### (1) Specific-modal Text Supervision
The specific-modal MedPromptCLIP config files have been provided at `configs/trainers/specific_modal/` directory. Separate config files are present for each dataset, e.g `ODIR.yaml` should be used to train MedPromptCLIP on ODIR. All hyper-parameters such as text-data path, prompt length and prompt depth etc., can be modified using these config files. No hyper-parameters or other settings should be changed in the config file during evaluation of pre-trained models.
You should ensure your /model contains `pmc_vit_l_14_weights.pt`

Now use the script `scripts/MedPromptCLIP/specific_modal.sh` and run the command below to calculate the results:
```bash
# Other possible dataset values includes [ODIR_3x200,f100images,OCT_C8,OTCDL,...]

# evaluate on both base and novel classes using pretrained weights
bash scripts/MedPromptCLIP/specific_modal.sh ODIR output/experiment/ODIR
```

This should evaluate and show results for all datasets in specific-modal text supervision setting.

#### (2) Multi-modal Text Supervision
In this experiment, we first train MedPromptCLIP using a multi-modal template and then evaluate the trained model directly on cross-datasets.

We provide the instructions below to reproduce cross-datasets and domain generalization results using MedPromptCLIP pre-trained models

You need prepare the multi-modal template. 

Then use the script `scripts/MedPromptCLIP/multi_modal_train.sh` and run the commands below to calculate the results for ODIR dataset over 3 seeds:
```bash
#train
bash scripts/MedPromptCLIP/multi_modal_train.sh ODIR output/full_modal
# Other possible dataset values for cross-datasets includes [ODIR_3x200,f100images,OCT_C8,OTCDL,...]

#test
bash scripts/MedPromptCLIP/multi_modal_test.sh f1000images output/full_modal
bash scripts/MedPromptCLIP/multi_modal_test.sh OCT_C8 output/full_modal
bash scripts/MedPromptCLIP/multi_modal_test.sh OTCDL output/full_modal
bash scripts/MedPromptCLIP/multi_modal_test.sh ODIR_3x200 output/full_modal
bash scripts/MedPromptCLIP/multi_modal_test.sh FFA output/full_modal
bash scripts/MedPromptCLIP/multi_modal_test.sh SLO output/full_modal
```

This should evaluate multi-modal text supervision results on eye-related disease and save the log files in `DATASET_multi_modal/` directory.



#### (3) Modality-Text Supervised Few-Shot Learning
In this experiment, we use trained MedPromptCLIP models to perform few-shot learning on different datasets.

You need to first train the MedPromptCLIP model, which is trained on Specific-modal Text Supervision setting, on the few-shot dataset using the script `scripts/MedPromptCLIP/fewshot.sh` 
```bash

Now use the evaluation script `scripts/protext/dg_test.sh` and run the commands below to calculate the results for out of distribution datasets:
```bash
bash scripts/MedPromptCLIP/fewshot.sh ODIR output/experiment/ODIR output/fewshot/ODIR
bash scripts/MedPromptCLIP/fewshot.sh f1000images output/experiment/f1000images output/fewshot/f1000images
bash scripts/MedPromptCLIP/fewshot.sh OCT_C8 output/experiment/OCT_C8 output/fewshot/OCT_C8

```
This should evaluate and save the log files in `output/fewshot/` directory. 





