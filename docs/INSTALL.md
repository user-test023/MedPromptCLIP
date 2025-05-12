# Environment Setup
In this markdown file, you can create a new conda environment and install dependencies.
 
* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -y -n MedPromptCLIP python=3.8

# Activate the environment
conda activate MedPromptCLIP

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1-f https://download.pytorch.org/whl/torch_stable.html
```

* Enter the code folder and install requirements
```bash

cd MedPromptCLIP/
# Install requirements

pip install -r requirements.txt

# Update setuptools package 
pip install setuptools==59.5.0

# If you encounter an error, try an updated version(optional)
pip install --upgrade setuptools
```

* Install dassl library.
* **Note:** We have modified original dassl library in order to configure text-based data-loaders for training. Therefore it is strongly recommended to directly utilize [Dassl library provided in this repository](../Dassl.pytorch).
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# No need to clone dassl library as it is already present in this repository
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
pip install -e . -v
cd ..
```