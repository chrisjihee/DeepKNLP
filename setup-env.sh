#!/bin/bash
# conda
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# project
mamba create -n DeepKNLP python=3.11 -y
mamba activate DeepKNLP
pip install -r requirements.txt
pip list | grep -E "torch|lightning|transformers|chris"

# chrisbase
rm -rf chrisbase*
pip download --no-binary :all: --no-deps chrisbase==0.5.1; tar zxf chrisbase-*.tar.gz; rm chrisbase-*.tar.gz;
# git clone https://github.com/chrisjihee/chrisbase.git --branch v0.5.1 --single-branch
pip install --editable chrisbase*

# pretrained
rm -rf pretrained*
git lfs install
git clone https://huggingface.co/jinmang2/kpfbert
git clone https://huggingface.co/klue/roberta-base
git clone https://huggingface.co/klue/roberta-small
git clone https://huggingface.co/klue/roberta-large
git clone https://huggingface.co/monologg/koelectra-base-v3-discriminator
git clone https://huggingface.co/monologg/koelectra-small-v3-discriminator
