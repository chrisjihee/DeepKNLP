#!/bin/bash
# DeepKNLP
mamba create -n DeepKNLP-24.03 python=3.11 -y
mamba activate DeepKNLP-24.03
pip install -r requirements.txt
pip list

# chrisbase (for public user)
rm -rf chrisbase*
pip download --no-binary :all: --no-deps chrisbase==0.5.0; tar zxf chrisbase-*.tar.gz; rm chrisbase-*.tar.gz;
pip install --editable chrisbase*

# chrisbase (for previleged user)
rm -rf chrisbase*
git clone https://github.com/chrisjihee/chrisbase.git
pip install --editable chrisbase*

# pretrained LM (for previleged user)
rm -rf pretrained*
git clone guest@129.254.164.137:git/pretrained

# pretrained LM (for git lfs installed environment)
rm -rf pretrained*
git lfs install
git clone https://github.com/KPFBERT/kpfbert pretrained/KPF-BERT
git clone https://huggingface.co/klue/roberta-base pretrained/KLUE-RoBERTa
git lfs uninstall
