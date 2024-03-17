#!/bin/bash
# DeepKNLP
mamba create -n DeepKNLP-24.03 python=3.11 -y
mamba activate DeepKNLP-24.03
pip install -r requirements.txt
pip list

# chrislab (for public user)
rm -rf chrisbase* chrislab*
pip download --no-binary :all: --no-deps chrisbase==0.5.0; tar zxf chrisbase-*.tar.gz; rm chrisbase-*.tar.gz;
pip download --no-binary :all: --no-deps chrislab==0.7.1; tar zxf chrislab-*.tar.gz; rm chrislab-*.tar.gz;
pip install --editable chrisbase*
pip install --editable chrislab*

# chrislab (for previleged user)
rm -rf chrisbase* chrislab*
git clone https://github.com/chrisjihee/chrisbase.git
git clone https://github.com/chrisjihee/chrislab.git
pip install --editable chrisbase*
pip install --editable chrislab*

# pretrained LM (for public user)
rm -rf pretrained*
git lfs install
git clone https://github.com/KPFBERT/kpfbert pretrained/KPF-BERT
git clone https://huggingface.co/klue/roberta-base pretrained/KLUE-RoBERTa
git clone https://huggingface.co/KETI-AIR/ke-t5-base-ko pretrained/KETI-KeT5
git clone https://huggingface.co/etri-lirs/kebyt5-base-preview pretrained/ETRI-KeByT5
git lfs uninstall

# pretrained LM (for previleged user)
rm -rf pretrained*
ln -s ../pretrained-com pretrained  # git clone guest@129.254.164.137:git/pretrained-com pretrained
ln -s ../pretrained-pro .
