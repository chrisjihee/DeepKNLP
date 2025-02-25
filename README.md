# DeepKNLP

Transformer-based Korean Natural Language Processing

## Code Reference

* ratsgo nlpbook: https://ratsgo.github.io/nlpbook | https://github.com/ratsgo/ratsnlp
* Pytorch Lightning: https://github.com/Lightning-AI/pytorch-lightning | https://lightning.ai/docs/fabric/stable
* HF(🤗) Datasets: https://huggingface.co/docs/datasets/index
* HF(🤗) Accelerate: https://huggingface.co/docs/accelerate/index
* HF(🤗) Transformers: https://github.com/huggingface/transformers | https://github.com/huggingface/transformers/tree/main/examples/pytorch

## Data Reference

* NSMC(Naver Sentiment Movie Corpus): https://huggingface.co/datasets/e9t/nsmc | https://github.com/e9t/nsmc
* KLUE(Korean Language Understanding Evaluation): https://huggingface.co/datasets/klue/klue | https://klue-benchmark.com
* KMOU(Korea Maritime and Ocean University) NER: https://huggingface.co/datasets/nlp-kmu/kor_ner | https://github.com/kmounlp/NER
* KorQuAD(Korean Question Answering Dataset): https://huggingface.co/datasets/KorQuAD/squad_kor_v1 | https://korquad.github.io/category/1.0_KOR.html

## Model Reference

* KPF-BERT: https://huggingface.co/jinmang2/kpfbert | https://github.com/KPFBERT/kpfbert
* KLUE-BERT: https://huggingface.co/klue/bert-base | https://github.com/KLUE-benchmark/KLUE
* KcBERT: https://huggingface.co/beomi/kcbert-base | https://github.com/Beomi/KcBERT
* KoELECTRA: https://huggingface.co/monologg/koelectra-base-v3-discriminator | https://github.com/monologg/KoELECTRA

## Installation

1. Install Miniforge
    ```bash
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```
2. Clone the repository
    ```bash
    rm -rf DeepKNLP*; git clone https://github.com/chrisjihee/DeepKNLP-25.git; cd DeepKNLP*;
    ```
3. Create a new environment
    ```bash
    conda search conda -c conda-forge | grep " 25."
    conda install -n base -c conda-forge conda=25.1.1 -y
    conda create -n DeepKNLP-25 python=3.12 -y; conda activate DeepKNLP-25
    conda install cuda-libraries=11.8 cuda-libraries-dev=11.8 cuda-cudart=11.8 cuda-cudart-dev=11.8 \
                  cuda-nvrtc=11.8 cuda-nvrtc-dev=11.8 cuda-driver-dev=11.8 \
                  cuda-nvcc=11.8 cuda-cccl=11.8 cuda-runtime=11.8 cuda-version=11.8 \
                  libcusparse=11 libcusparse-dev=11 libcublas=11 libcublas-dev=11 \
                  -c nvidia -c pytorch -y
    pip list; echo ==========; conda --version; echo ==========; conda list
    ```
4. Install the required packages
    ```bash
    pip install -r requirements.txt
    #export CUDA_HOME=""; DS_BUILD_FUSED_ADAM=1 pip install --no-cache deepspeed; ds_report
    rm -rf transformers; git clone https://github.com/chrisjihee/transformers.git; pip install -U -e transformers
    rm -rf chrisbase;    git clone https://github.com/chrisjihee/chrisbase.git;    pip install -U -e chrisbase
    rm -rf chrisdata;    git clone https://github.com/chrisjihee/chrisdata.git;    pip install -U -e chrisdata
    rm -rf chrislab;     git clone https://github.com/chrisjihee/chrislab.git;     pip install -U -e chrislab
    rm -rf progiter;     git clone https://github.com/chrisjihee/progiter.git;     pip install -U -e progiter
    pip list | grep -E "torch|lightn|trans|accel|speed|flash|numpy|piece|chris|prog|pydantic"
    ```
5. Unzip some archived data
    ```bash
    cd data; tar zxf united.tar.gz; cd ..;
    ```
6. Login to Hugging Face and link the cache
    ```bash
    huggingface-cli whoami
    huggingface-cli login
    ln -s ~/.cache/huggingface ./.cache_hf
    ```

## Target Task

* Text Classification: https://ratsgo.github.io/nlpbook/docs/doc_cls
    - `python task1-cls.py --help`
    - `python task1-cls.py train --help`
    - `python task1-cls.py test --help`
    - `python task1-cls.py serve --help`
* Pair Classification: https://ratsgo.github.io/nlpbook/docs/pair_cls
* Sequence Labelling: https://ratsgo.github.io/nlpbook/docs/ner
    - `python task2-ner.py --help`
    - `python task2-ner.py train --help`
    - `python task2-ner.py test --help`
    - `python task2-ner.py serve --help`
* Question Answering: https://ratsgo.github.io/nlpbook/docs/qa
* Text Generation: https://ratsgo.github.io/nlpbook/docs/generation
