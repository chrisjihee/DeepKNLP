# DeepKNLP
Transformer-based Korean Natural Language Processing

## Main Reference
![overview](images/overview.png?raw=true)
* ratsgo ratsnlp: https://github.com/ratsgo/ratsnlp
* ratsgo nlpbook: https://ratsgo.github.io/nlpbook/
* HF Transformers: https://github.com/huggingface/transformers
* Lightning Fabric: https://lightning.ai/docs/fabric/stable/

## Installation

1. Install Miniforge
    ```bash
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```
2. Clone the repository
    ```bash
    rm -rf DeepKNLP*
    git clone git@github.com:chrisjihee/DeepKNLP.git
    cd DeepKNLP*
    ```
3. Create a new environment
    ```bash
    conda update -n base -c conda-forge conda
    conda create -n DeepKNLP python=3.12 -y; conda activate DeepKNLP
    conda install cuda-libraries=11.8 cuda-libraries-dev=11.8 cuda-cudart=11.8 cuda-cudart-dev=11.8 \
                  cuda-nvrtc=11.8 cuda-nvrtc-dev=11.8 cuda-driver-dev=11.8 \
                  cuda-nvcc=11.8 cuda-cccl=11.8 cuda-runtime=11.8 cuda-version=11.8 \
                  libcusparse=11 libcusparse-dev=11 libcublas=11 libcublas-dev=11 \
                  -c nvidia -c pytorch -y
    ```
4. Install the required packages
    ```bash
    pip install -r requirements.txt
    DS_BUILD_FUSED_ADAM=1 pip install --no-cache deepspeed; ds_report
    MAX_JOBS=12 pip install flash-attn==2.0.4 --no-build-isolation;  # for Phi family models
    rm -rf transformers; git clone git@github.com:chrisjihee/transformers.git; pip install -U -e transformers
    rm -rf chrisbase;    git clone git@github.com:chrisjihee/chrisbase.git;    pip install -U -e chrisbase
    rm -rf chrisdata;    git clone git@github.com:chrisjihee/chrisdata.git;    pip install -U -e chrisdata
    rm -rf progiter;     git clone git@github.com:chrisjihee/progiter.git;     pip install -U -e progiter
    pip list | grep -E "torch|lightn|trans|accel|speed|numpy|piece|chris|prog|pydantic"
    ```
5. Unzip some archived data
    ```bash
    cd data/gner; tar zxf each-EQ.tar.gz; tar zxf each-sampled-EQ.tar.gz; cd ../..;
    ```
6. Log in to Huggingface
    ```bash
    huggingface-cli whoami
    huggingface-cli login
    ```
7. Link huggingface cache (optional)
    ```bash
    ln -s ~/.cache/huggingface ./.cache_hf
    ```

## Target Task
* Document Classification: https://ratsgo.github.io/nlpbook/docs/doc_cls/overview/
  - `python task1-cls.py --help`
  - `python task1-cls.py train --help`
  - `python task1-cls.py test --help`
  - `python task1-cls.py serve --help`
* Word Sequence Labelling: https://ratsgo.github.io/nlpbook/docs/ner/overview/
  - `python task2-ner.py --help`
  - `python task2-ner.py train --help`
  - `python task2-ner.py test --help`
  - `python task2-ner.py serve --help`
* Word Sequence Labelling (GenerativeNER) [no-trainer]:
  - `python task2-nerG-no-trainer.py --help`
  - `python task2-nerG-no-trainer.py train --help`
* [Not Yet Provided] Sentence Pair Classification: https://ratsgo.github.io/nlpbook/docs/pair_cls/overview/
* [Not Yet Provided] Extractive Question Answering: https://ratsgo.github.io/nlpbook/docs/qa/overview/
