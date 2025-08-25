# DeepKNLP

Transformer-based Korean Natural Language Processing

## Code Reference

* ratsgo nlpbook: https://ratsgo.github.io/nlpbook | https://github.com/ratsgo/ratsnlp | https://ratsgo.github.io/nlpbook/docs/tutorial_links
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

* Encoder: https://huggingface.co/docs/transformers/main/en/model_summary#nlp-encoder
    - KPF-BERT: https://huggingface.co/jinmang2/kpfbert | https://github.com/KPFBERT/kpfbert
    - KLUE-BERT: https://huggingface.co/klue/bert-base | https://github.com/KLUE-benchmark/KLUE
    - KcBERT: https://huggingface.co/beomi/kcbert-base | https://github.com/Beomi/KcBERT
    - KoELECTRA: https://huggingface.co/monologg/koelectra-base-v3-discriminator | https://github.com/monologg/KoELECTRA
    - Finetuned by KorQuAD: https://huggingface.co/models?pipeline_tag=question-answering&sort=downloads&search=korquad
* Encoder-Decoder: https://huggingface.co/docs/transformers/main/en/model_summary#nlp-encoder-decoder
    - KoT5: https://huggingface.co/wisenut-nlp-team/KoT5-base | https://github.com/wisenut-research/KoT5
    - KE-T5: https://huggingface.co/KETI-AIR/ke-t5-large | https://github.com/airc-keti/ke-t5
    - pko-T5: https://huggingface.co/paust/pko-t5-large | https://github.com/paust-team/pko-t5
    - Finetuned by KorQuAD: https://huggingface.co/models?pipeline_tag=text2text-generation&sort=downloads&search=korquad
* Decoder: https://huggingface.co/docs/transformers/main/en/model_summary#nlp-decoder
    - KoGPT2(125M): https://huggingface.co/skt/kogpt2-base-v2 | https://github.com/SKT-AI/KoGPT2
    - Ko-GPT-Trinity-1.2B: https://huggingface.co/skt/ko-gpt-trinity-1.2B-v0.5
    - Polyglot-Ko-1.3B: https://huggingface.co/EleutherAI/polyglot-ko-1.3b | https://github.com/EleutherAI/polyglot

## Installation

1. Install Miniforge
    ```bash
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```
2. Clone the repository
    ```bash
    rm -rf DeepKNLP*; git clone https://github.com/chrisjihee/DeepKNLP-25.git; cd DeepKNLP*;
    rm -rf DeepKNLP*; git clone https://git.etri.re.kr/nlp/DeepKNLP-25.git; cd DeepKNLP*;
    ```
3. Create a new environment
    ```bash
    conda search conda -c conda-forge | grep " 25."
    conda install -n base -c conda-forge conda=25.7.0 -y
    conda create -n DeepKNLP-25 python=3.12 -y
    conda install -n DeepKNLP-25 nvidia/label/cuda-12.8.1::cuda -y
    ```
4. Install the required packages
    ```bash
    conda activate DeepKNLP-25  # MUST be activated
    pip list; echo ==========; conda --version; echo ==========; conda list
    pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
    rm -rf transformers; git clone https://github.com/chrisjihee/transformers.git; pip install -U -e transformers
    rm -rf ratsnlp;      git clone https://github.com/chrisjihee/ratsnlp.git;      pip install -U -e ratsnlp
    pip list | grep -E "torch|lightn|trans|accel|speed|flash|numpy|piece|chris|prog|pydantic"
    ```
5. Login to Hugging Face and link the cache
    ```bash
    hf auth whoami
    hf auth login
    rm -f .cache_hf; ln -s ~/.cache/huggingface ./.cache_hf
    ```

## Target Tasks

* Sentence Classification: https://ratsgo.github.io/nlpbook/docs/doc_cls
    - `python task1-cls/run_cls.py --help`
    - `python task1-cls/run_cls.py train`
    - `python task1-cls/run_cls.py test`
    - `python task1-cls/run_cls.py serve`
* Sequence Labelling: https://ratsgo.github.io/nlpbook/docs/ner
    - `python task2-ner/run_ner.py --help`
    - `python task2-ner/run_ner.py train`
    - `python task2-ner/run_ner.py test`
    - `python task2-ner/run_ner.py serve`
* Question Answering: https://ratsgo.github.io/nlpbook/docs/qa
    - `bash task3-qa/train_qa-1.sh`
    - `bash task3-qa/train_qa-2.sh`
    - `bash task3-qa/train_qa-3.sh`
    - `bash task3-qa/train_qa_seq2seq-1.sh`
    - `bash task3-qa/train_qa_seq2seq-2.sh`
    - `bash task3-qa/train_qa_seq2seq-3.sh`
    - `bash task3-qa/eval_qa-1.sh`
    - `bash task3-qa/eval_qa-2.sh`
    - `bash task3-qa/eval_qa-3.sh`
    - `bash task3-qa/eval_qa_seq2seq-1.sh`
    - `bash task3-qa/eval_qa_seq2seq-2.sh`
    - `bash task3-qa/eval_qa_seq2seq-3.sh`
* Sentence Generation: https://ratsgo.github.io/nlpbook/docs/generation
    - `CUDA_VISIBLE_DEVICES=0 python task4-gen/infer_gen-1.py`
    - `CUDA_VISIBLE_DEVICES=1 python task4-gen/infer_gen-2.py`
    - `CUDA_VISIBLE_DEVICES=2 python task4-gen/infer_gen-3.py`
    - `CUDA_VISIBLE_DEVICES=0 python task4-gen/train_gen-1.py`
    - `CUDA_VISIBLE_DEVICES=1 python task4-gen/train_gen-2.py`
    - `CUDA_VISIBLE_DEVICES=2 python task4-gen/train_gen-3.py`
    - `CUDA_VISIBLE_DEVICES=0 python task4-gen/serve_gen-1.py`
    - `CUDA_VISIBLE_DEVICES=1 python task4-gen/serve_gen-2.py`
    - `CUDA_VISIBLE_DEVICES=2 python task4-gen/serve_gen-3.py`
