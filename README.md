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
    - KE-T5: https://huggingface.co/KETI-AIR/ke-t5-base | https://github.com/airc-keti/ke-t5
    - pko-T5: https://huggingface.co/paust/pko-t5-base | https://github.com/paust-team/pko-t5
    - Finetuned by KorQuAD: https://huggingface.co/models?pipeline_tag=text2text-generation&sort=downloads&search=korquad
* Decoder: https://huggingface.co/docs/transformers/main/en/model_summary#nlp-decoder
    - KoGPT2(125M): https://huggingface.co/skt/kogpt2-base-v2 | https://github.com/SKT-AI/KoGPT2
    - Ko-GPT-Trinity-1.2B: https://huggingface.co/skt/ko-gpt-trinity-1.2B-v0.5
    - Polyglot-Ko-1.3B: https://huggingface.co/EleutherAI/polyglot-ko-1.3b | https://github.com/EleutherAI/polyglot

## Installation

1. Install uv(Universal Virtualenv)
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. Clone the repository
    ```bash
    rm -rf DeepKNLP*; git clone https://github.com/chrisjihee/DeepKNLP-26.03.git; cd DeepKNLP*;
    ```
3. Monitor Nvidia GPU
    ```bash
    watch -d -n 3 nvidia-smi
    ```
4. Create a new environment and activate it (MUST be activated)
    ```bash
    uv python install 3.12
    uv venv DeepKNLP-env --python 3.12 --python-preference only-managed
    source DeepKNLP-env/bin/activate
    ```
5. Install the required packages
    ```bash
    echo ==========; which python; python --version; echo ==========; uv pip list; echo ==========; uv --version;
    uv pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
    rm -rf transformers; git clone https://github.com/chrisjihee/transformers.git; uv pip install -U -e transformers
    rm -rf ratsnlp;      git clone https://github.com/chrisjihee/ratsnlp.git;      uv pip install -U -e ratsnlp
    uv pip list | grep -E "torch|lightn|trans|accel|speed|flash|numpy|piece|chris|rats|prog|pydantic|tensor"
    ```
6. Login to Hugging Face and link the cache
    ```bash
    hf auth whoami
    hf auth login
    rm -f .cache_hf; ln -s ~/.cache/huggingface ./.cache_hf
    ```
7. Logout from Hugging Face
    ```bash
    hf auth logout
    rm -f ~/.huggingface/token
    rm -f ~/.cache/huggingface/token
    ```

## Target Tasks

* Sentence Classification: https://ratsgo.github.io/nlpbook/docs/doc_cls
    - `See task1-cls/README.md`
    - `python task1-cls/run_cls.py --help`
    - `python task1-cls/run_cls.py train`
    - `python task1-cls/run_cls.py test`
    - `python task1-cls/run_cls.py serve`
* Sequence Labelling: https://ratsgo.github.io/nlpbook/docs/ner
    - `See task2-ner/README.md`
    - `python task2-ner/run_ner.py --help`
    - `python task2-ner/run_ner.py train`
    - `python task2-ner/run_ner.py test`
    - `python task2-ner/run_ner.py serve`
* Sentence Generation: https://ratsgo.github.io/nlpbook/docs/generation
    - `See task3-gen/README.md`
    - `python task3-gen/run_gen.py --help`
    - `python task3-gen/run_gen.py step1 --model-preset kogpt2`
    - `python task3-gen/run_gen.py step2 --model-preset kogpt2 --epochs 1`
    - `python task3-gen/run_gen.py step3 --model-preset kogpt2 --port 9001`
    - `task3-gen/infer_gen.ipynb` is an auxiliary inference notebook, not the main train/serve entrypoint
* Question Answering (Extractive): https://ratsgo.github.io/nlpbook/docs/qa
    - `See task4A-qa-ext/README.md`
    - `python task4A-qa-ext/train_qa.py --help`
    - `python task4A-qa-ext/infer_qa.py`
    - `bash task4A-qa-ext/train_qa-1.sh`
    - `bash task4A-qa-ext/train_qa-2.sh`
    - `bash task4A-qa-ext/train_qa-3.sh`
    - `bash task4A-qa-ext/eval_qa-1.sh`
    - `bash task4A-qa-ext/eval_qa-2.sh`
    - `bash task4A-qa-ext/eval_qa-3.sh`
* Question Answering (Generative):
    - `See task4B-qa-gen/README.md`
    - `python task4B-qa-gen/train_qa_seq2seq.py --help`
    - `python task4B-qa-gen/infer_qa_seq2seq.py`
    - `bash task4B-qa-gen/train_qa_seq2seq-1.sh`
    - `bash task4B-qa-gen/train_qa_seq2seq-2.sh`
    - `bash task4B-qa-gen/eval_qa_seq2seq-1.sh`
    - `bash task4B-qa-gen/eval_qa_seq2seq-2.sh`
