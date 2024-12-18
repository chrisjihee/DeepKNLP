cd ~/proj/DeepKNLP || exit
export CUDA_VISIBLE_DEVICES=4,5,6,7
#python task2-nerG.py train --run_version LLaMA-3B-KGC3 --pretrained meta-llama/Llama-3.2-3B --study_file data/gner/KG-generation-YAGO3-53220@3.jsonl
python task2-nerG.py train --run_version FlanT5-3B-KGC3 --pretrained google/flan-t5-xl --study_file data/gner/KG-generation-YAGO3-53220@3.jsonl
