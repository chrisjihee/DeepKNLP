#export CUDA_VISIBLE_DEVICES=4,5,6,7

cd ~/proj/DeepKNLP || exit
#python task2-nerG.py train --run_version FlanT5-3B-KGC2 --pretrained google/flan-t5-xl --study_file data/gner/KG-generation-YAGO3-53220@2.jsonl
#python task2-nerG.py train --run_version LLaMA-3B-KGC2 --pretrained meta-llama/Llama-3.2-3B --study_file data/gner/KG-generation-YAGO3-53220@2.jsonl
python task2-nerG.py train --run_version LLaMA-8B-KGC2 --pretrained meta-llama/Llama-3.1-8B --study_file data/gner/KG-generation-YAGO3-53220@2.jsonl
