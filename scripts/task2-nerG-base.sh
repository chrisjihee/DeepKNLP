cd ~/proj/DeepKNLP || exit
export CUDA_VISIBLE_DEVICES=4,5,6,7
#python task2-nerG.py train --run_version EAGLE-1B-base --pretrained etri-lirs/egpt-1.3b-preview
#python task2-nerG.py train --run_version EAGLE-3B-base --pretrained etri-lirs/eagle-3b-preview
#python task2-nerG.py train --run_version LLaMA-3B-base --pretrained meta-llama/Llama-3.2-3B
python task2-nerG.py train --run_version FlanT5-3B-base --pretrained google/flan-t5-xl
