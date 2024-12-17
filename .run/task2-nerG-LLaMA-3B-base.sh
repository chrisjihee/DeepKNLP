cd ~/proj/DeepKNLP || exit
export CUDA_VISIBLE_DEVICES=4,5,6,7
python task2-nerG.py train --run_version LLaMA-3B-base
