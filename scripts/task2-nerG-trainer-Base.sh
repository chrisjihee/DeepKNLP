#!/bin/bash

set -x  # Enable debug mode to print each command before execution

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Generate a random port for distributed training
port=$(shuf -i 25000-30000 -n1)

# Training parameters
num_train_epochs=12
eval_epochs=0.5
trainer_deepspeed="configs/deepspeed/ds2_llama.json"

# List of pretrained models and their corresponding run versions
declare -A models
models=(
  ["google/flan-t5-large"]="FLAN-T5-Large"
  ["google/flan-t5-xl"]="FLAN-T5-3B"
  ["meta-llama/Llama-3.2-1B"]="Llama-3-1B"
  ["meta-llama/Llama-3.2-3B"]="Llama-3-3B"
  ["etri-lirs/egpt-1.3b-preview"]="EAGLE-1B"
  ["etri-lirs/eagle-3b-preview"]="EAGLE-3B"
  ["microsoft/Phi-3.5-mini-instruct"]="Phi-3_5-mini"
)

# List of datasets
datasets=(
  "crossner_ai"
  "crossner_literature"
  "crossner_music"
  "crossner_politics"
  "crossner_science"
  "mit-movie"
  "mit-restaurant"
)

# Loop through each model and dataset
for pretrained in "${!models[@]}"; do
  run_version="${models[$pretrained]}-Baseline"

  for dataset in "${datasets[@]}"; do
    if [[ "$dataset" == "mit-movie" || "$dataset" == "mit-restaurant" ]]; then
      batch_size=8
    else
      batch_size=1
    fi

    # Generate log and metric filenames dynamically
    logging_file="train-loggings-${dataset}-${num_train_epochs}ep.out"
    output_file="train-metrics-${dataset}-${num_train_epochs}ep.csv"

    python -m deepspeed.launcher.runner --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $port \
      task2-nerG-trainer.py "data/gner/each/${dataset}-train.jsonl" \
      --eval_file "data/gner/each-sampled/${dataset}-dev=100.jsonl" \
      --pretrained $pretrained \
      --run_version $run_version \
      --eval_epochs $eval_epochs \
      --num_train_epochs $num_train_epochs \
      --per_device_train_batch_size $batch_size \
      --logging_file $logging_file \
      --output_file $output_file \
      --trainer_deepspeed $trainer_deepspeed
  done
done
