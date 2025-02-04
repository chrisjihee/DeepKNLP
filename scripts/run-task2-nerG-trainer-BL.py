import os
import random
import subprocess
import socket

# Environment variables
debug = True
hostname = socket.gethostname()
port = random.randint(25000, 30000)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Training parameters
run_suffix = "BL"
eval_epochs = 0.5
train_epochs = 12
eval_dir = "data/gner/each-sampled"
train_dir = "data/gner/each"

# List of datasets
datasets = [
    "mit-movie",
    "mit-restaurant",
    "crossner_ai",
    "crossner_music",
    "crossner_science",
    "crossner_politics",
    "crossner_literature",
]

# List of pretrained models
models_4B_or_less = [
    # ("configs/deepspeed/ds3_t5.json", "FlanT5-Small", "google/flan-t5-small"),
    # ("configs/deepspeed/ds3_t5.json", "FlanT5-Base", "google/flan-t5-base"),
    # ("configs/deepspeed/ds3_t5.json", "FlanT5-1B", "google/flan-t5-large"),
    # ("configs/deepspeed/ds3_t5.json", "FlanT5-3B", "google/flan-t5-xl"),

    # ("configs/deepspeed/ds2_llama.json", "EAGLE-1B", "etri-lirs/egpt-1.3b-preview"),
    # ("configs/deepspeed/ds2_llama.json", "EAGLE-3B", "etri-lirs/eagle-3b-preview"),

    # ("configs/deepspeed/ds2_llama.json", "Llama3-1B", "meta-llama/Llama-3.2-1B"),
    # ("configs/deepspeed/ds2_llama.json", "Llama3-3B", "meta-llama/Llama-3.2-3B"),
    # ("configs/deepspeed/ds2_llama.json", "Qwen2-1B", "Qwen/Qwen2.5-1.5B"),
    # ("configs/deepspeed/ds2_llama.json", "Qwen2-3B", "Qwen/Qwen2.5-3B"),
    ("configs/deepspeed/ds2_llama.json", "Phi3-4B", "microsoft/Phi-3.5-mini-instruct"),
    ("configs/deepspeed/ds2_llama.json", "Gemma2-2B", "google/gemma-2-2b"),
    ("configs/deepspeed/ds2_llama.json", "Ministral-3B", "ministral/Ministral-3b-instruct"),
    ("configs/deepspeed/ds2_llama.json", "Ministral-4B", "ministral/Ministral-4b-instruct"),
]
models_7B_or_more = [
    # ("configs/deepspeed/ds2_llama.json", "Llama3-8B", "meta-llama/Llama-3.1-8B"),
    ("configs/deepspeed/ds2_llama.json", "Gemma2-9B", "google/gemma-2-9b"),
    ("configs/deepspeed/ds2_llama.json", "Ministral-8B", "mistralai/Ministral-8B-Instruct-2410"),
    ("configs/deepspeed/ds2_llama.json", "Qwen2-7B", "Qwen/Qwen2.5-7B"),
    ("configs/deepspeed/ds2_llama.json", "Phi3-7B", "microsoft/Phi-3-small-8k-instruct"),
    ("configs/deepspeed/ds2_llama.json", "Gemma-7B", "google/gemma-7b"),
    ("configs/deepspeed/ds2_llama.json", "Mistral-7B", "mistralai/Mistral-7B-Instruct-v0.3"),

    # ("configs/deepspeed/ds2_llama.json", "Alpaca-7B", "allenai/open-instruct-stanford-alpaca-7b"),
    # ("configs/deepspeed/ds2_llama.json", "Llama2-7B", "meta-llama/Llama-2-7b-hf"),

    # ("configs/deepspeed/ds3_t5.json", "FlanT5-11B", "google/flan-t5-xxl"),
    # ("configs/deepspeed/ds3_llama.json", "Llama3-11B", "meta-llama/Llama-3.2-11B-Vision"),
    # ("configs/deepspeed/ds3_llama.json", "Mistral-12B", "mistralai/Mistral-Nemo-Instruct-2407"),
    # ("configs/deepspeed/ds3_llama.json", "Phi3-14B", "microsoft/phi-4"),
    # ("configs/deepspeed/ds3_llama.json", "Qwen2-14B", "Qwen/Qwen2.5-14B"),
    # ("configs/deepspeed/ds3_llama.json", "DeepSeek2-Lite-16B", "deepseek-ai/DeepSeek-V2-Lite"),
    # ("configs/deepspeed/ds3_llama.json", "Mistral-22B", "mistralai/Mistral-Small-Instruct-2409"),
]
if hostname == "lirs-b1":
    models = models_7B_or_more
else:
    models = models_4B_or_less

# Loop through each model and dataset
for ds_config, run_version, pretrained in models:
    run_version = f"{run_version}-{run_suffix}"

    for dataset in datasets:
        grad_steps = 8 if dataset in ["mit-movie", "mit-restaurant"] else 1

        command = f"""
            python -m
                deepspeed.launcher.runner
                    --include=localhost:{os.environ['CUDA_VISIBLE_DEVICES']}
                    --master_port={port}
                task2-nerG-trainer.py
                    --trainer_deepspeed {ds_config}
                    --run_version {run_version}
                    --pretrained {pretrained}
                    --eval_epochs {eval_epochs}
                    --num_train_epochs {train_epochs}
                    --gradient_accumulation_steps {grad_steps}
                    --eval_file {eval_dir}/{dataset}-dev=100.jsonl
                    --train_file {train_dir}/{dataset}-train.jsonl
                    --output_file train-metrics-{dataset}-{train_epochs}ep.csv
                    --logging_file train-loggings-{dataset}-{train_epochs}ep.out
        """
        command = command.strip().split()
        print("*" * 120)
        print("[COMMAND]", " ".join(command))
        print("*" * 120)

        subprocess.run(command)
        print("\n" * 3)
