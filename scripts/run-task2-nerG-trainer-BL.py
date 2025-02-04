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
train_epochs = 12
eval_epochs = 0.5
trainer_deepspeed = "configs/deepspeed/ds2_llama.json"
run_suffix = "Baseline"
train_dir = "data/gner/each"
eval_dir = "data/gner/each-sampled"

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
models_3B_or_less = [
    ("google/flan-t5-small", "FLAN-T5-Small"),
    ("google/flan-t5-base", "FLAN-T5-Base"),
    # ("google/flan-t5-large", "FLAN-T5-Large"),
    # ("google/flan-t5-xl", "FLAN-T5-3B"),

    # ("etri-lirs/egpt-1.3b-preview", "EAGLE-1B"),
    # ("meta-llama/Llama-3.2-1B", "Llama-3-1B"),
    ("Qwen/Qwen2.5-1.5B", "Qwen2-1_5B"),
    # ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-1_5B"),

    # ("etri-lirs/eagle-3b-preview", "EAGLE-3B"),
    # ("meta-llama/Llama-3.2-3B", "Llama-3-3B"),
    ("Qwen/Qwen2.5-3B", "Qwen2-3B"),
]
models_7B_or_more = [
    ("google/flan-t5-xxl", "FLAN-T5-11B"),

    # ("meta-llama/Llama-2-7b-hf", "Llama-2-7B"),
    # ("meta-llama/CodeLlama-7b-hf", "CodeLlama-7B"),
    # ("meta-llama/Llama-3.1-8B", "Llama-3-8B"),
    # ("meta-llama/Llama-3.2-11B-Vision", "Llama-3-11B-Vision"),
    # ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-8B"),
    # ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "DeepSeek-R1-14B"),

    # ("Qwen/Qwen2.5-7B", "Qwen2-7B"),
    # ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-7B"),

    # ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B"),
    # ("mistralai/Ministral-8B-Instruct-2410", "Ministral-8B"),
    # ("allenai/open-instruct-stanford-alpaca-7b", "Stanford-Alpaca-7B"),

    # ("deepseek-ai/DeepSeek-V2-Lite", "DeepSeek-V2-Lite-16B"),
    # ("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", "DeepSeek-Coder-V2-16B"),
    # ("mistralai/Mistral-Small-Instruct-2409", "Mistral-22B"),
]
if hostname == "lirs-b1":
    models = models_7B_or_more
else:
    models = models_3B_or_less

# Loop through each model and dataset
for pretrained, run_version in models:
    run_version = f"{run_version}-{run_suffix}"

    for dataset in datasets:
        batch_size = 8 if dataset in ["mit-movie", "mit-restaurant"] else 1

        command = f"""
            python -m
                deepspeed.launcher.runner
                    --include=localhost:{os.environ['CUDA_VISIBLE_DEVICES']}
                    --master_port={port}
                task2-nerG-trainer.py
                    --pretrained {pretrained}
                    --run_version {run_version}
                    --trainer_deepspeed {trainer_deepspeed}
                    --eval_epochs {eval_epochs}
                    --num_train_epochs {train_epochs}
                    --per_device_train_batch_size {batch_size}
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
