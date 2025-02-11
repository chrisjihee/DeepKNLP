import os
import random
import socket
import subprocess

# Environment variables
debugging = False
port = random.randint(25000, 30000)
hostname = socket.gethostname()
cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3" if not debugging else "0,1")
large_machines = []

# Training arguments
eval_epochs = 0.5
eval_batch = 50
train_batch = 8 if not debugging else 16
train_epochs = 12
logging_steps = 5
generation_max_length = 640
small_grad_steps = 1
large_grad_steps = 4

# List of pretrained models
models_4B_or_less = [
    ("configs/deepspeed/ds2_t5.json", "FlanT5-Base", "google/flan-t5-base"),
    ("configs/deepspeed/ds2_t5.json", "FlanT5-1B", "google/flan-t5-large"),
    # ("configs/deepspeed/ds2_t5.json", "FlanT5-3B", "google/flan-t5-xl"),

    ("configs/deepspeed/ds2_llama.json", "Llama3-1B", "meta-llama/Llama-3.2-1B"),
    # ("configs/deepspeed/ds2_llama.json", "Llama3-3B", "meta-llama/Llama-3.2-3B"),

    ("configs/deepspeed/ds2_llama.json", "EAGLE-1B", "etri-lirs/egpt-1.3b-preview"),
    # ("configs/deepspeed/ds2_llama.json", "EAGLE-3B", "etri-lirs/eagle-3b-preview"),

    # ("configs/deepspeed/ds2_llama.json", "Qwen2-1B", "Qwen/Qwen2.5-1.5B"),
    # ("configs/deepspeed/ds2_llama.json", "Qwen2-3B", "Qwen/Qwen2.5-3B"),

    # ("configs/deepspeed/ds2_llama.json", "Phi3-4B", "microsoft/Phi-3-mini-4k-instruct"),  # modeling_phi3.py: get_max_length -> get_max_cache_shape
]
models_7B_or_more = [
    # ("configs/deepspeed/ds2_llama.json", "Phi3-7B", "microsoft/Phi-3-small-8k-instruct"),
    # ("configs/deepspeed/ds2_llama.json", "Qwen2-7B", "Qwen/Qwen2.5-7B"),
    # ("configs/deepspeed/ds2_llama.json", "Llama2-7B", "meta-llama/Llama-2-7b-hf"),
    # ("configs/deepspeed/ds2_llama.json", "Llama3-8B", "meta-llama/Llama-3.1-8B"),
    # ("configs/deepspeed/ds2_llama.json", "Mistral-7B", "mistralai/Mistral-7B-Instruct-v0.3"),
    # ("configs/deepspeed/ds2_llama.json", "Ministral-8B", "mistralai/Ministral-8B-Instruct-2410"),

    # ("configs/deepspeed/ds3_t5.json", "FlanT5-11B", "google/flan-t5-xxl"),
    # ("configs/deepspeed/ds3_llama.json", "Mistral-12B", "mistralai/Mistral-Nemo-Instruct-2407"),
    # ("configs/deepspeed/ds3_llama.json", "Llama2-13B", "meta-llama/Llama-2-13b-hf"),
    # ("configs/deepspeed/ds3_llama.json", "Phi4-14B", "microsoft/phi-4"),
    # ("configs/deepspeed/ds3_llama.json", "Qwen2-14B", "Qwen/Qwen2.5-14B"),
]
if hostname not in large_machines:
    models = models_4B_or_less
else:
    models = models_7B_or_more

# List of datasets
datasets = [
    "crossner_ai",
    "crossner_music",
    "crossner_science",
    "crossner_politics",
    "crossner_literature",
    "mit-movie",
    "mit-restaurant",
]
large_datasets = [
    "mit-movie",
    "mit-restaurant",
]

# Loop through each model and dataset
for ds_config, run_prefix, pretrained in models:
    for dataset in datasets:
        suffix = f"-BL"
        eval_dir = f"data/gner/each-sampled{suffix}"
        train_dir = f"data/gner/each{suffix}"
        run_version = f"{run_prefix}{suffix}"
        grad_steps = large_grad_steps if dataset in large_datasets else small_grad_steps
        use_flash_attention = pretrained.startswith("microsoft/Phi")

        command = f"""
            python -m
                deepspeed.launcher.runner
                    --include=localhost:{cuda_devices}
                    --master_port={port}
                task2-nerG-trainer.py
                    --trainer_deepspeed {ds_config}
                    --run_version {run_version}
                    --pretrained {pretrained}
                    --eval_epochs {eval_epochs}
                    --logging_steps {logging_steps}
                    --num_train_epochs {train_epochs}
                    --per_device_eval_batch_size {eval_batch}
                    --per_device_train_batch_size {train_batch}
                    --gradient_accumulation_steps {grad_steps}
                    --generation_max_length {generation_max_length}
                    --eval_file {eval_dir}/{dataset}-dev=100{suffix}.jsonl
                    --train_file {train_dir}/{dataset}-train{suffix}.jsonl
                    --output_file train-metrics-{dataset}-{train_epochs}ep.csv
                    --logging_file train-loggings-{dataset}-{train_epochs}ep.out
                    --{'' if use_flash_attention else 'no_'}use_flash_attention
                    --{'' if debugging else 'no_'}debugging
        """
        command = command.strip().split()
        print("*" * 120)
        print("[COMMAND]", " ".join(command))
        print("*" * 120)

        subprocess.run(command)
        print("\n" * 3)
