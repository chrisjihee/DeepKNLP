import os
import random
import socket
import subprocess

from base import *

# Environment variables
debugging = False
port = random.randint(25000, 30000)
hostname = socket.gethostname()
cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3" if not debugging else "0")

# Training arguments
experiment_type = "BL"
generation_max_length = 640
train_epochs = 12
eval_epochs = 0.5
logging_steps = 5
small_grad_steps = 1
large_grad_steps = 4
train_batch = 16
eval_batch = 25

# Loop through each model and dataset
for ds_config, run_prefix, pretrained in model_specs:
    for dataset in each_datasets:
        suffix = f"-{experiment_type}"
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
                    --eval_file {eval_dir}/{dataset}-dev=100.jsonl
                    --train_file {train_dir}/{dataset}-train.jsonl
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
