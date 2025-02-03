import os
import random
import subprocess

# Enable debug mode
debug = True

# Set CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Generate a random port for distributed training
port = random.randint(25000, 30000)

# Training parameters
num_train_epochs = 12
eval_epochs = 0.5
trainer_deepspeed = "configs/deepspeed/ds2_llama.json"
run_suffix = "Baseline"

# List of pretrained models and their corresponding run versions
models = [
    ("google/flan-t5-large", "FLAN-T5-Large"),
    ("meta-llama/Llama-3.2-1B", "Llama-3-1B"),
    ("etri-lirs/egpt-1.3b-preview", "EAGLE-1B"),
    ("google/flan-t5-xl", "FLAN-T5-3B"),
    ("meta-llama/Llama-3.2-3B", "Llama-3-3B"),
    ("etri-lirs/eagle-3b-preview", "EAGLE-3B"),
    ("microsoft/Phi-3.5-mini-instruct", "Phi-3_5-mini"),
]

# List of datasets
datasets = [
    "crossner_ai",
    "crossner_literature",
    "crossner_music",
    "crossner_politics",
    "crossner_science",
    "mit-movie",
    "mit-restaurant",
]

# Loop through each model and dataset
for pretrained, run_version in reversed(models):
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
                    --train_file data/gner/each/{dataset}-train.jsonl
                    --eval_file data/gner/each-sampled/{dataset}-dev=100.jsonl
                    --num_train_epochs {num_train_epochs}
                    --eval_epochs {eval_epochs}
                    --per_device_train_batch_size {batch_size}
                    --output_file train-metrics-{dataset}-{num_train_epochs}ep.csv
                    --logging_file train-loggings-{dataset}-{num_train_epochs}ep.out
                    --trainer_deepspeed {trainer_deepspeed}
        """
        command = command.strip().split()
        print("*" * 120)
        print("[COMMAND]", " ".join(command))
        print("*" * 120)

        subprocess.run(command)
        print("\n" * 3)
