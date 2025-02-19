
# List of pretrained models
model_specs = [
    # ("configs/deepspeed/ds2_t5.json", "FlanT5-Base", "google/flan-t5-base"),
    ("configs/deepspeed/ds2_t5.json", "FlanT5-Large", "google/flan-t5-large"),
    # ("configs/deepspeed/ds2_t5.json", "FlanT5-3B", "google/flan-t5-xl"),
    # ("configs/deepspeed/ds2_llama.json", "Llama3-1B", "meta-llama/Llama-3.2-1B"),
    # ("configs/deepspeed/ds2_llama.json", "Llama3-3B", "meta-llama/Llama-3.2-3B"),
    # ("configs/deepspeed/ds2_llama.json", "EAGLE-1B", "etri-lirs/egpt-1.3b-preview"),
    # ("configs/deepspeed/ds2_llama.json", "EAGLE-3B", "etri-lirs/eagle-3b-preview"),
    # ("configs/deepspeed/ds2_llama.json", "Qwen2-1B", "Qwen/Qwen2.5-1.5B"),
    # ("configs/deepspeed/ds2_llama.json", "Qwen2-3B", "Qwen/Qwen2.5-3B"),
    # ("configs/deepspeed/ds2_llama.json", "Phi3-4B", "microsoft/Phi-3-mini-4k-instruct"),  # modeling_phi3.py: get_max_length -> get_max_cache_shape

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

# List of datasets
each_datasets = [
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
