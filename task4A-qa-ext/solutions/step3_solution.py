"""Step 3 answer block for task4A-qa-ext/train_qa.py."""


def complete_step3_build_serve_hint(output_dir):
    return (
        "Step 3 serve command: "
        f'python task4A-qa-ext/serve_qa.py serve --pretrained "{output_dir}/checkpoint-*"'
    )
