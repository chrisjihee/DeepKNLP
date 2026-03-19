"""Step 3 answer block for task4B-qa-gen/train_qa_seq2seq.py."""


def complete_step3_build_serve_hint(output_dir):
    return (
        "Step 3 serve command: "
        f'python task4B-qa-gen/serve_qa_seq2seq.py serve --pretrained "{output_dir}/checkpoint-*"'
    )
