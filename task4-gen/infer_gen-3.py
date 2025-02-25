# https://colab.research.google.com/github/ratsgo/nlpbook/blob/master/examples/sentence_generation/deploy_colab1.ipynb
import torch

from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

if __name__ == "__main__":
    pretrained = "EleutherAI/polyglot-ko-1.3b"

    model = GPT2LMHeadModel.from_pretrained(pretrained)
    model.eval()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        pretrained,
        eos_token="</s>",
    )

    input_sentence = "안녕하세요" or "대한민국의 수도는"
    input_ids = tokenizer.encode(input_sentence, return_tensors="pt")

    print(f"[1] " + "-" * 80)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=False,
            min_length=10,
            max_length=50,
        )
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    print(f"[2] " + "-" * 80)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=False,
            min_length=10,
            max_length=50,
            num_beams=3,
        )
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    print(f"[3] " + "-" * 80)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=False,
            min_length=10,
            max_length=50,
            num_beams=1,
        )
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    print(f"[4] " + "-" * 80)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=False,
            min_length=10,
            max_length=50,
            no_repeat_ngram_size=3,
        )
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    print(f"[5] " + "-" * 80)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=False,
            min_length=10,
            max_length=50,
            repetition_penalty=1.0,
        )
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    print(f"[6] " + "-" * 80)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=False,
            min_length=10,
            max_length=50,
            repetition_penalty=1.1,
        )
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    print(f"[7] " + "-" * 80)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=False,
            min_length=10,
            max_length=50,
            repetition_penalty=1.2,
        )
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    print(f"[8] " + "-" * 80)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=False,
            min_length=10,
            max_length=50,
            repetition_penalty=1.5,
        )
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    print(f"[9] " + "-" * 80)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=10,
            max_length=50,
            top_k=50,
        )
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    print(f"[10] " + "-" * 80)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=10,
            max_length=50,
            top_k=1,
        )
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    print(f"[11] " + "-" * 80)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=10,
            max_length=50,
            top_k=50,
            temperature=0.01,
        )
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    print(f"[12] " + "-" * 80)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=10,
            max_length=50,
            top_k=50,
            temperature=1.0,
        )
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    print(f"[13] " + "-" * 80)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=10,
            max_length=50,
            top_k=50,
            temperature=100000000.0,
        )
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    print(f"[14] " + "-" * 80)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=10,
            max_length=50,
            top_p=0.92,
        )
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    print(f"[15] " + "-" * 80)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=10,
            max_length=50,
            top_p=0.01,
        )
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    print(f"[16] " + "-" * 80)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=10,
            max_length=50,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            temperature=0.9,
            top_k=50,
            top_p=0.92,
        )
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))
