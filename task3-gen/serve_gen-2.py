from pathlib import Path

import torch

from ratsnlp.nlpbook.generation import GenerationDeployArguments
from ratsnlp.nlpbook.generation import get_web_service_app
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = GenerationDeployArguments(
        pretrained_model_name="skt/ko-gpt-trinity-1.2B-v0.5",
        downstream_model_dir="output/nsmc-gen/train_gen-by-kogpt-trinity",
    )

    pretrained_model_config = GPT2Config.from_pretrained(
        args.pretrained_model_name,
    )
    model = GPT2LMHeadModel(pretrained_model_config)
    fine_tuned_model_ckpt = torch.load(
        args.downstream_model_checkpoint_fpath,
        map_location=device,
    )
    model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
    model.eval()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        args.pretrained_model_name,
        eos_token="</s>",
    )


    def inference_fn(
            prompt,
            min_length=10,
            max_length=20,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            temperature=1.0,
    ):
        try:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    do_sample=True,
                    top_p=float(top_p),
                    top_k=int(top_k),
                    min_length=int(min_length),
                    max_length=int(max_length),
                    repetition_penalty=float(repetition_penalty),
                    no_repeat_ngram_size=int(no_repeat_ngram_size),
                    temperature=float(temperature),
                )
            generated_sentence = tokenizer.decode([el.item() for el in generated_ids[0]])
        except:
            generated_sentence = """처리 중 오류가 발생했습니다. <br>
                변수의 입력 범위를 확인하세요. <br><br> 
                min_length: 1 이상의 정수 <br>
                max_length: 1 이상의 정수 <br>
                top-p: 0 이상 1 이하의 실수 <br>
                top-k: 1 이상의 정수 <br>
                repetition_penalty: 1 이상의 실수 <br>
                no_repeat_ngram_size: 1 이상의 정수 <br>
                temperature: 0 이상의 실수
                """
        return {
            'result': generated_sentence,
        }


    app = get_web_service_app(inference_fn, template_folder=Path("templates").resolve(), server_page="serve_gen.html")
    app.run(host="0.0.0.0", port=9002)
