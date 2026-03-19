"""Step 3 in-place answer snippets for task3-gen/run_gen.py.

Paste the following blocks into the matching TODO Step 3 locations.

step3 checkpoint loading block:

    pretrained_model_config = GPT2Config.from_pretrained(args.pretrained_model_name)
    model = GPT2LMHeadModel(pretrained_model_config)
    fine_tuned_model_ckpt = torch.load(args.downstream_model_checkpoint_fpath, map_location=device)
    model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt["state_dict"].items()})
    model.eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.pretrained_model_name, eos_token="</s>")

step3 inference_fn block:

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
            result = decode_generation(
                model,
                tokenizer,
                prompt,
                do_sample=True,
                top_p=float(top_p),
                top_k=int(top_k),
                min_length=int(min_length),
                max_length=int(max_length),
                repetition_penalty=float(repetition_penalty),
                no_repeat_ngram_size=int(no_repeat_ngram_size),
                temperature=float(temperature),
            )
        except Exception as exc:
            result = f"처리 중 오류가 발생했습니다: {exc}"
        return {"result": result}
"""
