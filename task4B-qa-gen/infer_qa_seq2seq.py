import logging
import os

import torch

from chrisbase.io import paths
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Local pretrained model path or Hugging Face Hub ID
# TODO: "output/korquad/train_qa_by-pkot5-*/checkpoint-*", or "paust/pko-t5-base-finetuned-korquad"
pretrained = "output/korquad/train_qa_by-pkot5-*/checkpoint-*"
checkpoint_paths = paths(pretrained)
if checkpoint_paths and len(checkpoint_paths) > 0:
    pretrained = str(sorted(checkpoint_paths, key=os.path.getmtime)[-1])

# 1. Load Tokenizer and Model
logger.info(f"Loading model from {pretrained}")
tokenizer = AutoTokenizer.from_pretrained(pretrained)
model = AutoModelForSeq2SeqLM.from_pretrained(pretrained)
model.eval()  # Set the model to evaluation mode

# 2. Example question/context
context = """대한민국은 동아시아의 한반도 군사 분계선 남부에 위치한 나라이다. 
약칭으로 한국(한국 한자: 韓國)과 남한(한국 한자: 南韓)으로 부르며 현정체제는 대한민국 제6공화국이다. 
대한민국의 국기는 대한민국 국기법에 따라 태극기이며, 국가는 관습상 애국가, 국화는 관습상 무궁화이다. 
공용어는 한국어와 한국 수어이다. 수도는 서울특별시이다. 
인구는 2024년 2월 기준으로 5,130만명이고, 이 중 절반이 넘는(50.74%) 2,603만명이 수도권에 산다."""

questions = [
    "대한민국의 수도는?",
    "대한민국의 국화는?",
    "대한민국의 국가는?",
    "대한민국의 위치는?",
    "대한민국의 약칭은?",
    "대한민국의 약칭 2가지는?",
    "대한민국의 인구는?",
    "대한민국의 공용어는?",
    "대한민국의 공용어 2가지는?",
    "대한민국의 헌정체제는?",
    "대한민국 대부분의 인구는 어디에 사는가?",
    "한반도는 지구상 어디에 있는가?",
]


# 3. Inference (Using `generate` method)
def answer_question(question, context, max_length=50, num_beams=5):
    """
    Generate an answer using the T5 model.

    :param question: The input question.
    :param context: The input context (passage).
    :param max_length: Maximum length of the generated answer.
    :param num_beams: Number of beams for beam search.
    :return: Generated answer.
    """
    # Format the input for T5 (question-context pair)
    input_text = f"question: {question} context: {context}"

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    # Generate the answer
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=num_beams
        )

    # Decode the generated answer
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return answer


# 4. Print answers for each question
for question in questions:
    answer = answer_question(question, context)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print()
