import logging
import os

import torch

from chrisbase.io import paths
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

logger = logging.getLogger(__name__)

# Local pretrained model path or Hugging Face Hub ID
# TODO: "output/korquad/train_qa-*/checkpoint-*" or "monologg/koelectra-base-v3-finetuned-korquad"
pretrained = "output/korquad/train_qa-*/checkpoint-*"
checkpoint_paths = paths(pretrained)
if checkpoint_paths and len(checkpoint_paths) > 0:
    pretrained = str(sorted(checkpoint_paths, key=os.path.getmtime)[-1])

# 1. Load Tokenizer and Model
logger.info(f"Loading model from {pretrained}")
tokenizer = AutoTokenizer.from_pretrained(pretrained)
model = AutoModelForQuestionAnswering.from_pretrained(pretrained)
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


# 3. Inference (Directly calling the model's forward method)
def answer_question(question, context):
    inputs = tokenizer.encode_plus(
        question, context, return_tensors="pt", truncation=True, padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)

    predict_answer_tokens = inputs["input_ids"][0, start_index: end_index + 1]
    answer = tokenizer.decode(predict_answer_tokens)

    return answer


# 4. Print answers for each question
for question in questions:
    answer = answer_question(question, context)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print()
