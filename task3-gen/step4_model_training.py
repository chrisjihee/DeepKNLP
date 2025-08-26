# === Step 4: GPT2 모델 Fine-tuning ===
# 수강생 과제: TODO 부분을 완성하여 NSMC 데이터로 GPT2 모델을 fine-tuning하세요.

import os
import torch
from Korpora import Korpora
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from ratsnlp import nlpbook
from ratsnlp.nlpbook.generation import GenerationTask
from ratsnlp.nlpbook.generation import GenerationTrainArguments
from ratsnlp.nlpbook.generation import NsmcCorpus, GenerationDataset
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

# 토크나이저 병렬 처리 비활성화 (성능 안정성)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    # === Step 4-1: 학습 인수 설정 ===
    
    # TODO: GenerationTrainArguments를 사용하여 학습 설정을 구성하세요
    args = GenerationTrainArguments(
        # TODO: 사전학습 모델명 지정 (힌트: "skt/kogpt2-base-v2")
        pretrained_model_name=# TODO: 완성하세요,
        
        # TODO: Fine-tuning 모델 저장 디렉토리 지정
        downstream_model_dir=# TODO: 완성하세요,
        
        # TODO: 학습에 사용할 데이터셋 이름 지정 (힌트: "nsmc")
        downstream_corpus_name=# TODO: 완성하세요,
        
        # TODO: 최대 시퀀스 길이 설정 (힌트: 32 또는 64)
        max_seq_length=# TODO: 완성하세요,
        
        # TODO: 배치 크기 설정 (GPU 사용 시 32, CPU 사용 시 4)
        batch_size=# TODO: 완성하세요,
        
        # TODO: 학습률 설정 (힌트: 5e-5)
        learning_rate=# TODO: 완성하세요,
        
        # TODO: 에포크 수 설정 (힌트: 3)
        epochs=# TODO: 완성하세요,
        
        # TODO: TPU 코어 수 설정 (GPU 사용 시 0, TPU 사용 시 8)
        tpu_cores=# TODO: 완성하세요,
        
        # TODO: 랜덤 시드 설정 (힌트: 7)
        seed=# TODO: 완성하세요,
    )

    # === Step 4-2: 시드 설정 및 데이터 다운로드 ===
    
    # TODO: 재현 가능한 실험을 위한 시드 설정
    # 힌트: nlpbook.set_seed(args) 사용
    # TODO: 완성하세요

    # TODO: NSMC 데이터셋 다운로드
    # 힌트: Korpora.fetch() 사용
    # TODO: 완성하세요

    # === Step 4-3: 토크나이저 로드 ===
    
    # TODO: 사전학습된 토크나이저 로드
    # 힌트: PreTrainedTokenizerFast.from_pretrained(), eos_token="</s>" 설정
    tokenizer = # TODO: 완성하세요

    # === Step 4-4: 데이터셋 및 데이터로더 구성 ===
    
    # TODO: NSMC 코퍼스 객체 생성
    # 힌트: NsmcCorpus() 사용
    corpus = # TODO: 완성하세요
    
    # TODO: 학습용 데이터셋 생성
    # 힌트: GenerationDataset(args, corpus, tokenizer, mode="train")
    train_dataset = # TODO: 완성하세요
    
    # TODO: 학습용 데이터로더 생성
    # 힌트: RandomSampler 사용, collate_fn=nlpbook.data_collator
    train_dataloader = DataLoader(
        # TODO: 필요한 인수들을 완성하세요
        # dataset, batch_size, sampler, collate_fn, drop_last, num_workers
    )

    # TODO: 검증용 데이터셋 생성 (test 모드 사용)
    # 힌트: GenerationDataset(args, corpus, tokenizer, mode="test")
    val_dataset = # TODO: 완성하세요
    
    # TODO: 검증용 데이터로더 생성
    # 힌트: SequentialSampler 사용
    val_dataloader = DataLoader(
        # TODO: 필요한 인수들을 완성하세요
        # dataset, batch_size, sampler, collate_fn, drop_last, num_workers
    )

    # === Step 4-5: 모델 로드 및 학습 태스크 설정 ===
    
    # TODO: 사전학습된 GPT2 모델 로드
    # 힌트: GPT2LMHeadModel.from_pretrained() 사용
    model = # TODO: 완성하세요

    # TODO: 생성 태스크 객체 생성
    # 힌트: GenerationTask(model, args) 사용
    task = # TODO: 완성하세요
    
    # TODO: Trainer 객체 가져오기
    # 힌트: nlpbook.get_trainer(args) 사용
    trainer = # TODO: 완성하세요

    # === Step 4-6: 모델 학습 실행 ===
    
    # TODO: 학습 실행
    # 힌트: trainer.fit(task, train_dataloaders, val_dataloaders) 사용
    # TODO: 완성하세요

    print("Fine-tuning 완료!")
    print(f"모델이 저장된 위치: {args.downstream_model_dir}")

"""
학습 목표:
1. GPT2 모델의 Fine-tuning 과정 이해
2. ratsnlp 라이브러리를 활용한 효율적 학습 파이프라인 체험
3. NSMC 데이터를 활용한 한국어 텍스트 생성 모델 구축
4. 학습 인수와 데이터 파이프라인의 중요성 인식

핵심 개념:

1. Fine-tuning vs Pre-training:
   - Pre-training: 대량 텍스트로 언어 모델 기본 학습
   - Fine-tuning: 특정 태스크/도메인 데이터로 모델 추가 학습
   - Transfer Learning의 핵심 원리

2. GenerationTrainArguments:
   - 학습 과정의 모든 설정을 관리하는 클래스
   - 모델, 데이터, 하드웨어, 학습 관련 인수 통합 관리
   - 재현 가능한 실험을 위한 필수 요소

3. NSMC 데이터셋:
   - 네이버 영화 리뷰 감정 분류 데이터
   - 한국어 텍스트 생성 학습에 활용
   - 긍정/부정 리뷰 텍스트로 자연스러운 한국어 생성 학습

4. 데이터 파이프라인:
   - NsmcCorpus: 원본 데이터 처리
   - GenerationDataset: 토크나이저와 연동된 데이터셋
   - DataLoader: 배치 단위 학습 데이터 제공

5. GenerationTask:
   - Lightning 기반 학습 태스크 추상화
   - Forward pass, Loss 계산, Optimization 자동 처리
   - 분산 학습, 체크포인트 저장 등 고급 기능 제공

학습 파라미터 설명:
- max_seq_length: 입력 텍스트 최대 길이 (메모리 효율성)
- batch_size: 한 번에 처리할 샘플 수 (GPU 메모리에 따라 조정)
- learning_rate: 가중치 업데이트 크기 (5e-5가 일반적)
- epochs: 전체 데이터를 몇 번 반복 학습할지
- seed: 재현 가능한 실험을 위한 랜덤 시드

기대 효과:
- NSMC 리뷰 스타일의 한국어 텍스트 생성 능력 향상
- 영화 리뷰와 유사한 문체와 어투 학습
- 감정 표현이 풍부한 자연스러운 한국어 생성

주의사항:
- GPU 메모리 부족 시 batch_size 조정 필요
- 학습 시간은 하드웨어 성능에 따라 달라짐
- 과적합 방지를 위해 validation loss 모니터링 중요
"""
