# === Step 1: NER 기본 구조 이해 ===
# 수강생 과제: TODO 부분을 완성하여 NERModel 클래스의 기본 초기화를 구현하세요.

# === 기본 라이브러리 ===
import logging  # 로깅 시스템
import os  # 운영체제 인터페이스
from pathlib import Path  # 경로 처리
from time import sleep  # 지연 처리
from typing import List, Tuple, Dict, Mapping, Any  # 타입 힌트

# === PyTorch 관련 ===
import torch  # PyTorch 메인 모듈
import typer  # CLI 프레임워크
from chrisbase.data import AppTyper, JobTimer, ProjectEnv  # 프로젝트 유틸리티
from chrisbase.io import LoggingFormat, make_dir, files, hr  # 입출력 유틸리티
from chrisbase.util import mute_tqdm_cls, tupled  # 진행률 표시 및 유틸리티
from flask import Flask, request, jsonify, render_template  # 웹 프레임워크
from flask_classful import FlaskView, route  # Flask 클래스 기반 뷰

# === Lightning 분산 학습 프레임워크 ===
from lightning import LightningModule  # Lightning 모듈 베이스
from lightning.fabric import Fabric  # 분산 학습 Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger  # 로깅 시스템
from lightning.pytorch.utilities.types import OptimizerLRScheduler  # 옵티마이저 타입

# === PyTorch 학습 관련 ===
from torch import Tensor  # 텐서 타입
from torch.optim import AdamW  # AdamW 옵티마이저
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler  # 데이터 로딩

# === Hugging Face Transformers (NER 특화) ===
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    CharSpan,
)
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import TokenClassifierOutput  # 토큰 분류 모델 출력

# === 프로젝트 내부 모듈 ===
from DeepKNLP.arguments import (
    DataFiles,
    DataOption,
    ModelOption,
    ServerOption,
    HardwareOption,
    PrintingOption,
    LearningOption,
)
from DeepKNLP.arguments import (
    TrainerArguments,
    TesterArguments,
    ServerArguments,
)  # 설정 클래스들
from DeepKNLP.helper import CheckpointSaver, epsilon, fabric_barrier  # 헬퍼 함수들
from DeepKNLP.metrics import (
    accuracy,
    NER_Char_MacroF1,
    NER_Entity_MacroF1,
)  # NER 전용 평가 메트릭
from DeepKNLP.ner import (
    NERCorpus,
    NERDataset,
    NEREncodedExample,
)  # NER 데이터 처리 클래스들

# 로거 및 CLI 앱 초기화
logger = logging.getLogger(__name__)
main = AppTyper()


class NERModel(LightningModule):
    """
    NER(Named Entity Recognition) 모델을 위한 LightningModule 클래스

    주요 기능:
    - BERT 계열 모델을 활용한 토큰 단위 개체명 인식
    - BIO/BILOU 태깅 스킴 지원
    - 토큰 레벨 예측을 문자 레벨로 변환하여 정확한 평가
    - 다양한 NER 메트릭 (정확도, 문자 단위 F1, 개체 단위 F1) 지원
    - 웹 서비스를 통한 실시간 개체명 인식
    """

    def __init__(self, args: TrainerArguments | TesterArguments | ServerArguments):
        """
        NERModel 초기화

        Args:
            args: 학습/테스트/서빙을 위한 설정 인수들
        """
        super().__init__()
        # 설정 저장
        self.args: TrainerArguments | TesterArguments | ServerArguments = args

        # TODO: NER 데이터 코퍼스 초기화
        # 힌트: NERCorpus 클래스를 사용하여 self.data를 초기화하세요
        self.data: NERCorpus = # TODO: 완성하세요

        # TODO: 라벨 정보 및 매핑 딕셔너리 초기화
        # 힌트: self.data.labels를 사용하여 라벨 리스트와 ID 매핑 딕셔너리들을 생성
        # ['O', 'B-PER', 'I-PER', 'B-LOC', ...] 형태의 라벨 리스트
        self.labels: List[str] = # TODO: 완성하세요
        
        # 라벨 → ID 매핑 딕셔너리 (예: {'O': 0, 'B-PER': 1, ...})
        self._label_to_id: Dict[str, int] = # TODO: 완성하세요
        
        # ID → 라벨 매핑 딕셔너리 (예: {0: 'O', 1: 'B-PER', ...})
        self._id_to_label: Dict[int, str] = # TODO: 완성하세요

        # 추론 시 사용할 데이터셋 (validation_step에서 토큰-문자 매핑을 위해 필요)
        self._infer_dataset: NERDataset | None = None

        # 라벨 수 검증
        assert self.data.num_labels > 0, f"Invalid num_labels: {self.data.num_labels}"

        # TODO: 토큰 분류용 사전학습 모델 설정 로드
        # 힌트: AutoConfig.from_pretrained를 사용하여 NER 라벨 수만큼 출력 차원 설정
        self.lm_config: PretrainedConfig = # TODO: 완성하세요

        # TODO: Fast 토크나이저 로드 (문자 오프셋 정보 필요)
        # 힌트: AutoTokenizer.from_pretrained를 사용하여 use_fast=True로 설정
        # NER에서는 token_to_chars() 메소드가 필요하므로 Fast 토크나이저 필수
        self.lm_tokenizer: PreTrainedTokenizerFast = # TODO: 완성하세요
        
        # Fast 토크나이저 검증 (token_to_chars 메소드 필요)
        assert isinstance(
            self.lm_tokenizer, PreTrainedTokenizerFast
        ), f"Our code support only PreTrainedTokenizerFast, not {type(self.lm_tokenizer)}"

        # TODO: 토큰 분류용 사전학습 모델 로드
        # 힌트: AutoModelForTokenClassification.from_pretrained를 사용
        self.lang_model: PreTrainedModel = # TODO: 완성하세요

    @staticmethod
    def label_to_char_labels(label, num_char):
        """
        토큰 레벨 라벨을 문자 레벨 라벨 시퀀스로 변환

        NER에서 하나의 토큰이 여러 문자를 포함할 때, BIO 태깅 규칙에 따라
        첫 번째 문자는 원래 라벨, 나머지 문자들은 I- 라벨로 변환

        Args:
            label: 토큰 레벨 라벨 (예: "B-PER", "O")
            num_char: 해당 토큰이 포함하는 문자 수

        Yields:
            str: 각 문자에 대한 라벨 (예: "B-PER", "I-PER", "I-PER")
        """
        for i in range(num_char):
            if i > 0 and ("-" in label):  # 두 번째 문자부터 && 개체 라벨인 경우
                yield "I-" + label.split("-", maxsplit=1)[-1]  # I- 접두사로 변경
            else:
                yield label  # 첫 번째 문자는 원래 라벨 또는 O 라벨

    def label_to_id(self, x):
        """라벨 문자열을 ID로 변환"""
        return self._label_to_id[x]

    def id_to_label(self, x):
        """ID를 라벨 문자열로 변환"""
        return self._id_to_label[x]

    # TODO: 다음 단계에서 완성할 메소드들
    # def configure_optimizers(self): pass
    # def train_dataloader(self): pass
    # def val_dataloader(self): pass
    # def test_dataloader(self): pass


# TODO: 나머지 함수들과 CLI 명령어들은 다음 단계에서 추가됩니다
if __name__ == "__main__":
    main()
