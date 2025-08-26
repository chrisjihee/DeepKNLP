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

        # NER 데이터 코퍼스 초기화
        self.data: NERCorpus = NERCorpus(args)

        # 라벨 정보 및 매핑 딕셔너리 초기화
        self.labels: List[str] = (
            self.data.labels
        )  # ['O', 'B-PER', 'I-PER', 'B-LOC', ...]
        self._label_to_id: Dict[str, int] = {
            label: i for i, label in enumerate(self.labels)
        }  # 라벨 → ID 매핑
        self._id_to_label: Dict[int, str] = {
            i: label for i, label in enumerate(self.labels)
        }  # ID → 라벨 매핑

        # 추론 시 사용할 데이터셋 (validation_step에서 토큰-문자 매핑을 위해 필요)
        self._infer_dataset: NERDataset | None = None

        # 라벨 수 검증
        assert self.data.num_labels > 0, f"Invalid num_labels: {self.data.num_labels}"

        # 토큰 분류용 사전학습 모델 설정 로드
        self.lm_config: PretrainedConfig = AutoConfig.from_pretrained(
            args.model.pretrained,
            num_labels=self.data.num_labels,  # NER 라벨 수만큼 출력 차원 설정
        )

        # Fast 토크나이저 로드 (문자 오프셋 정보 필요)
        self.lm_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            args.model.pretrained,
            use_fast=True,  # 문자 단위 매핑을 위해 Fast 토크나이저 필수
        )
        # Fast 토크나이저 검증 (token_to_chars 메소드 필요)
        assert isinstance(
            self.lm_tokenizer, PreTrainedTokenizerFast
        ), f"Our code support only PreTrainedTokenizerFast, not {type(self.lm_tokenizer)}"

        # 토큰 분류용 사전학습 모델 로드
        self.lang_model: PreTrainedModel = (
            AutoModelForTokenClassification.from_pretrained(
                args.model.pretrained,
                config=self.lm_config,
            )
        )

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

    def to_checkpoint(self) -> Dict[str, Any]:
        """
        체크포인트 저장을 위한 상태 딕셔너리 생성

        Returns:
            Dict: 모델 상태와 진행 정보를 포함한 딕셔너리
        """
        return {
            "lang_model": self.lang_model.state_dict(),
            "args_prog": self.args.prog,
        }

    def from_checkpoint(self, ckpt_state: Dict[str, Any]):
        """
        체크포인트에서 모델 상태 복원

        Args:
            ckpt_state: 저장된 체크포인트 상태 딕셔너리
        """
        self.lang_model.load_state_dict(ckpt_state["lang_model"])
        self.args.prog = ckpt_state["args_prog"]
        self.eval()

    def load_checkpoint_file(self, checkpoint_file):
        """
        체크포인트 파일에서 모델 로드

        Args:
            checkpoint_file: 체크포인트 파일 경로
        """
        assert Path(
            checkpoint_file
        ).exists(), f"Model file not found: {checkpoint_file}"
        self.fabric.print(f"Loading model from {checkpoint_file}")
        self.from_checkpoint(self.fabric.load(checkpoint_file))

    def load_last_checkpoint_file(self, checkpoints_glob):
        """
        glob 패턴으로 찾은 체크포인트 중 가장 최근 파일 로드

        Args:
            checkpoints_glob: 체크포인트 파일 찾기 패턴
        """
        checkpoint_files = files(checkpoints_glob)
        assert checkpoint_files, f"No model file found: {checkpoints_glob}"
        self.load_checkpoint_file(checkpoint_files[-1])

    def configure_optimizers(self):
        """
        AdamW 옵티마이저 설정

        Returns:
            AdamW: 설정된 학습률을 가진 AdamW 옵티마이저
        """
        return AdamW(self.lang_model.parameters(), lr=self.args.learning.learning_rate)

    def train_dataloader(self):
        """
        학습용 데이터로더 생성

        Returns:
            DataLoader: NER 학습용 데이터로더 (랜덤 샘플링)
        """
        # 분산 학습 시 로깅 설정
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug

        # NER 학습 데이터셋 생성
        train_dataset = NERDataset("train", data=self.data, tokenizer=self.lm_tokenizer)

        # 학습용 데이터로더 생성 (랜덤 셔플, NER 전용 collate 함수)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset, replacement=False),
            num_workers=self.args.hardware.cpu_workers,
            batch_size=self.args.hardware.train_batch,
            collate_fn=self.data.encoded_examples_to_batch,  # NER 전용 배치 생성
            drop_last=False,
        )

        self.fabric.print(
            f"Created train_dataset providing {len(train_dataset)} examples"
        )
        self.fabric.print(
            f"Created train_dataloader providing {len(train_dataloader)} batches"
        )
        return train_dataloader

    def val_dataloader(self):
        """
        검증용 데이터로더 생성

        Returns:
            DataLoader: NER 검증용 데이터로더 (순차 샘플링)
        """
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug

        # NER 검증 데이터셋 생성
        val_dataset = NERDataset("valid", data=self.data, tokenizer=self.lm_tokenizer)

        # 검증용 데이터로더 생성 (순차 순서)
        val_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            num_workers=self.args.hardware.cpu_workers,
            batch_size=self.args.hardware.infer_batch,
            collate_fn=self.data.encoded_examples_to_batch,
            drop_last=False,
        )

        self.fabric.print(f"Created val_dataset providing {len(val_dataset)} examples")
        self.fabric.print(
            f"Created val_dataloader providing {len(val_dataloader)} batches"
        )

        # validation_step에서 토큰-문자 매핑을 위해 데이터셋 저장
        self._infer_dataset = val_dataset
        return val_dataloader

    def test_dataloader(self):
        """
        테스트용 데이터로더 생성

        Returns:
            DataLoader: NER 테스트용 데이터로더 (순차 샘플링)
        """
        self.fabric.print = logger.info if self.fabric.local_rank == 0 else logger.debug

        # NER 테스트 데이터셋 생성
        test_dataset = NERDataset("test", data=self.data, tokenizer=self.lm_tokenizer)

        # 테스트용 데이터로더 생성 (순차 순서)
        test_dataloader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            num_workers=self.args.hardware.cpu_workers,
            batch_size=self.args.hardware.infer_batch,
            collate_fn=self.data.encoded_examples_to_batch,
            drop_last=False,
        )

        self.fabric.print(
            f"Created test_dataset providing {len(test_dataset)} examples"
        )
        self.fabric.print(
            f"Created test_dataloader providing {len(test_dataloader)} batches"
        )

        # test_step에서 토큰-문자 매핑을 위해 데이터셋 저장
        self._infer_dataset = test_dataset
        return test_dataloader

    def training_step(self, inputs, batch_idx):
        """
        학습 단계에서 한 배치 처리

        Args:
            inputs: 토크나이즈된 입력 데이터 (input_ids, attention_mask, labels, example_ids)
            batch_idx: 배치 인덱스

        Returns:
            Dict: loss와 accuracy를 포함한 딕셔너리
        """
        # example_ids는 학습에 불필요하므로 제거
        inputs.pop("example_ids")

        # 토큰 분류 모델 순전파
        outputs: TokenClassifierOutput = self.lang_model(**inputs)

        # 라벨과 예측값 추출
        labels: torch.Tensor = inputs["labels"]
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)

        # 정확도 계산 (패딩 토큰 ignore_index=0 제외)
        acc: torch.Tensor = accuracy(preds=preds, labels=labels, ignore_index=0)

        return {
            "loss": outputs.loss,  # 토큰 분류 손실
            "acc": acc,  # 토큰 레벨 정확도
        }

    @torch.no_grad()
    def validation_step(self, inputs, batch_idx):
        """
        검증 단계에서 한 배치 처리 - NER 특화 복잡한 토큰-문자 매핑 수행

        NER의 핵심: 토큰 레벨 예측을 문자 레벨로 변환하여 정확한 평가
        - 토큰 경계와 문자 경계가 다름 (서브워드 토크나이제이션)
        - BIO 태깅 규칙에 따른 라벨 변환
        - 문자 단위 정확한 평가를 위한 오프셋 매핑

        Args:
            inputs: 토크나이즈된 입력 데이터
            batch_idx: 배치 인덱스

        Returns:
            Dict: loss, 문자 레벨 예측값들, 문자 레벨 라벨들
        """
        # 예제 ID 추출 (토큰-문자 매핑을 위해 필요)
        example_ids: List[int] = inputs.pop("example_ids").tolist()

        # 토큰 분류 모델 순전파
        outputs: TokenClassifierOutput = self.lang_model(**inputs)
        preds: torch.Tensor = outputs.logits.argmax(dim=-1)

        dict_of_token_pred_ids: Dict[int, List[int]] = {}
        dict_of_char_label_ids: Dict[int, List[int]] = {}
        dict_of_char_pred_ids: Dict[int, List[int]] = {}
        for token_pred_ids, example_id in zip(preds.tolist(), example_ids):
            token_pred_tags: List[str] = [self.id_to_label(x) for x in token_pred_ids]
            encoded_example: NEREncodedExample = self._infer_dataset[example_id]
            offset_to_label: Dict[int, str] = (
                encoded_example.raw.get_offset_label_dict()
            )
            all_char_pair_tags: List[Tuple[str | None, str | None]] = [
                (None, None)
            ] * len(encoded_example.raw.character_list)
            for token_id in range(self.args.model.seq_len):
                token_span: CharSpan = encoded_example.encoded.token_to_chars(token_id)
                if token_span:
                    char_pred_tags = NERModel.label_to_char_labels(
                        token_pred_tags[token_id], token_span.end - token_span.start
                    )
                    for offset, char_pred_tag in zip(
                        range(token_span.start, token_span.end), char_pred_tags
                    ):
                        all_char_pair_tags[offset] = (
                            offset_to_label[offset],
                            char_pred_tag,
                        )
            valid_char_pair_tags = [(a, b) for a, b in all_char_pair_tags if a and b]
            valid_char_label_ids = [
                self.label_to_id(a) for a, b in valid_char_pair_tags
            ]
            valid_char_pred_ids = [self.label_to_id(b) for a, b in valid_char_pair_tags]
            dict_of_token_pred_ids[example_id] = token_pred_ids
            dict_of_char_label_ids[example_id] = valid_char_label_ids
            dict_of_char_pred_ids[example_id] = valid_char_pred_ids

        if self.args.env.debugging:
            logger.debug(hr())
        list_of_char_pred_ids: List[int] = []
        list_of_char_label_ids: List[int] = []
        for encoded_example in [self._infer_dataset[i] for i in example_ids]:
            char_label_ids = dict_of_char_label_ids[encoded_example.idx]
            char_pred_ids = dict_of_char_pred_ids[encoded_example.idx]
            assert len(char_pred_ids) == len(char_label_ids)
            list_of_char_pred_ids.extend(char_pred_ids)
            list_of_char_label_ids.extend(char_label_ids)
            if self.args.env.debugging:
                token_pred_ids = dict_of_token_pred_ids[encoded_example.idx]
                logger.debug(
                    f"  - encoded_example.idx                = {encoded_example.idx}"
                )
                logger.debug(
                    f"  - encoded_example.raw.entity_list    = ({len(encoded_example.raw.entity_list)}) {encoded_example.raw.entity_list}"
                )
                logger.debug(
                    f"  - encoded_example.raw.origin         = ({len(encoded_example.raw.origin)}) {encoded_example.raw.origin}"
                )
                logger.debug(
                    f"  - encoded_example.raw.character_list = ({len(encoded_example.raw.character_list)}) {' | '.join(f'{x}/{y}' for x, y in encoded_example.raw.character_list)}"
                )
                logger.debug(
                    f"  - encoded_example.encoded.tokens()   = ({len(encoded_example.encoded.tokens())}) {' '.join(encoded_example.encoded.tokens())}"
                )

                def id_label(x):
                    return f"{self.id_to_label(x):5s}"

                logger.debug(
                    f"  - encoded_example.label_ids          = ({len(encoded_example.label_ids)}) {' '.join(map(str, map(id_label, encoded_example.label_ids)))}"
                )
                logger.debug(
                    f"  - encoded_example.token_pred_ids     = ({len(token_pred_ids)}) {' '.join(map(str, map(id_label, token_pred_ids)))}"
                )
                logger.debug(
                    f"  - encoded_example.char_label_ids     = ({len(char_label_ids)}) {' '.join(map(str, map(id_label, char_label_ids)))}"
                )
                logger.debug(
                    f"  - encoded_example.char_pred_ids      = ({len(char_pred_ids)}) {' '.join(map(str, map(id_label, char_pred_ids)))}"
                )
                logger.debug(hr("-"))
        assert len(list_of_char_pred_ids) == len(list_of_char_label_ids)

        if self.args.env.debugging:

            def id_str(x):
                return f"{x:02d}"

            logger.debug(
                f"  - list_of_char_label_ids = ({len(list_of_char_label_ids)}) {' '.join(map(str, map(id_str, list_of_char_label_ids)))}"
            )
            logger.debug(
                f"  - list_of_char_pred_ids  = ({len(list_of_char_pred_ids)}) {' '.join(map(str, map(id_str, list_of_char_pred_ids)))}"
            )
        return {
            "loss": outputs.loss,
            "preds": list_of_char_pred_ids,
            "labels": list_of_char_label_ids,
        }

    @torch.no_grad()
    def test_step(self, inputs, batch_idx):
        """
        테스트 단계에서 한 배치 처리 (검증과 동일)

        Args:
            inputs: 토크나이즈된 입력 데이터
            batch_idx: 배치 인덱스

        Returns:
            Dict: validation_step과 동일한 출력
        """
        return self.validation_step(inputs, batch_idx)

    @torch.no_grad()
    def infer_one(self, text: str):
        """
        단일 텍스트에 대한 NER 추론

        Args:
            text: 개체명 인식을 수행할 텍스트

        Returns:
            Dict: 토큰별 개체명 라벨과 확률을 포함한 결과
        """
        # 텍스트를 튜플로 감싸서 토크나이즈 (batch dimension)
        inputs = self.lm_tokenizer(
            tupled(text),
            max_length=self.args.model.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 토큰 분류 모델 추론
        outputs: TokenClassifierOutput = self.lang_model(**inputs)

        # 각 토큰에 대한 라벨 확률 계산
        all_probs: Tensor = outputs.logits[0].softmax(dim=1)
        top_probs, top_preds = torch.topk(all_probs, dim=1, k=1)

        # 토큰과 라벨 정보 추출
        tokens = self.lm_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        top_labels = [self.id_to_label(pred[0].item()) for pred in top_preds]

        # 특수 토큰 제외하고 결과 구성
        result = []
        for token, label, top_prob in zip(tokens, top_labels, top_probs):
            if token in self.lm_tokenizer.all_special_tokens:
                continue  # [CLS], [SEP], [PAD] 등 제외
            result.append(
                {
                    "token": token,
                    "label": label,
                    "prob": f"{round(top_prob[0].item(), 4):.4f}",
                }
            )

        return {
            "sentence": text,
            "result": result,
        }

    def run_server(self, server: Flask, *args, **kwargs):
        """
        Flask 웹 서버 실행

        Args:
            server: Flask 앱 인스턴스
            *args, **kwargs: Flask 서버 실행 옵션들
        """
        NERModel.WebAPI.register(route_base="/", app=server, init_argument=self)
        server.run(*args, **kwargs)

    class WebAPI(FlaskView):
        """
        Flask 기반 웹 API 클래스 - NER 서비스 제공
        """

        def __init__(self, model: "NERModel"):
            """
            WebAPI 초기화

            Args:
                model: 학습된 NERModel 인스턴스
            """
            self.model = model

        @route("/")
        def index(self):
            """
            메인 페이지 렌더링

            Returns:
                HTML: 웹 인터페이스 페이지
            """
            return render_template(self.model.args.server.page)

        @route("/api", methods=["POST"])
        def api(self):
            """
            NER API 엔드포인트

            POST /api
            Request Body: JSON 형태의 텍스트

            Returns:
                JSON: 개체명 인식 결과 (토큰별 라벨과 확률)
            """
            response = self.model.infer_one(text=request.json)
            return jsonify(response)


def train_loop(
    model: NERModel,
    optimizer: OptimizerLRScheduler,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    checkpoint_saver: CheckpointSaver | None = None,
):
    """
    NER 모델 학습 루프 - 전체 에포크에 걸친 학습 진행

    Args:
        model: 학습할 NERModel
        optimizer: 옵티마이저
        dataloader: 학습 데이터로더
        val_dataloader: 검증 데이터로더
        checkpoint_saver: 체크포인트 저장 관리자 (선택사항)
    """
    fabric = model.fabric
    fabric.barrier()
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    num_batch = len(dataloader)
    print_interval = (
        model.args.printing.print_rate_on_training * num_batch - epsilon
        if model.args.printing.print_step_on_training < 1
        else model.args.printing.print_step_on_training
    )
    check_interval = model.args.learning.check_rate_on_training * num_batch - epsilon
    model.args.prog.global_step = 0
    model.args.prog.global_epoch = 0.0
    for epoch in range(model.args.learning.num_epochs):
        progress = mute_tqdm_cls(bar_size=30, desc_size=8)(
            range(num_batch), unit=f"x{dataloader.batch_size}b", desc="training"
        )
        for i, batch in enumerate(dataloader, start=1):
            model.train()
            model.args.prog.global_step += 1
            model.args.prog.global_epoch = model.args.prog.global_step / num_batch
            optimizer.zero_grad()
            outputs = model.training_step(batch, i)
            fabric.backward(outputs["loss"])
            optimizer.step()
            progress.update()
            fabric.barrier()
            with torch.no_grad():
                model.eval()
                metrics: Mapping[str, Any] = {
                    "step": round(
                        fabric.all_gather(
                            torch.tensor(model.args.prog.global_step * 1.0)
                        )
                        .mean()
                        .item()
                    ),
                    "epoch": round(
                        fabric.all_gather(torch.tensor(model.args.prog.global_epoch))
                        .mean()
                        .item(),
                        4,
                    ),
                    "loss": fabric.all_gather(outputs["loss"]).mean().item(),
                    "acc": fabric.all_gather(outputs["acc"]).mean().item(),
                }
                fabric.log_dict(metrics=metrics, step=metrics["step"])
                if i % print_interval < 1:
                    fabric.print(
                        f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}"
                        f" | {model.args.printing.tag_format_on_training.format(**metrics)}"
                    )
                if model.args.prog.global_step % check_interval < 1:
                    val_loop(model, val_dataloader, checkpoint_saver)
        fabric_barrier(fabric, "[after-epoch]", c="=")
    fabric_barrier(fabric, "[after-train]")


@torch.no_grad()
def val_loop(
    model: NERModel,
    dataloader: DataLoader,
    checkpoint_saver: CheckpointSaver | None = None,
):
    """
    NER 검증 루프 - 전체 검증 데이터에 대한 성능 평가

    NER 특화 메트릭:
    - val_acc: 토큰 레벨 정확도
    - val_F1c: 문자 레벨 Macro F1 (Character-level)
    - val_F1e: 개체 레벨 Macro F1 (Entity-level)

    Args:
        model: 평가할 NERModel
        dataloader: 검증 데이터로더
        checkpoint_saver: 체크포인트 저장 관리자 (선택사항)
    """
    fabric = model.fabric
    fabric.barrier()
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    num_batch = len(dataloader)
    print_interval = (
        model.args.printing.print_rate_on_validate * num_batch - epsilon
        if model.args.printing.print_step_on_validate < 1
        else model.args.printing.print_step_on_validate
    )
    preds: List[int] = []
    labels: List[int] = []
    losses: List[torch.Tensor] = []
    progress = mute_tqdm_cls(bar_size=10, desc_size=8)(
        range(num_batch), unit=f"x{dataloader.batch_size}b", desc="checking"
    )
    for i, batch in enumerate(dataloader, start=1):
        outputs = model.validation_step(batch, i)
        preds.extend(outputs["preds"])
        labels.extend(outputs["labels"])
        losses.append(outputs["loss"])
        progress.update()
        if i < num_batch and i % print_interval < 1:
            fabric.print(f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}")
    fabric.barrier()
    all_preds: torch.Tensor = fabric.all_gather(torch.tensor(preds)).flatten()
    all_labels: torch.Tensor = fabric.all_gather(torch.tensor(labels)).flatten()
    metrics: Mapping[str, Any] = {
        "step": round(
            fabric.all_gather(torch.tensor(model.args.prog.global_step * 1.0))
            .mean()
            .item()
        ),
        "epoch": round(
            fabric.all_gather(torch.tensor(model.args.prog.global_epoch)).mean().item(),
            4,
        ),
        "val_loss": fabric.all_gather(torch.stack(losses)).mean().item(),
        "val_acc": accuracy(all_preds, all_labels, ignore_index=0).item(),
        "val_F1c": NER_Char_MacroF1.all_in_one(
            all_preds, all_labels, label_info=model.labels
        ),
        "val_F1e": NER_Entity_MacroF1.all_in_one(
            all_preds, all_labels, label_info=model.labels
        ),
    }
    fabric.log_dict(metrics=metrics, step=metrics["step"])
    fabric.print(
        f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}"
        f" | {model.args.printing.tag_format_on_validate.format(**metrics)}"
    )
    fabric_barrier(fabric, "[after-check]")
    if checkpoint_saver:
        checkpoint_saver.save_checkpoint(
            metrics=metrics, ckpt_state=model.to_checkpoint()
        )


@torch.no_grad()
def test_loop(
    model: NERModel,
    dataloader: DataLoader,
    checkpoint_path: str | Path | None = None,
):
    """
    NER 테스트 루프 - 최종 테스트 데이터에 대한 성능 평가

    NER 특화 메트릭:
    - test_acc: 토큰 레벨 정확도
    - test_F1c: 문자 레벨 Macro F1 (Character-level)
    - test_F1e: 개체 레벨 Macro F1 (Entity-level)

    Args:
        model: 평가할 NERModel
        dataloader: 테스트 데이터로더
        checkpoint_path: 로드할 체크포인트 경로 (선택사항)
    """
    if checkpoint_path:
        model.load_checkpoint_file(checkpoint_path)
    fabric = model.fabric
    fabric.barrier()
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    num_batch = len(dataloader)
    print_interval = (
        model.args.printing.print_rate_on_evaluate * num_batch - epsilon
        if model.args.printing.print_step_on_evaluate < 1
        else model.args.printing.print_step_on_evaluate
    )
    preds: List[int] = []
    labels: List[int] = []
    losses: List[torch.Tensor] = []
    progress = mute_tqdm_cls(bar_size=10, desc_size=8)(
        range(num_batch), unit=f"x{dataloader.batch_size}b", desc="testing"
    )
    for i, batch in enumerate(dataloader, start=1):
        outputs = model.test_step(batch, i)
        preds.extend(outputs["preds"])
        labels.extend(outputs["labels"])
        losses.append(outputs["loss"])
        progress.update()
        if i < num_batch and i % print_interval < 1:
            fabric.print(f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}")
    fabric.barrier()
    all_preds: torch.Tensor = fabric.all_gather(torch.tensor(preds)).flatten()
    all_labels: torch.Tensor = fabric.all_gather(torch.tensor(labels)).flatten()
    metrics: Mapping[str, Any] = {
        "step": round(
            fabric.all_gather(torch.tensor(model.args.prog.global_step * 1.0))
            .mean()
            .item()
        ),
        "epoch": round(
            fabric.all_gather(torch.tensor(model.args.prog.global_epoch)).mean().item(),
            4,
        ),
        "test_loss": fabric.all_gather(torch.stack(losses)).mean().item(),
        "test_acc": accuracy(all_preds, all_labels, ignore_index=0).item(),
        "test_F1c": NER_Char_MacroF1.all_in_one(
            all_preds, all_labels, label_info=model.labels
        ),
        "test_F1e": NER_Entity_MacroF1.all_in_one(
            all_preds, all_labels, label_info=model.labels
        ),
    }
    fabric.log_dict(metrics=metrics, step=metrics["step"])
    fabric.print(
        f"(Ep {model.args.prog.global_epoch:4.2f}) {progress}"
        f" | {model.args.printing.tag_format_on_evaluate.format(**metrics)}"
    )
    fabric_barrier(fabric, "[after-test]")


@main.command()
def train(
    # === 실행 환경 설정 ===
    verbose: int = typer.Option(
        default=2, help="출력 상세도 (0: 최소, 1: 기본, 2: 상세)"
    ),
    # env - 프로젝트 환경 설정
    project: str = typer.Option(default="DeepKNLP", help="프로젝트 이름"),
    job_name: str = typer.Option(default=None, help="작업 이름 (기본값: 모델명)"),
    job_version: int = typer.Option(default=None, help="작업 버전 (기본값: 자동 할당)"),
    debugging: bool = typer.Option(default=False, help="디버깅 모드 활성화"),
    logging_file: str = typer.Option(default="logging.out", help="로그 파일명"),
    argument_file: str = typer.Option(
        default="arguments.json", help="인수 저장 파일명"
    ),
    # data - 데이터 설정
    data_home: str = typer.Option(default="data", help="데이터 홈 디렉토리"),
    data_name: str = typer.Option(
        default="klue-ner", help="데이터셋 이름 (klue-ner, kmou-ner)"
    ),
    train_file: str = typer.Option(default="train.jsonl", help="학습 데이터 파일"),
    valid_file: str = typer.Option(default="valid.jsonl", help="검증 데이터 파일"),
    test_file: str = typer.Option(default=None, help="테스트 데이터 파일"),
    num_check: int = typer.Option(default=3, help="데이터 미리보기 개수"),
    # model - 모델 설정
    pretrained: str = typer.Option(default="klue/roberta-base", help="사전학습 모델명"),
    finetuning: str = typer.Option(default="output", help="Fine-tuning 출력 디렉토리"),
    model_name: str = typer.Option(default=None, help="모델 이름 (기본값: 자동 생성)"),
    seq_len: int = typer.Option(
        default=128, help="최대 시퀀스 길이 (64, 128, 256, 512)"
    ),
    # hardware - 하드웨어 설정
    cpu_workers: int = typer.Option(
        default=min(os.cpu_count() / 2, 10), help="CPU 워커 수"
    ),
    train_batch: int = typer.Option(default=50, help="학습 배치 크기"),
    infer_batch: int = typer.Option(default=50, help="추론 배치 크기"),
    accelerator: str = typer.Option(
        default="cuda", help="가속기 타입 (cuda, cpu, mps)"
    ),
    precision: str = typer.Option(
        default="16-mixed", help="정밀도 (32-true, bf16-mixed, 16-mixed)"
    ),
    strategy: str = typer.Option(default="ddp", help="분산 전략"),
    device: List[int] = typer.Option(default=[0], help="사용할 GPU 장치 번호들"),
    # printing - 출력 설정
    print_rate_on_training: float = typer.Option(
        default=1 / 20, help="학습 중 출력 주기 (비율)"
    ),
    print_rate_on_validate: float = typer.Option(
        default=1 / 2, help="검증 중 출력 주기 (비율)"
    ),
    print_rate_on_evaluate: float = typer.Option(
        default=1 / 2, help="평가 중 출력 주기 (비율)"
    ),
    print_step_on_training: int = typer.Option(
        default=-1, help="학습 중 출력 주기 (스텝)"
    ),
    print_step_on_validate: int = typer.Option(
        default=-1, help="검증 중 출력 주기 (스텝)"
    ),
    print_step_on_evaluate: int = typer.Option(
        default=-1, help="평가 중 출력 주기 (스텝)"
    ),
    tag_format_on_training: str = typer.Option(
        default="st={step:d}, ep={epoch:.2f}, loss={loss:06.4f}, acc={acc:06.4f}",
        help="학습 로그 형식",
    ),
    tag_format_on_validate: str = typer.Option(
        default="st={step:d}, ep={epoch:.2f}, val_loss={val_loss:06.4f}, val_acc={val_acc:06.4f}, val_F1c={val_F1c:05.2f}, val_F1e={val_F1e:05.2f}",
        help="검증 로그 형식 (NER F1 포함)",
    ),
    tag_format_on_evaluate: str = typer.Option(
        default="st={step:d}, ep={epoch:.2f}, test_loss={test_loss:06.4f}, test_acc={test_acc:06.4f}, test_F1c={test_F1c:05.2f}, test_F1e={test_F1e:05.2f}",
        help="평가 로그 형식 (NER F1 포함)",
    ),
    # learning - 학습 설정
    learning_rate: float = typer.Option(default=5e-5, help="학습률"),
    random_seed: int = typer.Option(default=7, help="랜덤 시드"),
    saving_mode: str = typer.Option(
        default="max val_F1c", help="모델 저장 기준 (NER은 F1c 기준)"
    ),
    num_saving: int = typer.Option(default=1, help="저장할 모델 개수"),
    num_epochs: int = typer.Option(default=1, help="학습 에포크 수"),
    check_rate_on_training: float = typer.Option(
        default=1 / 5, help="학습 중 검증 주기 (비율)"
    ),
    name_format_on_saving: str = typer.Option(
        default="ep={epoch:.1f}, loss={val_loss:06.4f}, acc={val_acc:06.4f}, F1c={val_F1c:05.2f}, F1e={val_F1e:05.2f}",
        help="저장 파일명 형식 (NER F1 포함)",
    ),
):
    """
    NER 모델 학습 명령어

    주요 기능:
    - BERT 계열 모델을 NER 데이터셋으로 fine-tuning
    - BIO/BILOU 태깅 스킴 지원
    - 다양한 NER 메트릭 (토큰/문자/개체 레벨) 평가
    - 분산 학습 및 혼합 정밀도 지원
    - 자동 체크포인트 저장 (문자 레벨 F1 기준)
    """
    torch.set_float32_matmul_precision("high")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.getLogger("c10d-NullHandler").setLevel(logging.INFO)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)

    pretrained = Path(pretrained)
    args = TrainerArguments(
        env=ProjectEnv(
            project=project,
            job_name=job_name if job_name else pretrained.name,
            job_version=job_version,
            debugging=debugging,
            message_level=logging.DEBUG if debugging else logging.INFO,
            message_format=(
                LoggingFormat.DEBUG_20 if debugging else LoggingFormat.CHECK_20
            ),
        ),
        data=DataOption(
            home=data_home,
            name=data_name,
            files=DataFiles(
                train=train_file,
                valid=valid_file,
                test=test_file,
            ),
            num_check=num_check,
        ),
        model=ModelOption(
            pretrained=pretrained,
            finetuning=finetuning,
            name=model_name,
            seq_len=seq_len,
        ),
        hardware=HardwareOption(
            cpu_workers=cpu_workers,
            train_batch=train_batch,
            infer_batch=infer_batch,
            accelerator=accelerator,
            precision=precision,
            strategy=strategy,
            devices=device,
        ),
        printing=PrintingOption(
            print_rate_on_training=print_rate_on_training,
            print_rate_on_validate=print_rate_on_validate,
            print_rate_on_evaluate=print_rate_on_evaluate,
            print_step_on_training=print_step_on_training,
            print_step_on_validate=print_step_on_validate,
            print_step_on_evaluate=print_step_on_evaluate,
            tag_format_on_training=tag_format_on_training,
            tag_format_on_validate=tag_format_on_validate,
            tag_format_on_evaluate=tag_format_on_evaluate,
        ),
        learning=LearningOption(
            learning_rate=learning_rate,
            random_seed=random_seed,
            saving_mode=saving_mode,
            num_saving=num_saving,
            num_epochs=num_epochs,
            check_rate_on_training=check_rate_on_training,
            name_format_on_saving=name_format_on_saving,
        ),
    )
    finetuning_home = Path(f"{finetuning}/{data_name}")
    output_name = (
        f"{args.tag}={args.env.job_name}={args.env.hostname}"
        if not args.model.name
        else args.model.name
    )
    make_dir(finetuning_home / output_name)
    args.env.job_version = (
        args.env.job_version
        if args.env.job_version
        else CSVLogger(finetuning_home, output_name).version
    )
    args.prog.tb_logger = TensorBoardLogger(
        finetuning_home, output_name, args.env.job_version
    )  # tensorboard --logdir finetuning --bind_all
    args.prog.csv_logger = CSVLogger(
        finetuning_home, output_name, args.env.job_version, flush_logs_every_n_steps=1
    )
    sleep(0.3)
    fabric = Fabric(
        loggers=[args.prog.tb_logger, args.prog.csv_logger],
        devices=(
            args.hardware.devices
            if args.hardware.accelerator in ["cuda", "gpu"]
            else (
                args.hardware.cpu_workers
                if args.hardware.accelerator == "cpu"
                else "auto"
            )
        ),
        strategy=(
            args.hardware.strategy
            if args.hardware.accelerator in ["cuda", "gpu"]
            else "auto"
        ),
        precision=(
            args.hardware.precision
            if args.hardware.accelerator in ["cuda", "gpu"]
            else None
        ),
        accelerator=args.hardware.accelerator,
    )
    fabric.launch()
    fabric.barrier()
    job_versions = fabric.all_gather(torch.tensor(args.env.job_version))
    assert (
        job_versions.min() == job_versions.max()
    ), f"Job version must be same across all processes: {job_versions.tolist()}"
    sleep(fabric.global_rank * 0.3)
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    args.env.set_logging_home(args.prog.csv_logger.log_dir)
    args.env.set_logging_file(logging_file)
    args.env.set_argument_file(argument_file)
    args.prog.world_size = fabric.world_size
    args.prog.local_rank = fabric.local_rank
    args.prog.global_rank = fabric.global_rank
    fabric.seed_everything(args.learning.random_seed)
    fabric.barrier()

    with JobTimer(
        f"python {args.env.current_file} {' '.join(args.env.command_args)}",
        rt=1,
        rb=1,
        rc="=",
        args=args if (debugging or verbose > 1) and fabric.local_rank == 0 else None,
        verbose=verbose > 0 and fabric.local_rank == 0,
        mute_warning="lightning.fabric.loggers.csv_logs",
    ):
        model = NERModel(args=args)
        optimizer = model.configure_optimizers()
        model, optimizer = fabric.setup(model, optimizer)
        fabric_barrier(fabric, "[after-model]", c="=")

        assert args.data.files.train, "No training file found"
        train_dataloader = model.train_dataloader()
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
        fabric_barrier(fabric, "[after-train_dataloader]", c="=")
        assert args.data.files.valid, "No validation file found"
        val_dataloader = model.val_dataloader()
        val_dataloader = fabric.setup_dataloaders(val_dataloader)
        fabric_barrier(fabric, "[after-val_dataloader]", c="=")
        checkpoint_saver = CheckpointSaver(
            fabric=fabric,
            output_home=model.args.env.logging_home,
            name_format=model.args.learning.name_format_on_saving,
            saving_mode=model.args.learning.saving_mode,
            num_saving=model.args.learning.num_saving,
        )
        train_loop(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            checkpoint_saver=checkpoint_saver,
        )

        if args.data.files.test:
            test_dataloader = model.test_dataloader()
            test_dataloader = fabric.setup_dataloaders(test_dataloader)
            fabric_barrier(fabric, "[after-test_dataloader]", c="=")
            test_loop(
                model=model,
                dataloader=test_dataloader,
                checkpoint_path=checkpoint_saver.best_model_path,
            )


@main.command()
def test(
    verbose: int = typer.Option(default=2),
    # env
    project: str = typer.Option(default="DeepKNLP"),
    job_name: str = typer.Option(default=None),
    job_version: int = typer.Option(default=None),
    debugging: bool = typer.Option(default=False),
    logging_file: str = typer.Option(default="logging.out"),
    argument_file: str = typer.Option(default="arguments.json"),
    # data
    data_home: str = typer.Option(default="data"),
    data_name: str = typer.Option(default="klue-ner"),  # TODO: -> kmou-ner, klue-ner
    test_file: str = typer.Option(default="test.jsonl"),
    num_check: int = typer.Option(default=3),  # TODO: -> 2
    # model
    pretrained: str = typer.Option(default="klue/roberta-base"),
    finetuning: str = typer.Option(default="output"),
    model_name: str = typer.Option(default="train=*"),
    seq_len: int = typer.Option(default=128),  # TODO: -> 64, 128, 256, 512
    # hardware
    cpu_workers: int = typer.Option(default=min(os.cpu_count() / 2, 10)),
    infer_batch: int = typer.Option(default=10),
    accelerator: str = typer.Option(default="cuda"),  # TODO: -> cuda, cpu, mps
    precision: str = typer.Option(
        default=None
    ),  # TODO: -> 32-true, bf16-mixed, 16-mixed
    strategy: str = typer.Option(default="auto"),
    device: List[int] = typer.Option(default=[0]),
    # printing
    print_rate_on_evaluate: float = typer.Option(
        default=1 / 10
    ),  # TODO: -> 1/2, 1/3, 1/5, 1/10, 1/50, 1/100
    print_step_on_evaluate: int = typer.Option(default=-1),
    tag_format_on_evaluate: str = typer.Option(
        default="st={step:d}, ep={epoch:.2f}, test_loss={test_loss:06.4f}, test_acc={test_acc:06.4f}, test_F1c={test_F1c:05.2f}, test_F1e={test_F1e:05.2f}"
    ),
):
    torch.set_float32_matmul_precision("high")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.getLogger("c10d-NullHandler").setLevel(logging.INFO)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)

    pretrained = Path(pretrained)
    args = TesterArguments(
        env=ProjectEnv(
            project=project,
            job_name=job_name if job_name else pretrained.name,
            job_version=job_version,
            debugging=debugging,
            message_level=logging.DEBUG if debugging else logging.INFO,
            message_format=(
                LoggingFormat.DEBUG_20 if debugging else LoggingFormat.CHECK_20
            ),
        ),
        data=DataOption(
            home=data_home,
            name=data_name,
            files=DataFiles(
                test=test_file,
            ),
            num_check=num_check,
        ),
        model=ModelOption(
            pretrained=pretrained,
            finetuning=finetuning,
            name=model_name,
            seq_len=seq_len,
        ),
        hardware=HardwareOption(
            cpu_workers=cpu_workers,
            infer_batch=infer_batch,
            accelerator=accelerator,
            precision=precision,
            strategy=strategy,
            devices=device,
        ),
        printing=PrintingOption(
            print_rate_on_evaluate=print_rate_on_evaluate,
            print_step_on_evaluate=print_step_on_evaluate,
            tag_format_on_evaluate=tag_format_on_evaluate,
        ),
    )
    finetuning_home = Path(f"{finetuning}/{data_name}")
    output_name = f"{args.tag}={args.env.job_name}={args.env.hostname}"
    make_dir(finetuning_home / output_name)
    args.env.job_version = (
        args.env.job_version
        if args.env.job_version
        else CSVLogger(finetuning_home, output_name).version
    )
    args.prog.csv_logger = CSVLogger(
        finetuning_home, output_name, args.env.job_version, flush_logs_every_n_steps=1
    )
    sleep(0.3)
    fabric = Fabric(
        devices=(
            args.hardware.devices
            if args.hardware.accelerator in ["cuda", "gpu"]
            else (
                args.hardware.cpu_workers
                if args.hardware.accelerator == "cpu"
                else "auto"
            )
        ),
        strategy=(
            args.hardware.strategy
            if args.hardware.accelerator in ["cuda", "gpu"]
            else "auto"
        ),
        precision=(
            args.hardware.precision
            if args.hardware.accelerator in ["cuda", "gpu"]
            else None
        ),
        accelerator=args.hardware.accelerator,
    )
    fabric.launch()
    fabric.barrier()
    job_versions = fabric.all_gather(torch.tensor(args.env.job_version))
    assert (
        job_versions.min() == job_versions.max()
    ), f"Job version must be same across all processes: {job_versions.tolist()}"
    sleep(fabric.global_rank * 0.3)
    fabric.print = logger.info if fabric.local_rank == 0 else logger.debug
    args.env.set_logging_home(args.prog.csv_logger.log_dir)
    args.env.set_logging_file(logging_file)
    args.env.set_argument_file(argument_file)
    args.prog.world_size = fabric.world_size
    args.prog.local_rank = fabric.local_rank
    args.prog.global_rank = fabric.global_rank
    fabric.barrier()

    with JobTimer(
        f"python {args.env.current_file} {' '.join(args.env.command_args)}",
        rt=1,
        rb=1,
        rc="=",
        args=args if (debugging or verbose > 1) and fabric.local_rank == 0 else None,
        verbose=verbose > 0 and fabric.local_rank == 0,
        mute_warning="lightning.fabric.loggers.csv_logs",
    ):
        model = NERModel(args=args)
        model = fabric.setup(model)
        fabric_barrier(fabric, "[after-model]", c="=")

        assert args.data.files.test, "No test file found"
        test_dataloader = model.test_dataloader()
        test_dataloader = fabric.setup_dataloaders(test_dataloader)
        fabric_barrier(fabric, "[after-test_dataloader]", c="=")

        for checkpoint_path in files(finetuning_home / args.model.name / "**/*.ckpt"):
            test_loop(
                model=model,
                dataloader=test_dataloader,
                checkpoint_path=checkpoint_path,
            )


@main.command()
def serve(
    verbose: int = typer.Option(default=2),
    # env
    project: str = typer.Option(default="DeepKNLP"),
    job_name: str = typer.Option(default=None),
    job_version: int = typer.Option(default=None),
    debugging: bool = typer.Option(default=False),
    logging_file: str = typer.Option(default="logging.out"),
    argument_file: str = typer.Option(default="arguments.json"),
    # data
    data_home: str = typer.Option(default="data"),
    data_name: str = typer.Option(default="klue-ner"),  # TODO: -> kmou-ner, klue-ner
    test_file: str = typer.Option(default="test.jsonl"),
    # model
    pretrained: str = typer.Option(default="klue/roberta-base"),
    finetuning: str = typer.Option(default="output"),
    model_name: str = typer.Option(default="train=*"),
    seq_len: int = typer.Option(default=128),  # TODO: -> 512
    # server
    server_port: int = typer.Option(default=9164),
    server_host: str = typer.Option(default="0.0.0.0"),
    server_temp: str = typer.Option(default="templates"),
    server_page: str = typer.Option(default="serve_ner.html"),
):
    torch.set_float32_matmul_precision("high")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.getLogger("c10d-NullHandler").setLevel(logging.INFO)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)

    pretrained = Path(pretrained)
    args = ServerArguments(
        env=ProjectEnv(
            project=project,
            job_name=job_name if job_name else pretrained.name,
            job_version=job_version,
            debugging=debugging,
            message_level=logging.DEBUG if debugging else logging.INFO,
            message_format=(
                LoggingFormat.DEBUG_20 if debugging else LoggingFormat.CHECK_20
            ),
        ),
        data=DataOption(
            home=data_home,
            name=data_name,
            files=DataFiles(
                test=test_file,
            ),
        ),
        model=ModelOption(
            pretrained=pretrained,
            finetuning=finetuning,
            name=model_name,
            seq_len=seq_len,
        ),
        server=ServerOption(
            port=server_port,
            host=server_host,
            temp=server_temp,
            page=server_page,
        ),
    )
    finetuning_home = Path(f"{finetuning}/{data_name}")
    output_name = f"{args.tag}={args.env.job_name}={args.env.hostname}"
    make_dir(finetuning_home / output_name)
    args.env.job_version = (
        args.env.job_version
        if args.env.job_version
        else CSVLogger(finetuning_home, output_name).version
    )
    args.prog.csv_logger = CSVLogger(
        finetuning_home, output_name, args.env.job_version, flush_logs_every_n_steps=1
    )
    fabric = Fabric(devices=1, accelerator="cpu")
    fabric.print = logger.info
    args.env.set_logging_home(args.prog.csv_logger.log_dir)
    args.env.set_logging_file(logging_file)
    args.env.set_argument_file(argument_file)

    with JobTimer(
        f"python {args.env.current_file} {' '.join(args.env.command_args)}",
        rt=1,
        rb=1,
        rc="=",
        args=args if (debugging or verbose > 1) else None,
        verbose=verbose > 0,
        mute_warning="lightning.fabric.loggers.csv_logs",
    ):
        model = NERModel(args=args)
        model = fabric.setup(model)
        model.load_last_checkpoint_file(finetuning_home / args.model.name / "**/*.ckpt")
        fabric_barrier(fabric, "[after-model]", c="=")

        model.run_server(
            server=Flask(output_name, template_folder=args.server.temp),
            host=args.server.host,
            port=args.server.port,
            debug=debugging,
        )


if __name__ == "__main__":
    main()
