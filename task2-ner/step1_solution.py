# === Step 1 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 1 TODO 해답:

1. NER 데이터 코퍼스 초기화:
   self.data: NERCorpus = NERCorpus(args)

2. 라벨 정보 및 매핑 딕셔너리 초기화:
   self.labels: List[str] = self.data.labels
   self._label_to_id: Dict[str, int] = {label: i for i, label in enumerate(self.labels)}
   self._id_to_label: Dict[int, str] = {i: label for i, label in enumerate(self.labels)}

3. 토큰 분류용 사전학습 모델 설정 로드:
   self.lm_config: PretrainedConfig = AutoConfig.from_pretrained(
       args.model.pretrained,
       num_labels=self.data.num_labels,
   )

4. Fast 토크나이저 로드:
   self.lm_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
       args.model.pretrained,
       use_fast=True,
   )

5. 토큰 분류용 사전학습 모델 로드:
   self.lang_model: PreTrainedModel = (
       AutoModelForTokenClassification.from_pretrained(
           args.model.pretrained,
           config=self.lm_config,
       )
   )

핵심 개념:
- NER vs 분류: 토큰 단위 예측 vs 문장 단위 예측
- AutoModelForTokenClassification: 각 토큰에 대해 개체명 라벨 예측
- PreTrainedTokenizerFast: token_to_chars() 메소드로 문자 오프셋 정보 제공
- BIO 태깅: B(Begin), I(Inside), O(Outside) 형태의 라벨 체계
- 라벨 매핑: 문자열 라벨과 숫자 ID 간의 양방향 변환 딕셔너리
- Fast 토크나이저 필수: NER에서는 토큰-문자 매핑이 정확한 평가에 핵심
"""
