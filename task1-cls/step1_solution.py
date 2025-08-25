# === Step 1 해답 ===
# 수강생들이 완성해야 할 TODO 부분의 해답을 제공합니다.

"""
Step 1 TODO 해답:

1. NSMC 데이터 코퍼스 초기화:
   self.data: NsmcCorpus = NsmcCorpus(args)

2. 사전학습 모델 설정 로드:
   self.lm_config: PretrainedConfig = AutoConfig.from_pretrained(
       args.model.pretrained, 
       num_labels=self.data.num_labels
   )

3. 토크나이저 로드:
   self.lm_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
       args.model.pretrained,
       use_fast=True,
   )

4. 분류용 사전학습 모델 로드:
   self.lang_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
       args.model.pretrained,
       config=self.lm_config,
   )

핵심 개념:
- AutoConfig: 모델 설정 로드, num_labels 파라미터로 분류 클래스 수 지정
- AutoTokenizer: 텍스트를 토큰으로 변환하는 도구, use_fast=True로 성능 최적화
- AutoModelForSequenceClassification: 분류 태스크용 사전학습 모델
- 모든 구성요소가 같은 pretrained 모델명을 사용하여 일관성 유지
"""
