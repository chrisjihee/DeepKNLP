# Sentence Generation

한국어 문장 생성 실습 태스크입니다.

참고 자료:
- https://ratsgo.github.io/nlpbook/docs/generation

학생 구현 대상:
- task3-gen/run_gen.py

실습 방식:
- 기존의 `train_gen-*`, `infer_gen-*`, `serve_gen-*`를 하나의 학생용 실행기로 통합했습니다.
- 파일 안의 `TODO Step 1`, `TODO Step 2`, `TODO Step 3` 표시를 따라가면 됩니다.
- 실제로 채워야 하는 핵심 블록은 실제 실행 함수 안에 in-place 형태로 비워 두었습니다.
- 모델 선택은 preset 인자로 처리합니다.

Step 1:
- 목표: 모델/토크나이저/코퍼스 로딩과 dry-run generation 확인
- 구현 포인트: `build_train_args`, `load_pretrained_components`, `prepare_generation_datasets`
- 실행 예시:
```bash
python task3-gen/run_gen.py step1 --model-preset kogpt2
```

Step 2:
- 목표: 학습과 생성 파라미터 실험 수행
- 구현 포인트: `step2`
- 실행 예시:
```bash
python task3-gen/run_gen.py step2 --model-preset kogpt2 --epochs 1
```

Step 3:
- 목표: 생성 모델을 웹으로 서빙
- 구현 포인트: `step3`
- 실행 예시:
```bash
python task3-gen/run_gen.py step3 --model-preset kogpt2 --port 9001
```

모델 preset:
- `kogpt2`
- `kogpt-trinity`
- `polyglot-ko`

해답 파일:
- 각 `stepX_solution.py`는 전체 정답 파일이 아니라, 해당 단계에서 채워야 할 in-place 블록 스니펫만 담고 있습니다.
- task3-gen/solutions/step1_solution.py
- task3-gen/solutions/step2_solution.py
- task3-gen/solutions/step3_solution.py
- task3-gen/solutions/train_gen_reference.py
- task3-gen/solutions/infer_gen_reference.py
- task3-gen/solutions/serve_gen_reference.py
