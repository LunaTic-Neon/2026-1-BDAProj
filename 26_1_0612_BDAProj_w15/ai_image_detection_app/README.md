# AI 활용 이미지 판별 앱

이미지 제작 과정에 AI가 활용되었을 가능성이 있는 생성/합성·아바타 계열 이미지(FAKE)인지 실제 사진(REAL)인지 판별하는 Streamlit 웹 앱입니다.
`project_template`을 복사해 시작했으며, 이미지 데이터(CSV+URL)에 맞게 `data_loader.py`·`1_EDA.py`를 교체했습니다.
문제정의는 [`../문제정의.md`](../문제정의.md) 참고.

> 이 프로젝트는 영상 기반 탐지가 아니라, URL 기반 얼굴 이미지의 AI 활용 가능성 판별 문제에 가깝습니다.

## 실행

```bash
cd ai_image_detection_app
pip install -r requirements.txt
streamlit run app.py
```

## 구조

```
ai_image_detection_app/
├── app.py                 # 진입점 (st.navigation)
├── pages/
│   ├── 1_EDA.py          # 데이터 요약·결측·분포 (1차 작업)
│   ├── 2_시각화.py        # 인사이트 그래프 (2차 작업)
│   └── 3_모델_서비스.py   # 단일 이미지 판별 + 샘플 성능 평가
├── src/
│   ├── data_loader.py    # CSV 적재 + 이미지 URL 다운로드 (@st.cache_data)
│   ├── face_preprocess.py # 얼굴 검출·크롭 보조 유틸
│   ├── model_eval.py     # 샘플 평가 유틸
│   └── features.py       # 정제·특성 (2차 작업)
├── data/
│   └── FINAL_DATASET.csv # Kaggle Deepfake Detection Dataset 2026 (gitignore)
└── requirements.txt
```

## 데이터

- 출처: Kaggle — Deepfake Detection Dataset 2026 (6,557행 × 17열)
- 이미지는 CSV의 `image_url`로 제공되어 실행 시 다운로드합니다.
- 데이터 파일(`data/*.csv`)은 용량 문제로 git에서 제외됩니다.

## 모델·서비스 기능

- 단일 이미지 판별: 파일 업로드, URL 입력, 로컬 데모 샘플 입력을 지원합니다.
- 품질 점검: 해상도, 종횡비, 밝기, 선명도 기준으로 입력 이미지 경고를 표시합니다.
- 샘플 성능 평가: `FINAL_DATASET.csv`의 일부 샘플을 다운로드해 실제 라벨과 모델 예측을 비교합니다.
- 평가 결과: accuracy, confusion matrix, 클래스별 요약, 오분류 사례, CSV 다운로드를 제공합니다.
- Ollama 해설: 선택적으로 로컬 LLM을 사용해 모델 예측 결과를 쉬운 문장으로 설명합니다.
- 보고서 자동 반영: 평가 결과와 특징추출 결과를 `../보고서.md`의 자동 반영 구역에 업데이트할 수 있습니다.

발표용 샘플 이미지는 아래 폴더에 넣으면 앱에서 자동으로 선택할 수 있습니다.

```
data/demo_samples/
├── real/
└── fake/
```

## Ollama 선택 기능

Ollama 해설은 선택 기능입니다. Ollama가 없어도 기본 판별, 평가, EDA, 특징추출 기능은 동작합니다.

사용하려면 별도 터미널에서 아래를 실행하세요.

```bash
ollama serve
ollama pull llama3.2
```

앱의 모델·서비스 페이지에서 `Ollama 해설 사용`을 체크하면 예측 라벨, 확률, 신뢰도, 이미지 품질 경고를 바탕으로 사용자용 해설을 생성합니다.

## 보고서 자동 반영

- 모델·서비스 페이지에서 샘플 성능 평가 후 `보고서용 평가 결과 저장` → `평가 결과를 보고서에 반영`을 누릅니다.
- 시각화 페이지에서 특징 파일을 선택한 뒤 `선택한 특징 파일을 보고서에 반영`을 누릅니다.
- 자동 반영은 `../보고서.md`의 `AUTO:EVAL_RESULTS`, `AUTO:FEATURE_RESULTS` 구역만 갱신합니다.

## 진행 상태

- [x] 앱 골격 + EDA 페이지 착수 (1차 작업)
- [x] 시각화 페이지 + 전처리 안정화 (2차 작업)
- [x] 판별 서비스 MVP + 샘플 평가 기능 (3차 작업)
