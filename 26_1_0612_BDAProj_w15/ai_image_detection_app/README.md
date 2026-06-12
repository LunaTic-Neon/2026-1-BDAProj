# AI 활용 이미지 판별 앱

이미지 제작 과정에 AI가 활용되었을 가능성이 있는 얼굴 이미지를 판별하는 Streamlit 앱입니다.
이 프로젝트는 평가기준에 맞춰 **EDA, 시각화, 모델·서비스**의 핵심만 유지합니다.

## 실행

```bash
cd ai_image_detection_app
pip install -r requirements.txt
streamlit run app.py
```

## 구조

```
ai_image_detection_app/
├── app.py
├── pages/
│   ├── 1_EDA.py
│   ├── 2_시각화.py
│   └── 3_모델_서비스.py
├── src/
│   ├── data_loader.py
│   ├── image_quality.py
│   ├── model_eval.py
│   └── ui_components.py
└── requirements.txt
```

## 핵심 기능

- **EDA**: 라벨 분포, 결측치, 누수 가능 컬럼, 샘플 이미지 확인
- **시각화**: 데이터 편향과 품질 분포 확인
- **모델·서비스**: 이미지 업로드 후 FAKE/REAL 판별, 샘플 평가, F1 score/accuracy 기반 결과 해석

## 발표/보고서 방향

- 데이터 누수 가능성을 명확히 설명
- 복잡한 기능보다 해석 가능한 핵심 기능 위주로 발표
- 결과 해석은 F1 score, accuracy, confusion matrix 같은 수치 중심으로 간단히 정리
- 한계점은 숨기지 않고 데이터 편향과 URL 기반 구조를 중심으로 정리
