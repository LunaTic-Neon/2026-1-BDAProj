# AI 활용 이미지 판별 앱

이미지 제작 과정에 AI가 활용되었을 가능성이 있는 얼굴 이미지를 판별하는 Streamlit 앱입니다.
이 프로젝트는 **EDA, 시각화, 특성추출, 이미지 판별, 샘플 평가** 중심으로 구성했습니다.

## 실행

```bash
cd ai_image_detection_app
pip install -r requirements.txt
streamlit run app.py
```

학교 PC에서는 GitHub에서 프로젝트를 클론한 뒤 위 명령만 실행하면 됩니다.
앱은 `models/ai_image_detector.pt`가 있으면 이 파인튜닝 모델을 우선 사용합니다.

## 구조

```text
ai_image_detection_app/
├── app.py
├── pages/
│   ├── 1_EDA.py
│   ├── 2_시각화.py
│   └── 3_모델_서비스.py
├── src/
│   ├── data_loader.py
│   ├── finetuned_resnet.py
│   ├── image_quality.py
│   ├── model_eval.py
│   └── ui_components.py
├── models/
│   └── ai_image_detector.pt
└── requirements.txt
```

## 핵심 기능

- **EDA**: 라벨 분포, 결측치, 누수 가능 컬럼, 샘플 이미지 확인
- **시각화**: 데이터 편향과 품질 분포 확인
- **모델·서비스**: 파인튜닝 MobileNetV3 Small 기반 FAKE/REAL 판별, 특성추출 시연, 샘플 평가

## 모델 구조

- 사전학습 모델: `torchvision` MobileNetV3 Small
- 학습 방식: backbone은 고정하고 마지막 분류층만 현재 데이터 이미지로 추가학습
- 저장 모델: `models/ai_image_detector.pt`
- 평가 결과: Accuracy 81.5%, FAKE F1 0.833, REAL F1 0.793

이 방식은 큰 모델 전체를 새로 학습하지 않아 학교 PC에서도 실행 부담이 작고, 과하게 데이터셋에 맞춘 모델보다 임의 이미지에서 덜 확신하도록 구성했습니다.

## 한계

- 데이터 자체가 REAL은 Unsplash, FAKE는 생성/프로필 이미지 출처에 강하게 묶여 있어 판별이 쉬운 편입니다.
- `source`, `fake_method`, `domain` 같은 메타정보는 라벨과 직접 연결되므로 모델 입력에 쓰지 않았습니다.
- 메타데이터만으로 얻을 수 있는 시각화 인사이트는 제한적이며, 서비스의 핵심은 이미지 판별입니다.
- 일반적인 AI 이미지 전체를 판별하는 모델은 아니며, 현재 데이터셋 분포와 다른 이미지에서는 성능이 낮아질 수 있습니다.

## 발표/보고서 방향

- 데이터 누수 가능성을 명확히 설명
- 복잡한 기능보다 해석 가능한 핵심 기능 위주로 발표
- 결과 해석은 Accuracy, F1 score, confusion matrix 중심으로 간단히 정리
- 한계점은 숨기지 않고 데이터 편향과 URL 기반 구조를 중심으로 정리

## 보고서 캡처 추천

- EDA: 라벨별 메타 요약 그래프
- EDA: REAL/FAKE 샘플 이미지 비교
- 시각화: 성별 Unknown 편향 그래프
- 시각화: 연령대/성별 조합별 FAKE 비율 히트맵
- 모델·서비스: 특성추출 화면
- 모델·서비스: 이미지 판별 결과 화면
- 모델·서비스: 샘플 평가 지표와 Confusion Matrix
