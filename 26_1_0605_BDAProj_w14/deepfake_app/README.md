# 딥페이크 이미지 판별 앱

이미지가 AI 합성(FAKE)인지 실제 사진(REAL)인지 판별하는 Streamlit 웹 앱입니다.
`project_template`을 복사해 시작했으며, 이미지 데이터(CSV+URL)에 맞게 `data_loader.py`·`1_EDA.py`를 교체했습니다.
문제정의는 [`../문제정의.md`](../문제정의.md) 참고.

## 실행

```bash
cd deepfake_app
pip install -r requirements.txt
streamlit run app.py
```

## 구조

```
deepfake_app/
├── app.py                 # 진입점 (st.navigation)
├── pages/
│   ├── 1_EDA.py          # 데이터 요약·결측·분포 (1차 작업)
│   ├── 2_시각화.py        # 인사이트 그래프 (2차 작업)
│   └── 3_모델_서비스.py   # 이미지 입력 → ViT 예측 → 결과 (3차 작업)
├── src/
│   ├── data_loader.py    # CSV 적재 + 이미지 URL 다운로드 (@st.cache_data)
│   └── features.py       # 정제·특성 (2차 작업)
├── data/
│   └── FINAL_DATASET.csv # Kaggle Deepfake Detection Dataset 2026 (gitignore)
└── requirements.txt
```

## 데이터

- 출처: Kaggle — Deepfake Detection Dataset 2026 (6,557행 × 17열)
- 이미지는 CSV의 `image_url`로 제공되어 실행 시 다운로드합니다.
- 데이터 파일(`data/*.csv`)은 용량 문제로 git에서 제외됩니다.

## 진행 상태

- [x] 앱 골격 + EDA 페이지 착수 (1차 작업)
- [ ] 시각화 페이지 + 전처리 (2차 작업)
- [ ] 판별 서비스 MVP (3차 작업)
