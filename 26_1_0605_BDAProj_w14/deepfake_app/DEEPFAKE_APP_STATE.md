# deepfake_app 상태 요약 (AI 에이전트용)

이 파일은 현재까지 프로젝트의 코드 변경/구현 현황을 AI 에이전트나 개발자가 빠르게 파악할 수 있도록 정리한 문서입니다.

---

## 개요
- 위치: `26_1_0605_BDAProj_w14/deepfake_app`
- 목적: CSV 기반 메타데이터 + 원격 이미지(URL)로 EDA, 시각화, 전처리/특성 추출 파이프라인을 구축
- 현재까지 구현: EDA 페이지(인터랙티브), 이미지 캐시/다운로드 유틸, 특성 추출 유틸, 배치 파이프라인 스크립트, 기본 얼굴 검출(폴백)

---

## 주요 파일 및 변경 사항

- `pages/1_EDA.py`
  - 완전 구현된 EDA UI
  - 사이드바 필터(로드 행수, 샘플 모드, 샘플 수, 그리드 열 수 등)
  - 상단 KPI 카드(총 이미지, FAKE/REAL 비율, 결측 URL 수, 캐시 정보)
  - 클래스 분포 차트, `age_group` 및 `detection_difficulty` 막대 차트
  - 결측치 리포트(다운로드 가능), 데이터 누수 컬럼 표시
  - 샘플 이미지 그리드: 썸네일 생성(센터 크롭, 고정 사이즈), 상세 보기(닫기 버튼 포함)
  - 캐시 검사(검사할 URL 개수 입력 가능), 캐시 삭제, 특성 추출 버튼(선택 샘플 대상)
  - 시각화: 수치형 특성 히스토그램(무의미 컬럼 제외 로직 적용)
  - CSS 보강: 이미지 겹침/캡션 겹침 문제 완화

- `pages/2_시각화.py`
  - 시각화에서 제외할 무의미한 컬럼(`EXCLUDE_VIS_COLS`) 및 유틸 함수(`get_numeric_vis_cols`) 추가(시각화 시 자동 제외 권장)
  - 수정사항 적용: 그래프 1(볼 컬럼) 및 그래프 2에서 아래 컬럼들은 시각화 대상에서 제외됩니다:
    - image_id, image_url, label_numeric, category, source, fake_method, date_collected, version, year, domain

- `src/data_loader.py`
  - 메타 데이터 적재 (`load_data`)
  - 이미지 URL -> 디스크 캐시 저장 유틸 (`_url_to_name`, `_download_single`, `download_images_bulk`)
  - 캐시 관련: `CACHE_DIR`, `cache_size_info`, `clear_cache`, `set_cache_max_bytes`, `_ensure_cache_under_limit`
  - 이미지 단일 fetch (`fetch_image`) 구현
  - (추가 예정) validate/quality 검사 유틸을 문서화해 둠

- `src/features.py`
  - 기본 이미지 특성 추출: 색상 평균/표준편차, 밝기, 선명도, mean/std pixel
  - 얼굴 검출: 우선 MTCNN(if facenet-pytorch 설치) → OpenCV Haar cascade 폴백
  - `extract_basic_features`, `batch_extract_features`, `save_features` 제공
  - RetinaFace/MTCNN 사용 시 자동 폴백 로직 포함(있는 경우만 사용)

- `src/feature_pipeline.py`
  - 청크 단위 배치 파이프라인 스크립트
  - 동작: 메타 로드 → 청크 반복 → 이미지 다운로드(청크) → image_path 부착 → batch_extract_features → 임시 parquet 저장 → 최종 parquet 합침
  - 실행 예시: `python -m src.feature_pipeline --out data/features_all.parquet --chunk 500 --workers 8 --limit 1000`

- `src/features_config.py`
  - 얼굴 검출/특성 추출 설정(예: USE_GPU, MIN_FACE_SIZE, MTCNN_THRESHOLDS, RETINAFACE_THRESHOLD)

- `requirements.txt` (deepfake_app 하위)
  - 핵심: streamlit, pandas, plotly, pillow, requests, torch(옵션), numpy, tqdm, opencv-python, pyarrow 등
  - facenet-pytorch / retina-face는 optional(환경에 따라 설치)

---

## 실행 요령 (간단)
1. 가상환경 생성 및 활성화 (PowerShell 예):
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
2. 패키지 설치:
   pip install -r deepfake_app\requirements.txt
   (facenet-pytorch/retina-face는 필요 시 별도 설치)
3. 앱 실행:
   cd deepfake_app
   .\.venv\Scripts\Activate.ps1; streamlit run app.py

---

## 알려진 제한/주의사항
- facenet-pytorch, retina-face, torch 등 무거운 라이브러리는 별도 설치 필요. 설치 없이도 폴백 로직으로 동작하지만 얼굴 검출 성능은 낮을 수 있음.
- 원격 이미지 다운로드는 느릴 수 있음 — 샘플 수 및 병렬 워커 수를 조정하세요.
- 일부 파일 생성/편집은 워크스페이스 제약으로 자동 반영이 어려운 경우가 있었음. 변경 사항은 위 목록을 기준으로 코드에 반영되어 있으니, 문제가 있으면 알려주세요.

---

## 전처리 / 특성 추출 진행 계획 (간단)
아래 계획은 우선순위별 단계와 예상 산출물을 제시합니다. 각 단계는 `src/` 아래에 유틸/스크립트를 구현하여 파이프라인으로 연결합니다.

1) 데이터 정합성 검사 (src/data_validation.py)
   - 수행 내용: 결측/중복/이상치 표준 검사, 타입 검사, 날짜 파싱
   - 산출물: `reports/data_validation_report.html` 또는 CSV, `meta/valid_metadata.csv`

2) 이미지 품질 검사 (src/image_quality.py)
   - 수행 내용: 원격 이미지 다운로드 후 손상 여부 검사, 최소 해상도/비율, 파일 크기 필터링
   - 산출물: `meta/valid_images.csv` (정상 이미지만), 로그/샘플 이미지

3) 얼굴 검출 및 정렬 (src/face_preprocess.py)
   - 수행 내용: MTCNN/RetinaFace(가능한 경우) 사용하여 얼굴 검출 → 얼굴 중심 정렬 → face-crop 저장
   - 출력: `cache/crops/<image_id>_face.jpg` 및 metadata에 `face_path`, `face_bbox`, `face_confidence` 컬럼 추가

4) 기본/고급 특성 추출 (src/features.py 확장)
   - 기본: RGB 평균/표준, 밝기, 선명도(variance of Laplacian), 색상 히스토그램
   - 고급: HSV 히스토그램, edge density(Canny), texture(GLCM), embedding(FaceNet/ResNet 옵션)
   - 병렬 배치: `src/feature_pipeline.py`로 청크 단위 병렬 처리 → Parquet 저장
   - 출력: `data/features_*.parquet`, 최종 `data/features_all.parquet`

5) 검증 및 시각화 (pages/2_시각화.py 업데이트)
   - 수행 내용: 추출된 특성 분포 시각화, 클래스별 비교, 이상치 샘플 시각화
   - 비고: 시각화에서 제외할 컬럼(그래프1, 그래프2 제외 리스트)은 아래와 같음
     - image_id, image_url, label_numeric, category, source, fake_method, date_collected, version, year, domain

6) 배포 가능한 파이프라인
   - 스크립트: `scripts/run_full_pipeline.py` (가상환경에서 실행 가능하도록 CLI 제공)
   - 옵션: 청크 사이즈, worker 수, 검증/덮어쓰기 옵션

7) 문서화 및 테스트
   - 문서: README에 실행 예시, 링크된 리포트
   - 유닛 테스트: 특성 추출/이미지 검사에 대해 일부 테스트 케이스 작성

---

## 다음 권장 작업(우선순위)
1. 데이터 정합성 검사(결측/중복/비정상) 자동화 및 리포트 생성
2. 이미지 품질 검사(해상도, 파일 크기, 손상 여부) 및 `valid_images.csv` 생성
3. 얼굴 중심 전처리: 검출 → 정렬 → face-crop 저장(모델 입력용)
4. 기본 특성 확장: HSV 히스토그램, edge density, texture features
5. 전체 데이터 배치 실행: `src/feature_pipeline.py`로 parquet 생성

---

문서가 더 필요하시면 어떤 형식(예: 체크리스트, 실행 스크립트, 튜닝 가이드)으로 정리할지 알려주세요.
