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
  - 특성 추출 버튼에서 얼굴 크롭 옵션(얼굴 검출 → face_path 우선 사용)으로 `src.face_preprocess.detect_and_crop_for_df`를 호출하도록 연결됨

- `pages/2_시각화.py`
  - 시각화에서 제외할 무의미한 컬럼(`EXCLUDE_VIS_COLS`) 및 유틸 함수(`get_numeric_vis_cols`) 추가(시각화 시 자동 제외 권장)
  - 수정사항 적용: 그래프 1(볼 컬럼) 및 그래프 2에서 아래 컬럼들은 시각화 대상에서 제외됩니다:
    - image_id, image_url, label_numeric, category, source, fake_method, date_collected, version, year, domain

- `src/data_loader.py`
  - 메타 데이터 적재 (`load_data`)
  - 이미지 URL -> 디스크 캐시 저장 유틸 (`_url_to_name`, `_download_single`, `download_images_bulk`)
  - 캐시 관련: `CACHE_DIR`, `cache_size_info`, `clear_cache`, `set_cache_max_bytes`, `_ensure_cache_under_limit`
  - 이미지 단일 fetch (`fetch_image`) 구현
  - 메타 검증/정규화 유틸 추가: `normalize_label`, `validate_metadata`, `save_report_json`
  - URL 상태 검사 유틸 추가: `_url_health_check`, `validate_urls` (병렬 검사, 샘플 검사 가능)
  - DataFrame 단위 다운로드/검증 편의함수 추가: `download_images_for_df`, `download_and_validate`

- `src/image_quality.py` (신규)
  - 이미지 품질 검사 유틸 구현: 손상/열기 실패 검사, 해상도(min width/height), 선명도(대체 방식), 밝기 검사
  - 주요 함수: `filter_valid_images` (DataFrame 단위), `filter_single`
  - `download_and_validate`에서 연동되어 사용 가능

- `src/face_preprocess.py` (신규)
  - 얼굴 검출 및 크롭 유틸 구현: RetinaFace → MTCNN → OpenCV 순 폴백
  - 주요 함수: `detect_and_crop_for_df` (DataFrame을 입력으로 face_path, face_bbox, face_found 컬럼 추가)
  - 크롭 파일 저장 경로: `data/cache/crops/` (기본)

- `src/features.py`
  - 기본 이미지 특성 추출: 색상 평균/표준편차, 밝기, 선명도, mean/std pixel, 얼굴 관련(feat: face_count, face_area_ratio)
  - `extract_basic_features`, `batch_extract_features`, `save_features` 제공
  - `batch_extract_features`는 DataFrame의 `image_path` 또는 `face_path`를 사용하여 병렬 특성 추출

- `src/feature_pipeline.py`
  - 청크 단위 배치 파이프라인 스크립트(기존 제공). 메타 로드 → 청크 반복 → 이미지 다운로드 → image_path 부착 → batch_extract_features → parquet 합침

- `requirements.txt` (deepfake_app 하위)
  - 핵심: streamlit, pandas, plotly, pillow, requests, torch(옵션), numpy, tqdm, opencv-python, pyarrow 등
  - facenet-pytorch / retina-face는 optional(환경에 따라 설치)

---

## 현재까지 실제로 구현/연동된 항목 (요약)
- Streamlit EDA UI(샘플 그리드, KPI, 결측/누수 리포트, 시각화) — 동작 확인됨
- 이미지 캐시 및 병렬 다운로드 유틸(`download_images_bulk`) — 페이지에서 샘플 캐시용으로 사용 중
- 메타데이터 검증/정규화 및 URL 상태 검사 유틸(`validate_metadata`, `validate_urls`) 추가됨
- 이미지 품질 검사 모듈(`src/image_quality.py`) 생성 — DataFrame 단위 품질 리포트 제공
- 얼굴 검출·크롭 모듈(`src/face_preprocess.py`) 생성 — 특성 추출 버튼에서 호출하도록 연결됨
- 특성 추출 로직(`src/features.py -> batch_extract_features`) 및 샘플 특성 CSV 저장 연결됨
- 시각화 페이지에서 민감/무의미 컬럼 제외 로직 적용됨

---

## 현재 진행상태에서 결과물(예: `meta/valid_images.csv`, `data/features_all.parquet`)을 얻기 위해 필요한 작업
아래 항목을 순서대로 실행하면 최종 산출물을 만들 수 있습니다. 각 단계별로 필요한 파일/명령과 예상 출력물을 명시합니다.

1) 개발 환경 준비
   - 작업: 가상환경 생성 및 의존성 설치
   - 명령(예시): PowerShell에서
     - python -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r deepfake_app\requirements.txt
   - 비고: facenet-pytorch/retina-face 설치는 GPU/환경에 따라 선택적으로 진행

2) 메타데이터 정합성 검사 실행
   - 작업: `src/data_loader.validate_metadata`를 사용해 CSV 검사 및 정규화
   - 출력: `reports/data_validation_report.json`(또는 CSV), `meta/valid_metadata.csv`(필요 시)
   - 예시 코드: (파이썬 스크립트 또는 Jupyter) load_data -> validate_metadata(df, date_cols=[...], save_report='reports/validation.json')

3) 전체/청크 단위 이미지 다운로드 및 캐시 생성
   - 작업: `src.data_loader.download_images_for_df` 또는 `download_and_validate`로 청크별 다운로드
   - 입력: 메타(필터된 행 수 또는 전체)
   - 출력: 메타에 `image_path` 컬럼 추가; 캐시 파일은 `data/image_cache/`에 저장
   - 비고: 캐시 용량 제한은 `set_cache_max_bytes`로 조정

4) 이미지 품질 검사 및 valid_images 목록 생성
   - 작업: `src.image_quality.filter_valid_images`로 `image_path`를 검사
   - 출력: `meta/valid_images.csv` (iq_pass == True 인 항목), 품질 리포트(이유별 샘플)
   - 비고: 샘플 단계에서 우선 small-batch(예: 500개)로 실행해 임계값 튜닝 권장

5) 얼굴 검출 및 face-crop 생성 (선택적이지만 권장)
   - 작업: `src.face_preprocess.detect_and_crop_for_df` 실행 → `face_path` 컬럼 생성
   - 출력: `data/cache/crops/<image_id>_face.jpg` 및 메타에 face 관련 컬럼 추가
   - 비고: facenet-pytorch/retina-face가 있으면 검출 성능 향상

6) 배치 특성 추출 (청크 단위 병렬)
   - 작업: `src.features.batch_extract_features` 또는 `src/feature_pipeline.py` 실행
   - 출력: 청크별 임시 parquet 또는 CSV, 최종 `data/features_all.parquet` 또는 `data/features_all.csv`
   - 권장 실행 예시: `python -m src.feature_pipeline --out data/features_all.parquet --chunk 500 --workers 8 --limit 10000`

7) 검증 및 시각화
   - 작업: 합쳐진 features와 메타 병합 → 이상치/결측 리포트 생성 → Streamlit `pages/2_시각화.py`에서 특성 시각화
   - 출력: 시각화 가능한 대시보드, 이상치 리스트, 모델 학습 전처리 체크리스트

---

## 우선순위별 실행 플랜 (단계별 작업 항목, 소요 추정)
- 1단계 (오늘/빠르게): 의존성 설치, `load_data`로 메타 확인, `validate_metadata`(샘플) 실행 — 10~30분
- 2단계 (단기): 소규모(500~2000) 샘플에 대해 `download_and_validate` 실행 → `filter_valid_images`로 임계값 조정 — 30분~2시간
- 3단계 (중기): 얼굴 검출·크롭(선택) 및 `batch_extract_features`로 샘플 특성 추출, Streamlit에서 시각화 확인 — 1~3시간
- 4단계 (완전 배치): 전체 데이터에 대해 `feature_pipeline.py` 실행(parquet 생성) — 데이터 크기에 따라 수시간~수일

---

## 현재 작업에서 나에게 요청드릴 항목
1. 어떤 단계부터 우선 실행할지 결정해주세요: (A) 소규모 샘플 실험, (B) 전체 데이터 배치 실행, (C) 얼굴 임베딩/고급 특성 우선
2. 로컬에서 실행 가능한 권한(네트워크 접근, 디스크 여유)과 GPU 사용 가능 여부를 알려주세요.
3. 원하는 출력 포맷(Parquet vs CSV)과 저장 위치가 있으면 알려주세요.

---

문서 추가/수정이 필요하면 형식(체크리스트/스크립트/튜닝 가이드)에 맞춰 바로 반영해 드리겠습니다.
