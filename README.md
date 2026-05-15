## 2026-1-BDAProj

---

### 빅데이터 분석 프로젝트 실습 저장소

2026학년도 1학기 **빅데이터분석프로그래밍** 수업의 주차별 실습 코드를 모아둔 저장소입니다.
Streamlit 기반 대시보드 구축, 공공데이터 API 활용, 로컬 LLM(Ollama) 응용,
머신러닝 기반 웹 공격 탐지(CSIC 2010) 등 한 학기 동안 다룬 실습을 주차 단위로 보관합니다.

---

### 폴더 구성

각 폴더는 `26_1_<MMDD>_BDAProj_w<N>` 규칙으로 명명되며, 해당 주차 수업일과 주차 번호를 의미합니다.

| 주차 | 폴더 | 주제 | 주요 내용 |
| --- | --- | --- | --- |
| w2 | `26_1_0313_BDAProj_w2` | Streamlit 입문 | 첫 Streamlit 앱 — 데이터프레임, 바 차트, 사이드바 위젯 기본 사용 |
| w3 | `26_1_0320_BDAProj_w3` | 공공데이터 API 수집 | 에어코리아 대기오염정보 API 호출 → JSON 파싱 → CSV 저장 |
| w4 | `26_1_0327_BDAProj_w4` | Streamlit 멀티페이지 | `pages/` 디렉터리 기반 멀티페이지 앱 (차트/지도/데이터 데모) |
| w5 | `26_1_0403_BDAProj_w5` | 따릉이 대시보드 | `st.navigation` 기반 멀티페이지 — 서울 따릉이 데이터 분석 |
| w6 | `26_1_0410_BDAProj_w6` | Ollama LLM 기초 + 영화 리뷰 분석 | `ollama` 패키지로 generate/chat/stream/multi-turn 실습, NSMC 영화 리뷰 감성 분석 대시보드 |
| w7 | `26_1_0417_BDAProj_w7` | LLM 응용 | 감성 분석, JSON 구조화 출력, 텍스트 분석, 프롬프트 비교, Streamlit 챗봇 풀버전 |
| w10 | `26_1_0508_BDAProj_w10` | 웹 공격 탐지 — 데이터 준비/EDA | Kaggle CSIC 2010 HTTP 데이터셋 정제, 공격 유형(SQLi/XSS/Path Traversal/CMD Injection) 퀴즈, EDA 노트북 |
| w11 | `26_1_0515_BDAProj_w11` | 웹 공격 탐지 — 전처리/특성 + LLM 분류 | CSIC 2010 전처리·특성 엔지니어링 노트북(23개 숫자형 특성, 80:20 분할, StandardScaler), Ollama `gemma3:4b` HTTP 요청 분류 + 프롬프트 4종(zero-shot / 2-shot / 7-shot / attack catalog) 비교 실험 (`LLM_PROMPT_REPORT.md`) |

---

### 기술 스택

- **언어**: Python 3
- **웹 대시보드**: Streamlit, Plotly, Altair
- **데이터 처리**: pandas, numpy
- **공공데이터/외부 API**: `requests`, `python-dotenv`
- **LLM**: Ollama (`gemma3:4b` 등 로컬 모델)
- **머신러닝/NLP**: HuggingFace `datasets`, scikit-learn 계열

전체 의존성은 루트의 `requirements.txt` 참고.

---

### 실행 방법

1. 가상환경 생성 및 활성화 (Windows PowerShell)

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. 의존성 설치

   ```powershell
   pip install -r requirements.txt
   ```

3. 주차별 폴더로 이동하여 실습 실행

   ```powershell
   # 예: w5 따릉이 대시보드
   cd 26_1_0403_BDAProj_w5
   streamlit run app.py

   # 예: w3 공공데이터 수집 (.env 파일에 API_KEY 필요)
   cd 26_1_0320_BDAProj_w3
   python data_collection.py
   ```

   > Ollama 기반 실습(w6, w7)은 로컬에 [Ollama](https://ollama.com)가 설치되어 있고
   > 해당 모델(`gemma3:4b` 등)이 `ollama pull`로 받아져 있어야 합니다.

---

### 참고

- `venv/`, `__pycache__/`, `.env`, `data/`, 압축 파일 등은 `.gitignore`로 제외됩니다.
- 외부 API 키(`API_KEY` 등)는 각 주차 폴더의 `.env`에 보관하며 저장소에는 포함하지 않습니다.
