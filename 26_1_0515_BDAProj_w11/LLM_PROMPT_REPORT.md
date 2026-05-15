# Ollama `gemma3:4b` 프롬프트 비교 실험 보고서

- 데이터: `processed_data.pkl` 의 `llm_sample.head(100)` — 정상 47건 / 공격 53건
- 모델: `gemma3:4b` (Ollama, `temperature=0`)
- 노트북: `llm_classification.ipynb`, 실험 스크립트: `_prompt_experiments.py`

## 결과 요약

| # | 프롬프트 | 정확도 | F1 | 분류 실패 | 시간(s) |
|---|---|---:|---:|---:|---:|
| 1 | **baseline_2shot** (노트북 기본) | **0.84** | **0.85** | 1 | 76.5 |
| 2 | zero_shot_strict_json | 0.59 | 0.69 | 3 | 91.5 |
| 3 | few_shot_7 (정상 2 + 공격 5) | 0.77 | 0.72 | 0 | 44.5 |
| 4 | attack_catalog (규칙 + 2-shot) | 0.79 | 0.77 | 1 | 48.6 |

**가장 정확한 프롬프트는 노트북 기본 2-shot (정확도 0.84).**

## 핵심 관찰

- **예시(few-shot) 개수를 늘리는 게 항상 좋지는 않음.** 7-shot 으로 다양한 공격을 보여줬더니 모델이 "교과서적인 패턴"만 공격으로 보고 CSIC 의 변형 공격은 정상으로 흘려보내 정확도가 떨어졌다.
- **예시 없는 zero-shot 은 명확히 불리** — 0.59. 형식만 강제해서는 부족하고 판단 기준 예시가 필요하다.
- **공격 카탈로그(분류 규칙)** 를 덧붙여도 baseline 보다 낮은 0.79. 규칙에 없는 변형 공격을 놓치는 경향이 있다.
- **속도**: 응답이 짧을수록 빠름. few-shot 으로 모델이 출력 형식을 학습하면 reason 이 짧아져 오히려 baseline 보다 30초 이상 단축됐다.

## 정확도를 더 올리려면

- CSIC 와 비슷한 도메인의 예시 1~2개로 2-shot 을 교체
- 예시는 정상=공격 균형, 4개 이내로 유지
- 모델이 reason 을 먼저 쓰게 해 추론을 유도 (`reason` → `label` 순서)

## 프롬프트별 본문

### 1) baseline_2shot — 정확도 0.84 (최고)

정상 1개 + SQL Injection 1개의 단순 2-shot. JSON 출력 형식을 예시로 학습시킴.

```
You are a web security expert. Classify each HTTP request as
"Normal" or "Anomalous" and provide a brief reason.

Examples:
Request: GET /index.jsp HTTP/1.1
Output: {"label": "Normal", "reason": "Standard page request, no suspicious pattern"}

Request: GET /search?q=' OR '1'='1 HTTP/1.1
Output: {"label": "Anomalous", "reason": "Classic SQL Injection pattern with OR 1=1"}

Now classify:
Request: <HTTP_REQUEST>
Output:
```

### 2) zero_shot_strict_json — 정확도 0.59

예시 없이 JSON 출력 형식만 강제. 가장 낮은 정확도.

```
You are a web security expert. Decide whether the following HTTP request is malicious.
Output ONLY a single JSON object with two string keys: "label" and "reason".
"label" must be exactly "Normal" or "Anomalous". No prose, no markdown.

Request: <HTTP_REQUEST>
Output:
```

### 3) few_shot_7 — 정확도 0.77

정상 2개 + 공격 5종(SQLi 2, XSS, Path Traversal, Cmd Injection)의 7-shot.
공격 recall 이 0.57로 떨어져 false negative 가 많았다.

### 4) attack_catalog — 정확도 0.79

공격 유형 6종(SQLi/XSS/Traversal/Cmd/CRLF/기타)의 패턴 카탈로그 + 2-shot.
명시되지 않은 변형 공격을 정상으로 분류하는 경향.

```
You are a senior web application security analyst.
Mark Anomalous if any of these patterns appear:
- SQL Injection: ' OR 1=1, UNION SELECT, --, /*, ...
- XSS: <script>, javascript:, on<event>=, <iframe>, ...
- Path Traversal: ../, ..%2f, /etc/passwd, ...
- Command Injection: ; cat, | nc, && rm, $(...), ...
- CRLF / Header injection: %0d, %0a, \r\n in URL
Otherwise Normal.

Output ONLY a JSON object {"label": "Normal"|"Anomalous", "reason": "..."}.

Examples: (정상 1개 + SQLi 1개 — 생략)

Now classify:
Request: <HTTP_REQUEST>
Output:
```
