"""ollama gemma3:4b 프롬프트 변형 실험.

같은 100건 샘플(llm_sample.head(100))에 대해 여러 프롬프트를 적용해
정확도/F1/분류 실패 수/소요 시간을 비교한다.
결과는 prompt_experiment_results.pkl 에 저장된다.
"""
import json
import pickle
import re
import time
from urllib.parse import unquote

import ollama
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

MODEL = "gemma3:4b"
N_SAMPLES = 100


def build_http_text(row) -> str:
    method = row.get("method", "GET")
    url = unquote(str(row.get("url", "")), encoding="latin-1")
    body = str(row.get("body_decoded", row.get("body", "")) or "")
    text = f"{method} {url} HTTP/1.1"
    if body and body != "nan":
        text += f"\nBody: {body[:200]}"
    return text


# ------------------------------------------------------------------
# 프롬프트 정의
# ------------------------------------------------------------------
PROMPTS = {}

# 1) Baseline (현재 노트북) — 영어, 2-shot, JSON 출력
PROMPTS["baseline_2shot"] = (
    'You are a web security expert. Classify each HTTP request as '
    '"Normal" or "Anomalous" and provide a brief reason.\n\n'
    'Examples:\n'
    'Request: GET /index.jsp HTTP/1.1\n'
    'Output: {{"label": "Normal", "reason": "Standard page request, no suspicious pattern"}}\n\n'
    "Request: GET /search?q=' OR '1'='1 HTTP/1.1\n"
    'Output: {{"label": "Anomalous", "reason": "Classic SQL Injection pattern with OR 1=1"}}\n\n'
    'Now classify:\nRequest: {http_text}\nOutput:'
)

# 2) Zero-shot — 예시 없이 강한 지시 + JSON 강제
PROMPTS["zero_shot_strict_json"] = (
    'You are a web security expert. Decide whether the following HTTP request is malicious.\n'
    'Output ONLY a single JSON object with two string keys: "label" and "reason".\n'
    '"label" must be exactly "Normal" or "Anomalous". No prose, no markdown.\n\n'
    'Request: {http_text}\nOutput:'
)

# 3) Few-shot 7개 (정상 2 + 공격 5) — 각 공격 유형 다양화
PROMPTS["few_shot_7"] = (
    'You are a web security expert. Classify each HTTP request as '
    '"Normal" or "Anomalous". Output ONLY a JSON object {{"label": "...", "reason": "..."}}.\n\n'
    'Examples:\n'
    'Request: GET /tienda1/index.jsp HTTP/1.1\n'
    'Output: {{"label": "Normal", "reason": "Standard page request"}}\n\n'
    'Request: GET /tienda1/publico/registro.jsp?modo=login&login=user1&pwd=secret HTTP/1.1\n'
    'Output: {{"label": "Normal", "reason": "Benign login form parameters"}}\n\n'
    "Request: GET /search?q=' OR '1'='1 HTTP/1.1\n"
    'Output: {{"label": "Anomalous", "reason": "SQL Injection with OR 1=1"}}\n\n'
    'Request: GET /page?id=1 UNION SELECT username,password FROM users-- HTTP/1.1\n'
    'Output: {{"label": "Anomalous", "reason": "SQL Injection UNION SELECT"}}\n\n'
    'Request: GET /search?q=<script>alert(1)</script> HTTP/1.1\n'
    'Output: {{"label": "Anomalous", "reason": "Reflected XSS payload"}}\n\n'
    'Request: GET /file?p=../../../../etc/passwd HTTP/1.1\n'
    'Output: {{"label": "Anomalous", "reason": "Path traversal to /etc/passwd"}}\n\n'
    'Request: GET /ping?host=127.0.0.1;cat%20/etc/passwd HTTP/1.1\n'
    'Output: {{"label": "Anomalous", "reason": "Command injection via shell metacharacter"}}\n\n'
    'Now classify:\nRequest: {http_text}\nOutput:'
)

# 4) Attack catalog + 2-shot — 공격 유형 카탈로그 명시 + 결정 규칙
PROMPTS["attack_catalog"] = (
    'You are a senior web application security analyst. Classify the HTTP request as '
    '"Normal" or "Anomalous".\n\n'
    'Mark Anomalous if any of these patterns appear:\n'
    "- SQL Injection: ' OR 1=1, UNION SELECT, --, /*, INSERT/UPDATE/DELETE keywords in values\n"
    '- XSS: <script>, javascript:, on<event>=, <iframe>, <img onerror=\n'
    '- Path Traversal: ../, ..%2f, /etc/passwd, /boot.ini\n'
    '- Command Injection: ; cat, | nc, && rm, /bin/sh, $(...), backticks\n'
    '- CRLF / Header injection: %0d, %0a, \\r\\n in URL\n'
    '- LDAP/NoSQL/Template injection or obvious obfuscation/encoding of payloads\n'
    'Otherwise Normal.\n\n'
    'Output ONLY a JSON object {{"label": "Normal"|"Anomalous", "reason": "..."}}.\n\n'
    'Examples:\n'
    'Request: GET /tienda1/index.jsp HTTP/1.1\n'
    'Output: {{"label": "Normal", "reason": "Standard page request"}}\n\n'
    "Request: GET /search?q=' OR '1'='1 HTTP/1.1\n"
    'Output: {{"label": "Anomalous", "reason": "SQL Injection OR 1=1"}}\n\n'
    'Now classify:\nRequest: {http_text}\nOutput:'
)


def parse_response(text: str) -> dict:
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        return {"label": "Unknown", "reason": text[:80]}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {"label": "Unknown", "reason": text[:80]}


def classify(http_text: str, template: str) -> dict:
    prompt = template.format(http_text=http_text)
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return parse_response(resp["message"]["content"])


def run_experiment(name: str, template: str, samples: pd.DataFrame) -> dict:
    print(f"\n=== [{name}] 시작 ({len(samples)}건) ===")
    rows = []
    t0 = time.time()
    for i, row in samples.iterrows():
        http_text = build_http_text(row)
        result = classify(http_text, template)
        true_label = "Anomalous" if row.get("is_attack", 0) == 1 else "Normal"
        rows.append({
            "idx": i,
            "true": true_label,
            "pred": result.get("label", "Unknown"),
            "reason": str(result.get("reason", ""))[:160],
        })
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{name}] {i+1}/{len(samples)}건  ({elapsed:.1f}s)")
    elapsed = time.time() - t0

    df = pd.DataFrame(rows)
    df["pred_clean"] = df["pred"].replace({"Unknown": "Normal"})
    y_true = (df["true"] == "Anomalous").astype(int)
    y_pred = (df["pred_clean"] == "Anomalous").astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    n_unknown = int((df["pred"] == "Unknown").sum())
    report = classification_report(
        y_true, y_pred, target_names=["Normal", "Anomalous"], digits=4
    )
    print(f"=== [{name}] 완료 acc={acc:.4f} f1={f1:.4f} 실패={n_unknown} time={elapsed:.1f}s ===")
    return {
        "name": name,
        "template": template,
        "df": df,
        "acc": acc,
        "f1": f1,
        "n_unknown": n_unknown,
        "elapsed": elapsed,
        "report": report,
    }


def main():
    with open("processed_data.pkl", "rb") as f:
        data = pickle.load(f)
    samples = data["llm_sample"].head(N_SAMPLES).reset_index(drop=True)
    print(f"샘플 {len(samples)}건 (정상 {(samples['is_attack']==0).sum()} / "
          f"공격 {(samples['is_attack']==1).sum()})")

    results = {}
    for name, tpl in PROMPTS.items():
        results[name] = run_experiment(name, tpl, samples)

    with open("prompt_experiment_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\n>>> 저장 완료: prompt_experiment_results.pkl")

    # 요약
    print("\n=== 요약 ===")
    print(f"{'프롬프트':<25s} | {'정확도':>8s} | {'F1':>8s} | {'실패':>5s} | {'시간(s)':>8s}")
    for r in results.values():
        print(
            f"{r['name']:<25s} | {r['acc']:>8.4f} | {r['f1']:>8.4f} | "
            f"{r['n_unknown']:>5d} | {r['elapsed']:>8.1f}"
        )


if __name__ == "__main__":
    main()
