from typing import Any

import requests


OLLAMA_BASE_URL = "http://localhost:11434"


def check_ollama_status(base_url: str = OLLAMA_BASE_URL, timeout: int = 3) -> dict[str, Any]:
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        models = [item.get("name", "") for item in payload.get("models", []) if item.get("name")]
        return {"available": True, "models": models, "error": None}
    except Exception as exc:
        return {"available": False, "models": [], "error": str(exc)}


def build_explanation_prompt(
    pred_label: str,
    score: float,
    confidence: str,
    quality_warnings: list[str] | None = None,
    candidate_results: list[dict[str, Any]] | None = None,
) -> str:
    warnings = quality_warnings or []
    warning_text = "없음" if not warnings else "; ".join(warnings)
    candidate_text = "없음"
    if candidate_results:
        candidate_text = ", ".join(
            f"{row.get('정규화 라벨', row.get('label', 'UNKNOWN'))}: {float(row.get('확률', row.get('score', 0))) * 100:.1f}%"
            for row in candidate_results
        )

    return f"""
다음은 AI 활용 이미지 판별 모델의 결과입니다.

예측 라벨: {pred_label}
예측 확률: {score * 100:.1f}%
신뢰도 구간: {confidence}
이미지 품질 경고: {warning_text}
후보별 확률: {candidate_text}

프로젝트 맥락:
- 이 서비스는 이미지 제작 과정에 AI가 활용되었을 가능성을 판별하는 보조 도구입니다.
- 영상 기반 탐지나 법적 판단 도구가 아닙니다.
- URL 출처 편향, 데이터 도메인 차이, 이미지 품질에 따라 결과가 제한될 수 있습니다.

요청:
일반 사용자가 이해하기 쉽게 한국어로 2~4문장 해설을 작성해 주세요.
모델 결과를 확정적 사실처럼 말하지 말고, 실제 AI 활용 여부를 보장하지 않는다는 한계를 반드시 포함해 주세요.
""".strip()


def generate_ollama_explanation(
    prompt: str,
    model_name: str = "llama3.2",
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 60,
) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload.get("response", "").strip()
        if not text:
            return {"ok": False, "text": "", "error": "Ollama가 빈 응답을 반환했습니다."}
        return {"ok": True, "text": text, "error": None}
    except Exception as exc:
        return {"ok": False, "text": "", "error": str(exc)}