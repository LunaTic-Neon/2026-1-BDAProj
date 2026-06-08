import json
from pathlib import Path
from typing import Optional

import pandas as pd


def find_project_report_path(app_dir: Optional[Path] = None) -> Path:
    base = app_dir or Path(__file__).resolve().parents[1]
    return base.parent / "보고서.md"


def ensure_report_markers(report_path: Path) -> None:
    if not report_path.exists():
        raise FileNotFoundError(f"보고서 파일을 찾을 수 없습니다: {report_path}")

    text = report_path.read_text(encoding="utf-8")
    feature_marker = "<!-- AUTO:FEATURE_RESULTS_START -->"
    eval_marker = "<!-- AUTO:EVAL_RESULTS_START -->"
    model_marker = "<!-- AUTO:MODEL_COMPARISON_START -->"
    llm_marker = "<!-- AUTO:LLM_EXPLANATION_START -->"

    if feature_marker not in text:
        target = "## 4. 모델 / 서비스"
        block = (
            "\n\n<!-- AUTO:FEATURE_RESULTS_START -->\n"
            "특징추출 결과가 아직 자동 반영되지 않았습니다.\n"
            "<!-- AUTO:FEATURE_RESULTS_END -->\n\n"
        )
        text = text.replace(target, block + target, 1) if target in text else text + block

    if eval_marker not in text:
        target = "## 5. Streamlit 앱"
        block = (
            "\n\n<!-- AUTO:EVAL_RESULTS_START -->\n"
            "평가 결과가 아직 자동 반영되지 않았습니다.\n"
            "<!-- AUTO:EVAL_RESULTS_END -->\n\n"
        )
        text = text.replace(target, block + target, 1) if target in text else text + block

    if model_marker not in text:
        target = "## 5. Streamlit 앱"
        block = (
            "\n\n<!-- AUTO:MODEL_COMPARISON_START -->\n"
            "경량 모델 비교 결과가 아직 자동 반영되지 않았습니다.\n"
            "<!-- AUTO:MODEL_COMPARISON_END -->\n\n"
        )
        text = text.replace(target, block + target, 1) if target in text else text + block

    if llm_marker not in text:
        target = "## 6. 결론 & 한계"
        block = (
            "\n\n<!-- AUTO:LLM_EXPLANATION_START -->\n"
            "LLM 해설 예시가 아직 자동 반영되지 않았습니다.\n"
            "<!-- AUTO:LLM_EXPLANATION_END -->\n\n"
        )
        text = text.replace(target, block + target, 1) if target in text else text + block

    report_path.write_text(text, encoding="utf-8")


def update_marked_section(report_path: Path, marker_name: str, markdown_content: str) -> Path:
    ensure_report_markers(report_path)
    text = report_path.read_text(encoding="utf-8")
    start = f"<!-- AUTO:{marker_name}_START -->"
    end = f"<!-- AUTO:{marker_name}_END -->"
    if start not in text or end not in text:
        raise ValueError(f"보고서에 {marker_name} 자동 반영 마커가 없습니다.")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    new_text = f"{before}{start}\n{markdown_content.strip()}\n{end}{after}"
    report_path.write_text(new_text, encoding="utf-8")
    return report_path


def _markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "표시할 데이터가 없습니다."
    shown = df.head(max_rows).reset_index()
    headers = [str(col) for col in shown.columns]
    rows = shown.map(lambda value: "" if pd.isna(value) else str(value)).values.tolist()
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + body_lines)


def summarize_eval_results(eval_csv_path: Path, eval_summary_path: Optional[Path] = None) -> str:
    eval_df = pd.read_csv(eval_csv_path)
    success_df = eval_df[eval_df["error"].isna() & eval_df["pred_label"].notna()].copy()
    attempted = len(eval_df)
    success = len(success_df)
    failed = attempted - success
    accuracy = success_df["is_correct"].mean() if success else None
    avg_score = success_df["score"].mean() if success and "score" in success_df.columns else None
    wrong_df = success_df[success_df["is_correct"] == False]

    confusion = pd.crosstab(success_df["true_label"], success_df["pred_label"]) if success else pd.DataFrame()
    summary_payload = {}
    if eval_summary_path and eval_summary_path.exists():
        summary_payload = json.loads(eval_summary_path.read_text(encoding="utf-8"))

    lines = [
        "### 샘플 성능 평가 자동 요약",
        "",
        f"- 평가 결과 파일: `{eval_csv_path.name}`",
        f"- 평가 시도 수: **{attempted:,}**",
        f"- 성공 평가 수: **{success:,}**",
        f"- 실패 수: **{failed:,}**",
        f"- 정확도: **{'-' if accuracy is None else f'{accuracy * 100:.1f}%'}**",
        f"- 평균 예측 확률: **{'-' if avg_score is None else f'{avg_score * 100:.1f}%'}**",
        f"- 오분류 수: **{len(wrong_df):,}**",
        "",
    ]
    if summary_payload:
        lines.append(f"- 저장된 요약 상태: `{summary_payload.get('status', 'saved')}`")
        lines.append("")
    lines.extend(["#### Confusion Matrix", "", _markdown_table(confusion), ""])
    if len(wrong_df):
        cols = [c for c in ["image_id", "true_label", "pred_label", "raw_pred_label", "score"] if c in wrong_df.columns]
        lines.extend(["#### 오분류 사례 일부", "", _markdown_table(wrong_df[cols].head(10)), ""])
    lines.append("> 이 평가는 전체 데이터셋 성능을 대표하지 않으며, URL 다운로드 성공 여부와 샘플 수에 영향을 받습니다.")
    return "\n".join(lines)


def summarize_feature_results(feature_csv_path: Path, feature_summary_path: Optional[Path] = None) -> str:
    feat_df = pd.read_csv(feature_csv_path)
    numeric_cols = [c for c in ["brightness", "sharpness", "face_area_ratio", "mean_pixel", "std_pixel", "avg_r", "avg_g", "avg_b"] if c in feat_df.columns]
    stats = feat_df[numeric_cols].agg(["mean", "std", "min", "max"]).round(3).T if numeric_cols else pd.DataFrame()
    label_stats = pd.DataFrame()
    if "label" in feat_df.columns and numeric_cols:
        label_stats = feat_df.groupby("label")[numeric_cols].mean().round(3)

    quality_pass = int(feat_df["iq_pass"].fillna(False).sum()) if "iq_pass" in feat_df.columns else None
    face_found = int(feat_df["face_found"].fillna(False).sum()) if "face_found" in feat_df.columns else None
    summary_payload = {}
    if feature_summary_path and feature_summary_path.exists():
        summary_payload = json.loads(feature_summary_path.read_text(encoding="utf-8"))

    lines = [
        "### 전처리·특징추출 자동 요약",
        "",
        f"- 특징 파일: `{feature_csv_path.name}`",
        f"- 특징 행 수: **{len(feat_df):,}**",
    ]
    if quality_pass is not None:
        lines.append(f"- 품질 검사 통과 행 수: **{quality_pass:,}**")
    if face_found is not None:
        lines.append(f"- 얼굴 검출 성공 행 수: **{face_found:,}**")
    if summary_payload:
        lines.append(f"- 파이프라인 상태: `{summary_payload.get('status', 'saved')}`")
    lines.extend(["", "#### 주요 이미지 특징 통계", "", _markdown_table(stats), ""])
    if not label_stats.empty:
        lines.extend(["#### 라벨별 평균 특징", "", _markdown_table(label_stats), ""])
    lines.append("> 도메인, source, fake_method 등 누수 가능 정보는 모델 입력이 아니라 편향 분석과 보고서 설명용으로만 사용합니다.")
    return "\n".join(lines)


def sync_eval_to_report(eval_csv_path: Path, report_path: Optional[Path] = None, eval_summary_path: Optional[Path] = None) -> Path:
    report = report_path or find_project_report_path()
    content = summarize_eval_results(eval_csv_path, eval_summary_path)
    return update_marked_section(report, "EVAL_RESULTS", content)


def sync_features_to_report(feature_csv_path: Path, report_path: Optional[Path] = None, feature_summary_path: Optional[Path] = None) -> Path:
    report = report_path or find_project_report_path()
    content = summarize_feature_results(feature_csv_path, feature_summary_path)
    return update_marked_section(report, "FEATURE_RESULTS", content)


def summarize_lightweight_model(model_summary_path: Path) -> str:
    payload = json.loads(model_summary_path.read_text(encoding="utf-8"))
    per_class = pd.DataFrame(payload.get("per_class", {})).T
    confusion = pd.DataFrame(
        payload.get("confusion_matrix", []),
        index=payload.get("labels", []),
        columns=payload.get("labels", []),
    )
    lines = [
        "### 경량 모델 학습·비교 자동 요약",
        "",
        f"- 모델 유형: `{payload.get('model_type', '-')}`",
        f"- 입력 특징 파일: `{Path(payload.get('feature_path', '-')).name}`",
        f"- 사용 특징 수: **{payload.get('feature_count', '-')}**",
        f"- 학습 행 수: **{payload.get('train_rows', '-')}**",
        f"- 평가 행 수: **{payload.get('test_rows', '-')}**",
        f"- 정확도: **{payload.get('accuracy', 0) * 100:.1f}%**",
        "",
        "#### 클래스별 성능",
        "",
        _markdown_table(per_class.round(3) if not per_class.empty else per_class),
        "",
        "#### Confusion Matrix",
        "",
        _markdown_table(confusion),
        "",
        "> 이 결과는 누수 가능 메타 컬럼을 제외한 특징/임베딩 기반 경량 모델 성능입니다. 데이터 출처 편향과 샘플 수에 따라 해석이 달라질 수 있습니다.",
    ]
    return "\n".join(lines)


def sync_lightweight_model_to_report(model_summary_path: Path, report_path: Optional[Path] = None) -> Path:
    report = report_path or find_project_report_path()
    content = summarize_lightweight_model(model_summary_path)
    return update_marked_section(report, "MODEL_COMPARISON", content)


def sync_llm_explanation_to_report(explanation_text: str, report_path: Optional[Path] = None) -> Path:
    report = report_path or find_project_report_path()
    content = (
        "### Ollama 기반 LLM 해설 예시\n\n"
        f"> {explanation_text.strip()}\n\n"
        "이 해설은 모델 예측 결과를 사용자가 이해하기 쉽게 풀어쓴 참고 문장입니다. "
        "LLM이 직접 이미지를 판별한 것은 아니며, 최종 보고서 제출 전 사람이 해석을 검토해야 합니다."
    )
    return update_marked_section(report, "LLM_EXPLANATION", content)