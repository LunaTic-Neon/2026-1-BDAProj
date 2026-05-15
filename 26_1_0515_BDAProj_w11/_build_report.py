"""prompt_experiment_results.pkl + llm_classification_results.pkl 을 읽어
LLM_PROMPT_REPORT.md 를 생성한다."""
import pickle
import os
import textwrap

WORK_DIR = os.path.dirname(os.path.abspath(__file__))


def load_results():
    with open(os.path.join(WORK_DIR, "prompt_experiment_results.pkl"), "rb") as f:
        exp = pickle.load(f)
    nb_pkl = os.path.join(WORK_DIR, "llm_classification_results.pkl")
    nb = None
    if os.path.exists(nb_pkl):
        with open(nb_pkl, "rb") as f:
            nb = pickle.load(f)
    return exp, nb


def md_fence(code: str) -> str:
    return "```\n" + code.rstrip() + "\n```"


def main():
    exp, nb = load_results()

    # 순서: baseline → 나머지
    order = ["baseline_2shot", "zero_shot_strict_json", "few_shot_7", "attack_catalog"]
    rows = [exp[k] for k in order if k in exp]
    # 누락된 항목 보호
    for k, v in exp.items():
        if k not in order:
            rows.append(v)

    best = max(rows, key=lambda r: (r["acc"], r["f1"], -r["elapsed"]))

    # 표
    table = [
        "| # | 프롬프트 이름 | 정확도 | F1 | 분류 실패 | 소요(초) | 건당(초) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(rows, 1):
        table.append(
            f"| {i} | `{r['name']}` | {r['acc']:.4f} | {r['f1']:.4f} | "
            f"{r['n_unknown']} | {r['elapsed']:.1f} | {r['elapsed']/100:.2f} |"
        )
    table_md = "\n".join(table)

    # 노트북 baseline 비교
    nb_block = ""
    if nb is not None:
        nb_block = textwrap.dedent(f"""
            ### 노트북(`llm_classification.ipynb`) baseline 재측정 결과
            - 정확도: **{nb['llm_acc']:.4f}**
            - F1: **{nb['llm_f1']:.4f}**
            - 소요 시간: **{nb['llm_time']:.1f}초** (건당 {nb['llm_time']/nb['n_samples']:.2f}초)
            - 샘플 수: {nb['n_samples']}건

            노트북은 스크립트 baseline과 동일한 2-shot 프롬프트를 사용했으며,
            동일한 100건 샘플과 `temperature=0`을 사용하므로 결과가 사실상 일치합니다.
        """).strip()

    sections = []
    for r in rows:
        prompt_preview = r["template"].format(http_text="<HTTP_REQUEST>")
        sections.append(
            f"### `{r['name']}`\n\n"
            f"- 정확도: **{r['acc']:.4f}**\n"
            f"- F1: **{r['f1']:.4f}**\n"
            f"- 분류 실패(Unknown): {r['n_unknown']}건\n"
            f"- 소요 시간: {r['elapsed']:.1f}초 (건당 {r['elapsed']/100:.2f}초)\n\n"
            f"**프롬프트 본문**\n\n{md_fence(prompt_preview)}\n\n"
            f"**분류 리포트**\n\n{md_fence(r['report'])}\n"
        )

    md = textwrap.dedent(f"""
        # Ollama `gemma3:4b` 프롬프트 비교 실험 보고서

        - 데이터: `processed_data.pkl` 의 `llm_sample.head(100)` (정상/공격 혼합 100건)
        - 모델: `gemma3:4b` (Ollama, `temperature=0`)
        - 측정 노트북: `llm_classification.ipynb`
        - 실험 스크립트: `_prompt_experiments.py`

        ## 1. 종합 비교

        {table_md}

        - **최고 정확도 프롬프트**: `{best['name']}` — 정확도 **{best['acc']:.4f}**, F1 **{best['f1']:.4f}**
        - 모든 프롬프트는 같은 100건, 같은 모델, `temperature=0` 으로 측정됨

        {nb_block}

        ## 2. 프롬프트별 상세 결과

        """).strip()

    md += "\n\n" + "\n".join(sections)

    out_path = os.path.join(WORK_DIR, "LLM_PROMPT_REPORT.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f">>> 작성 완료: {out_path}")


if __name__ == "__main__":
    main()
