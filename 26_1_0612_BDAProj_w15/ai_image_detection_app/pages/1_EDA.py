# pages/1_EDA.py — 이미지 EDA (CSV+URL 데이터용)
# 완전 구현: UI(사이드바, KPI, 탭), 샘플 이미지 그리드, 결측/캐시 리포트, 캐시 관리
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.express as px
import pandas as pd
from pathlib import Path
from io import BytesIO
import base64
import os
import time
from datetime import datetime
from src.features import batch_extract_features, save_features, url_domain
from src.image_quality import filter_valid_images
from src.data_loader import (
    load_data,
    data_missing_message,
    fetch_image,
    download_images_bulk,
    CACHE_DIR,
    cache_size_info,
    clear_cache,
    LEAKAGE_COLS,
    set_cache_max_bytes,
)
# optional face_preprocess import: 모듈이 없으면 None으로 폴백
try:
    from src.face_preprocess import detect_and_crop_for_df
except Exception:
    detect_and_crop_for_df = None

import numpy as np

st.title("📊 이미지 EDA — AI 활용 이미지 판별")

# thumbnail 기본 크기 (페이지 전체에서 재사용)
THUMB_W = 320
THUMB_H = 220

# ------------------------ 사이드바: 필터 / 옵션 ---------------------------
st.sidebar.header("데이터 및 표시 옵션")
rows_opt = st.sidebar.selectbox("로드할 메타 데이터 행 수", options=[None, 100, 500, 2000], index=0, format_func=lambda x: "전체" if x is None else str(x))
sample_mode = st.sidebar.radio("샘플 방식", ("클래스별(기본)", "전체에서 랜덤"))
sample_per_class = st.sidebar.slider("클래스당/전체 샘플 수", 1, 24, 6)
grid_cols = st.sidebar.slider("이미지 그리드 열수", 1, 6, 4)

st.sidebar.markdown("---")
st.sidebar.header("캐시 관리")
cache_info_btn = st.sidebar.button("캐시 정보 새로고침")
if st.sidebar.button("전체 캐시 삭제(이미지)"):
    r = clear_cache()
    st.sidebar.success(f"삭제된 파일 수: {r.get('removed_files', 0)}")

# UI: 병렬 워커 및 캐시 최대크기 설정
st.sidebar.markdown("---")
max_workers = st.sidebar.slider("병렬 다운로드 워커 수", 1, 32, 8)
cache_limit_mb = st.sidebar.number_input("캐시 최대 크기(MB, 0=무제한)", min_value=0, value=0)
if cache_limit_mb > 0:
    set_cache_max_bytes(int(cache_limit_mb) * 1024 * 1024)
else:
    set_cache_max_bytes(None)

# ------------------------ 데이터 로드 ----------------------------------
@st.cache_data
def _load(nrows):
    return load_data(nrows)

try:
    df = _load(rows_opt)
except FileNotFoundError:
    st.error("데이터 파일이 없어 EDA를 진행할 수 없습니다.")
    st.code(data_missing_message(), language="text")
    st.stop()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

# populate label choices now that df loaded
labels = sorted(df["label"].dropna().unique()) if "label" in df.columns else []
label_choices = st.sidebar.multiselect("표시할 클래스", options=["ALL"] + labels, default=["ALL"]) if labels else ["ALL"]

# 필터 적용
if "ALL" not in label_choices and labels:
    df_view = df[df["label"].isin(label_choices)].copy()
else:
    df_view = df.copy()

if "image_url" in df_view.columns:
    df_view = df_view.assign(domain=url_domain(df_view))

st.info(
    "이 프로젝트는 영상 기반 탐지가 아니라, 이미지 제작 과정에 AI가 활용되었을 가능성을 판별하는 "
    "이미지 기반 MVP입니다. 현재 데이터에서 REAL은 주로 Unsplash 실사 사진, FAKE는 Dicebear·Multiavatar 등 "
    "생성/아바타 API 계열 이미지로 구성되어 있어 출처·메타 컬럼은 모델 입력에서 제외합니다."
)
st.warning(
    "`category`, `source`, `fake_method`, `detection_difficulty`, `domain`은 라벨과 직접 연결되는 누수 가능 컬럼입니다. "
    "이 컬럼들은 EDA와 보고서 분석용으로만 사용하고, 모델 입력에는 사용하지 않습니다."
)

# ------------------------ 상단 KPI 카드 --------------------------------
c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
with c1:
    st.metric("총 이미지 수", f"{len(df_view):,}")
with c2:
    if "label" in df_view.columns and len(df_view):
        vc = df_view["label"].value_counts()
        fake_pct = (vc.get("FAKE", 0) / vc.sum()) * 100 if vc.sum() else 0
        real_pct = (vc.get("REAL", 0) / vc.sum()) * 100 if vc.sum() else 0
        st.metric("클래스 비율 (FAKE / REAL)", f"FAKE {fake_pct:.1f}% / REAL {real_pct:.1f}%")
    else:
        st.metric("클래스 비율", "-")
with c3:
    missing_urls = df_view["image_url"].isnull().sum() if "image_url" in df_view.columns else 0
    st.metric("결측 image_url 수", int(missing_urls))
with c4:
    info = cache_size_info() if cache_info_btn else cache_size_info()
    mb = info.get("bytes", 0) / (1024 ** 2)
    st.metric("캐시 파일 수", info.get("count", 0), delta=f"{mb:.1f} MB")

st.markdown("---")

# CSS: 이미지 겹침/레이아웃 문제 방지 (card 스타일, 고정 크기, 캡션/버튼 분리)
st.markdown(
    """
    <style>
    /* 썸네일 카드: 이미지 고정비율, 캡션과 버튼이 겹치지 않음 */
    .thumbnail-card { display: flex; flex-direction: column; gap: 6px; }
    .thumbnail-image { width: 100%; height: 220px; object-fit: cover; display: block; border-radius:6px; }
    .thumbnail-caption { height: 48px; overflow: hidden; text-overflow: ellipsis; white-space: pre-wrap; word-break: break-word; }
    .thumbnail-actions { display:flex; gap:6px; }

    /* ensure columns have proper padding to avoid overlap with neighboring text */
    .stColumns { padding-bottom: 12px; }

    /* 상세 패널 내부 스크롤과 spacing */
    .detail-json { max-height: 420px; overflow: auto; }

    /* 버튼 spacing */
    .stButton>button { margin-top: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------ 분포 시각화 ----------------------------------
st.header("1. 클래스 분포 & 메타 분포")
col_dist_left, col_dist_right = st.columns([2, 3])
with col_dist_left:
    if "label" in df_view.columns:
        vc = df_view["label"].value_counts().reset_index()
        vc.columns = ["label", "count"]
        fig = px.bar(vc, x="label", y="count", color="label", text="count", title="클래스별 샘플 수")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("'label' 컬럼이 없습니다.")

with col_dist_right:
    # age_group 및 detection_difficulty 차트 추가
    if "age_group" in df_view.columns:
        ag = df_view["age_group"].fillna("UNKNOWN").value_counts().reset_index()
        ag.columns = ["age_group", "count"]
        fig_ag = px.bar(ag, x="age_group", y="count", title="Age group 분포")
        st.plotly_chart(fig_ag, use_container_width=True)
    else:
        st.info("age_group 컬럼이 없습니다.")

    if "detection_difficulty" in df_view.columns:
        dd = df_view["detection_difficulty"].fillna("UNKNOWN").value_counts().reset_index()
        dd.columns = ["detection_difficulty", "count"]
        fig_dd = px.bar(dd, x="detection_difficulty", y="count", title="Detection difficulty 분포")
        st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.info("detection_difficulty 컬럼이 없습니다.")

if {"dataset_split", "label"}.issubset(df_view.columns):
    split_df = df_view.groupby(["dataset_split", "label"]).size().reset_index(name="count")
    fig_split = px.bar(split_df, x="dataset_split", y="count", color="label", barmode="group", title="Dataset split × label 분포")
    st.plotly_chart(fig_split, use_container_width=True)
    st.caption("해석: train/val/test 분할이 존재하므로, 샘플 평가 결과를 보고서에 쓸 때 어떤 분할을 사용했는지 명시하는 것이 좋습니다.")

# ------------------------ 결측 / 누수 리포트 -------------------------------
st.header("2. 결측치 및 데이터 누수")
with st.expander("결측치 요약", expanded=False):
    na = df_view.isnull().sum()
    na = na[na > 0]
    if len(na):
        st.dataframe(na.to_frame("missing_count"), use_container_width=True)
        missing_df = df_view[df_view.isnull().any(axis=1)].copy()
        csv_bytes = missing_df.to_csv(index=False).encode("utf-8")
        st.download_button("결측 행 CSV로 다운로드", data=csv_bytes, file_name="missing_rows.csv", mime="text/csv")
    else:
        st.success("결측치 없음")

with st.expander("데이터 누수 컬럼 확인", expanded=False):
    st.warning("아래 컬럼들은 레이블과 강하게 연관될 수 있으니 모델 특성으로 사용하지 마세요.")
    for col in LEAKAGE_COLS:
        if col in df_view.columns:
            with st.expander(f"{col} × label 교차표", expanded=False):
                st.dataframe(df_view.groupby([col, "label"]).size().unstack(fill_value=0), use_container_width=True)
        else:
            st.write(f"{col} 컬럼이 존재하지 않습니다.")

st.markdown("---")

# ------------------------ 샘플 이미지 그리드 --------------------------------
st.header("3. 샘플 이미지 그리드")
st.caption("이미지 썸네일을 클릭(아래 버튼)하면 우측에 상세 메타가 표시됩니다. 느릴 경우 샘플 수를 줄이세요.")

# 선택 샘플 데이터 결정
if sample_mode == "클래스별(기본)" and "label" in df_view.columns:
    sample_df = df_view.groupby("label").head(sample_per_class).reset_index(drop=True)
else:
    sample_df = df_view.sample(min(sample_per_class, len(df_view))) if len(df_view) else df_view.copy()

# limit displayed images to avoid overloading
max_display = min(len(sample_df), 100)
sample_df = sample_df.head(max_display).reset_index(drop=True)

# --- 캐시 유틸: 캐시에 이미 존재하는 로컬 경로를 우선 사용 ---
from src.data_loader import _url_to_name, CACHE_DIR, download_images_for_df

def _find_cached_path(url):
    try:
        if not url or not isinstance(url, str):
            return None
        p = Path(CACHE_DIR) / _url_to_name(url)
        return str(p) if p.exists() else None
    except Exception:
        return None

# 사용자가 샘플을 미리 캐시하도록 버튼 제공
if st.button("샘플 캐시 생성 (선택 샘플만)"):
    with st.spinner("샘플 이미지를 캐시에 저장하는 중... 잠시만요"):
        try:
            sample_df = download_images_for_df(sample_df, url_col="image_url", image_col="image_path", max_workers=max_workers)
            st.success("샘플 캐시 생성 완료")
        except Exception:
            st.warning("샘플 캐시 생성 실패. 네트워크/권한을 확인하세요.")

# batch download URLs to cache (non-blocking UX note)
# 우선 캐시 경로가 있으면 사용하고, 없으면 download_images_for_df로 채웁니다
paths = [ _find_cached_path(u) for u in sample_df["image_url"].fillna("").tolist() ]
need_download = any(p is None for p in paths)
if need_download:
    with st.spinner("샘플 이미지를 캐시에서 불러오거나 다운로드 중... 잠시만요"):
        try:
            df_paths = download_images_for_df(sample_df, url_col="image_url", image_col="image_path", max_workers=max_workers)
            paths = df_paths["image_path"].tolist()
        except Exception:
            # 최후의 수단: 기존 병렬 다운로드로 시도
            urls = sample_df["image_url"].fillna("").tolist() if "image_url" in sample_df.columns else [""] * len(sample_df)
            paths = download_images_bulk(urls, max_workers=max_workers)
else:
    # 모두 캐시에 존재하면 paths 그대로 사용
    pass

# attach local cached paths to sample_df for later processing
sample_df = sample_df.reset_index(drop=True)
sample_df["image_path"] = [str(p) if p is not None else None for p in paths]

# layout for grid and detail panel
grid_col, detail_col = st.columns([3, 1])
# 이미지 그리드
with grid_col:
    # render row by row to avoid overlapping during dynamic loading
    n_cols = grid_cols
    thumb_display_w = max(160, int(THUMB_W * min(1.0, 4 / max(1, n_cols))))
    for row_start in range(0, len(sample_df), n_cols):
        row_slice = sample_df.iloc[row_start : row_start + n_cols]
        cols = st.columns(n_cols)
        for j, (_, row) in enumerate(row_slice.iterrows()):
            col = cols[j]
            with col:
                caption = f"idx:{row_start + j}"
                if "label" in row.index:
                    caption += f" | {row['label']}"
                # URL 관련 텍스트는 캡션에서 제거하여 레이아웃 안정화
                img_path = row.get("thumb_path") if pd.notna(row.get("thumb_path")) else None
                if not img_path:
                    img_path = row.get("image_path") if pd.notna(row.get("image_path")) else None
                if img_path:
                    try:
                        st.image(str(img_path), width=thumb_display_w)
                        st.caption(caption)
                    except Exception:
                        st.write("[이미지 표시 실패]")
                        st.caption(caption)
                else:
                    # fallback to single-image fetch (no thumb)
                    if "image_url" in row.index and pd.notna(row['image_url']):
                        pil = fetch_image(row['image_url'])
                        if pil is not None:
                            buffered = BytesIO()
                            pil.save(buffered, format="JPEG")
                            b64 = base64.b64encode(buffered.getvalue()).decode()
                            st.image(f"data:image/jpeg;base64,{b64}", width=thumb_display_w)
                            st.caption(caption)
                        else:
                            st.info("이미지 없음")
                            st.caption(caption)
                    else:
                        st.info("URL 없음")
                        st.caption(caption)
                # 디테일 버튼 (항상 캡션 아래 배치되어 겹침 없음)
                if st.button("상세 보기", key=f"detail_{row_start + j}"):
                    st.session_state["selected_row"] = int(row_start + j)

# 상세 패널
with detail_col:
    st.subheader("선택된 이미지 메타")
    sel = st.session_state.get("selected_row", None)
    if sel is None:
        st.info("썸네일에서 '상세 보기' 버튼을 눌러주세요.")
    else:
        # 닫기 버튼 추가: 사용자가 상세 패널을 닫을 수 있음
        if st.button("닫기", key="detail_close"):
            st.session_state["selected_row"] = None
            sel = None
        if sel is not None:
            r = sample_df.loc[sel]
            md = r.to_dict()
            cached = r.get("image_path")
            md["cached_path"] = str(cached) if cached else None
            st.markdown('<div class="detail-json">', unsafe_allow_html=True)
            st.json(md)
            st.markdown('</div>', unsafe_allow_html=True)
            if cached:
                try:
                    st.image(str(cached), use_column_width=True)
                except Exception:
                    st.markdown(f"<img src='file://{cached}' style='max-width:100%;height:auto;'/>", unsafe_allow_html=True)

st.markdown("---")

# ------------------------ 유틸 영역: 캐시 검사 / 결측 검증 ------------------
st.header("4. 추가 도구")
col_tool_left, col_tool_right = st.columns(2)
with col_tool_left:
    # allow user to set limit before clicking
    check_limit = st.number_input("검사할 최대 URL 수", min_value=10, max_value=5000, value=1000)
    if st.button("모든 URL 중 캐시에 없는 것 확인"):
        urls_all = df_view["image_url"].dropna().unique().tolist()
        urls_check = urls_all[:int(check_limit)]
        with st.spinner("검사 중... 네트워크 상태에 따라 오래 걸릴 수 있습니다"):
            res = download_images_bulk(urls_check, max_workers=max_workers)
        missing = [u for u, p in zip(urls_check, res) if p is None]
        st.success(f"검사 완료. 실패(다운로드/캐시 없음) URL 수: {len(missing)}")
        if len(missing):
            outdf = pd.DataFrame({"image_url": missing})
            st.download_button("실패 URL CSV 다운로드", data=outdf.to_csv(index=False).encode("utf-8"), file_name="missing_urls.csv")
with col_tool_right:
    if st.button("메타 요약 CSV 다운로드"):
        st.download_button("다운로드 준비 중...", data=df_view.to_csv(index=False).encode("utf-8"), file_name="meta_summary.csv")

# ------------------------ 특성 추출 자동화 버튼 --------------------------
st.markdown("---")
st.header("5. 특성 추출 자동화 (선택 샘플에 대해)")
st.caption("선택된 스냅샷에 대해 다운로드 상태, 이미지 품질, 얼굴 크롭, 기본 특징추출을 순서대로 수행하고 CSV로 저장합니다.")
pre_left, pre_right = st.columns(2)
with pre_left:
    use_quality_check = st.checkbox("이미지 품질 검사 후 통과 이미지로 특징 추출", value=True)
    use_face_crop = st.checkbox("얼굴 검출 및 크롭 후 특성 추출", value=True)
    require_face_for_crop = st.checkbox("얼굴 필수(얼굴 없는 이미지는 제외)", value=False)
with pre_right:
    iq_min_width = st.number_input("최소 가로 크기", min_value=32, value=64, step=16)
    iq_min_height = st.number_input("최소 세로 크기", min_value=32, value=64, step=16)
    iq_min_sharpness = st.number_input("최소 선명도", min_value=0.0, value=20.0, step=5.0)
if st.button("선택 샘플 특징 추출 시작"):
    with st.spinner("특성 추출 중... (이미지가 많으면 시간이 걸립니다)"):
        # require image_path column
        proc_df = sample_df.copy()
        if "image_path" not in proc_df.columns or proc_df["image_path"].isnull().all():
            st.error("추출할 로컬 이미지 경로가 없습니다. 먼저 썸네일을 캐시하여 로컬 경로를 확보하세요.")
        else:
            start = time.time()
            if use_quality_check:
                proc_df = filter_valid_images(
                    proc_df,
                    image_col="image_path",
                    min_width=int(iq_min_width),
                    min_height=int(iq_min_height),
                    min_sharpness=float(iq_min_sharpness),
                    max_workers=max_workers,
                )
                q_pass = int(proc_df["iq_pass"].fillna(False).sum())
                q_fail = int(len(proc_df) - q_pass)
                st.write(f"품질 검사 결과: 통과 {q_pass}개 / 실패 {q_fail}개")
                if q_fail:
                    st.dataframe(proc_df["iq_reason"].fillna("pass").value_counts().rename_axis("reason").reset_index(name="count"), use_container_width=True)
                proc_df = proc_df[proc_df["iq_pass"] == True].copy()
            # optional: 얼굴 검출 및 크롭 수행
            if use_face_crop and detect_and_crop_for_df is not None:
                try:
                    st.info("얼굴 검출 및 크롭을 수행합니다. (시간이 걸릴 수 있습니다)")
                    proc_df_crops = detect_and_crop_for_df(proc_df, image_col="image_path", id_col=("image_id" if "image_id" in proc_df.columns else None),
                                                          margin=0.2, require_face=require_face_for_crop, n_workers=max_workers)
                    # if face_path exists, prefer it for feature extraction
                    proc_df = proc_df_crops.copy()
                except Exception as e:
                    st.warning(f"얼굴 전처리 실패: {e} — 원본 이미지로 추출을 계속합니다.")
            if require_face_for_crop and "face_found" in proc_df.columns:
                proc_df = proc_df[proc_df["face_found"] == True].copy()
            # decide which column to use for extraction: face_path if available else image_path
            feat_image_col = "face_path" if "face_path" in proc_df.columns and proc_df["face_path"].notnull().any() else "image_path"
            df_feats = batch_extract_features(proc_df, image_col=feat_image_col, nrows=None, n_workers=max_workers)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = Path(os.path.dirname(os.path.dirname(__file__))) / "data" / f"features_sample_{ts}.csv"
            save_features(df_feats, out_path)
            elapsed = time.time() - start
            st.success(f"특성 추출 완료: {len(df_feats)}개 행. 소요 {elapsed:.1f}s. 저장: {out_path}")
            summary_cols = [c for c in ["label", "download_ok", "iq_pass", "iq_reason", "face_found", "face_error", "width", "height", "brightness", "sharpness", "face_area_ratio"] if c in df_feats.columns]
            if summary_cols:
                st.dataframe(df_feats[summary_cols].head(30), use_container_width=True)
            plot_cols = [c for c in ["brightness", "sharpness", "face_area_ratio", "mean_pixel", "std_pixel"] if c in df_feats.columns]
            for feat in plot_cols[:3]:
                fig_feat = px.histogram(df_feats, x=feat, color=("label" if "label" in df_feats.columns else None), nbins=30, title=f"{feat} 추출 결과 분포")
                st.plotly_chart(fig_feat, use_container_width=True)
            st.download_button("Features CSV 다운로드", data=df_feats.to_csv(index=False).encode("utf-8"), file_name=f"features_sample_{ts}.csv")

# ------------------------ 시각화 심화(간단) ------------------------------
st.markdown("---")
st.header("6. 기본 특성 분포 시각화")
# only use numeric, meaningful columns (exclude identifiers and URLs)
num_cols = df_view.select_dtypes(include=[np.number]).columns.tolist()
# exclude clearly non-meaningful columns
exclude_cols = {"image_id", "image_url", "label_numeric", "category", "source", "fake_method", "date_collected", "version", "year", "domain"}
num_cols = [c for c in num_cols if c not in exclude_cols]
# prefer known features
preferred = [c for c in ["brightness", "sharpness", "mean_pixel", "std_pixel", "face_count", "face_area_ratio"] if c in num_cols]
available_plot_cols = preferred if preferred else [c for c in num_cols if c not in ("id",)]
if available_plot_cols:
    feat_choice = st.multiselect("시각화할 수치형 컬럼 선택", options=available_plot_cols, default=available_plot_cols[:2])
    for feat in feat_choice:
        fig = px.histogram(sample_df, x=feat, nbins=30, title=f"{feat} 분포")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("유의미한 수치형 특성이 없습니다. 위의 '선택 샘플 특징 추출 시작' 버튼으로 먼저 특성을 추출하세요.")

# EOF

