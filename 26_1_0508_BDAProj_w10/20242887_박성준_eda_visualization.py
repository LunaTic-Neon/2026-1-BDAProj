import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from urllib.parse import unquote
import os
import re

# ============================================================
# 1. 페이지 구성 및 다크모드 전용 고대비 스타일링
# ============================================================
st.set_page_config(
    page_title="CSIC 2010 데이터셋 비교 분석 리포트",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 12px;
    }
    div[data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-weight: 800;
        font-size: 1.5rem !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: white !important;
    }
    h1 { color: #f0f6fc; border-bottom: 2px solid #30363d; padding-bottom: 15px; }
    h2, h3 { color: #58a6ff; margin-top: 25px; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 2. 데이터 로드 및 비교용 전처리
# ============================================================
@st.cache_data
def load_comparison_data():
    files = {
        "10K 데이터": "csic2010_requests_10000.csv",
        "61K 데이터": "csic2010_requests_61000.csv"
    }
    
    combined_list = []
    
    for label, filename in files.items():
        data_path = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(data_path):
            temp_df = pd.read_csv(data_path)
            temp_df["dataset_name"] = label  # 데이터셋 구분 컬럼 추가
            
            # 전처리 공통 적용
            temp_df["url_decoded"] = temp_df["url"].apply(lambda x: unquote(str(x), encoding="latin-1"))
            temp_df["body_decoded"] = temp_df["body"].fillna("").apply(lambda x: unquote(str(x), encoding="latin-1"))
            temp_df["full_text"] = temp_df["url_decoded"] + " " + temp_df["body_decoded"]
            temp_df["url_length"] = temp_df["url_decoded"].str.len()
            temp_df["body_length"] = temp_df["body_decoded"].str.len()
            temp_df["is_attack"] = (temp_df["label"] == "Anomalous").astype(int)
            
            combined_list.append(temp_df)
        else:
            st.warning(f"{filename} 파일을 찾을 수 없습니다.")

    if not combined_list:
        st.error("분석할 데이터 파일이 없습니다.")
        st.stop()
        
    return pd.concat(combined_list, ignore_index=True)

df = load_comparison_data()

# ============================================================
# 3. 사이드바: 데이터셋별 핵심 지표 비교
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/compare.png", width=60)
    st.title("데이터셋 비교 요약")
    
    for name in df["dataset_name"].unique():
        sub_df = df[df["dataset_name"] == name]
        st.subheader(f"📊 {name}")
        st.metric("총 요청 수", f"{len(sub_df):,} 건")
        
        c1, c2 = st.columns(2)
        with c1:
            st.caption("정상")
            st.write(f"{(sub_df['label']=='Normal').sum():,}")
        with c2:
            st.caption("이상")
            st.write(f"{(sub_df['label']=='Anomalous').sum():,}")
        
        attack_ratio = (sub_df['is_attack'].mean() * 100)
        st.progress(attack_ratio / 100)
        st.markdown(f"공격 비중: **{attack_ratio:.2f}%**")
        st.markdown("---")

# ============================================================
# 4. 메인 비교 리포트 영역
# ============================================================
st.title("⚖️ CSIC 2010 데이터셋 대조 분석")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 클래스 분포 비교", 
    "📐 특징값 편차 분석", 
    "🎯 시그니처 검출 비교", 
    "📄 샘플 로그 대조"
])

# ─── 탭 1: 클래스 분포 비교 ──────────────────────────────────────
with tab1:
    st.subheader("데이터셋별 라벨 및 메서드 구성 차이")
    
    col1, col2 = st.columns(2)
    with col1:
        # 데이터셋별 라벨 비중 비교 (Grouped Bar)
        label_dist = df.groupby(["dataset_name", "label"]).size().reset_index(name="count")
        fig1 = px.bar(
            label_dist, x="dataset_name", y="count", color="label",
            title="데이터셋별 클래스 균형도 비교",
            barmode="group", template="plotly_dark",
            color_discrete_map={"Normal": "#238636", "Anomalous": "#da3633"}
        )
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        # 메서드 분포 비교
        method_dist = df.groupby(["dataset_name", "method"]).size().reset_index(name="count")
        fig2 = px.bar(
            method_dist, x="method", y="count", color="dataset_name",
            title="데이터셋별 HTTP 메서드 활용 차이",
            barmode="group", template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig2, use_container_width=True)

# ─── 탭 2: 특징값 편차 분석 ──────────────────────────────────────
with tab2:
    st.subheader("데이터셋별 특징(Feature) 분포 비교")
    st.info("두 데이터셋 간에 공격 페이로드의 물리적 특성(길이 등)이 유사하게 유지되는지 확인합니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        fig3 = px.box(
            df, x="dataset_name", y="url_length", color="label",
            title="데이터셋별 URL 길이 분포 편차", template="plotly_dark",
            color_discrete_map={"Normal": "#238636", "Anomalous": "#da3633"}
        )
        st.plotly_chart(fig3, use_container_width=True)
        
    with col2:
        # 바디 길이는 POST 요청에 대해서만 비교
        post_only = df[df["method"] == "POST"]
        fig4 = px.violin(
            post_only, x="dataset_name", y="body_length", color="label",
            box=True, points="all", title="POST 바디 길이 밀도 비교",
            template="plotly_dark", color_discrete_map={"Normal": "#238636", "Anomalous": "#da3633"}
        )
        st.plotly_chart(fig4, use_container_width=True)

# ─── 탭 3: 공격 패턴 추출 ────────────────────────────────
with tab3:
    st.subheader("데이터셋별 보안 시그니처 탐지 빈도")
    
    attack_data = {
        "SQLi": ["select", "union", "1=1", "--"],
        "XSS": ["<script", "alert", "onerror"],
        "Path Traversal": ["../", "/etc/passwd"]
    }
    
    kw_comparison = []
    for d_name in df["dataset_name"].unique():
        target_df = df[(df["dataset_name"] == d_name) & (df["label"] == "Anomalous")]
        for cat, kws in attack_data.items():
            for kw in kws:
                count = target_df["full_text"].str.contains(re.escape(kw), case=False).sum()
                # 비율로 계산하여 데이터 크기 차이 보정
                ratio = (count / len(target_df)) * 100 if len(target_df) > 0 else 0
                kw_comparison.append({"데이터셋": d_name, "키워드": kw, "탐지율(%)": ratio})
            
    fig5 = px.bar(
        pd.DataFrame(kw_comparison), x="키워드", y="탐지율(%)", color="데이터셋",
        title="공격 데이터 내 주요 키워드 점유율 비교 (%)",
        barmode="group", template="plotly_dark", text_auto=".1f"
    )
    st.plotly_chart(fig5, use_container_width=True)

# ─── 탭 4: 샘플 데이터 대조 ─────────────────────────────────
with tab4:
    st.subheader("데이터셋별 개별 샘플 비교")
    
    sel_dataset = st.selectbox("조회할 데이터셋 선택", df["dataset_name"].unique())
    view_df = df[df["dataset_name"] == sel_dataset]
    
    idx_c1, idx_c2 = st.columns(2)
    with idx_c1:
        n_id = st.number_input(f"{sel_dataset} 정상 인덱스", 0, (view_df['label']=='Normal').sum()-1, 0)
        n_row = view_df[view_df["label"] == "Normal"].iloc[n_id]
        st.success(f"**🟢 {sel_dataset} 정상 샘플**")
        st.code(f"URL: {n_row['url_decoded']}\nBody: {n_row['body_decoded']}", language="http")
        
    with idx_c2:
        a_id = st.number_input(f"{sel_dataset} 이상 인덱스", 0, (view_df['label']=='Anomalous').sum()-1, 0)
        a_row = view_df[view_df["label"] == "Anomalous"].iloc[a_id]
        st.error(f"**🔴 {sel_dataset} 이상 샘플**")
        st.code(f"URL: {a_row['url_decoded']}\nBody: {a_row['body_decoded']}", language="http")

# ============================================================
# 5. 최종 비교 시사점
# ============================================================
st.divider()
st.subheader("📝 데이터셋 간 주요 차이점 요약")
res_c1, res_c2, res_c3 = st.columns(3)

# 실제 데이터 값에 기반한 동적 요약 예시
res_c1.info(f"**규모 편차**\n\n61K 데이터셋이 10K 대비 약 {len(df[df['dataset_name']=='61K 데이터'])/len(df[df['dataset_name']=='10K 데이터']):.1f}배 더 많은 샘플을 보유하고 있어 학습 안정성이 높을 것으로 예상됩니다.")
res_c2.success("**일관성 확인**\n\n두 데이터셋 모두 이상 트래픽에서 URL 및 Body 길이가 길어지는 경향성이 공통적으로 관찰됩니다.")
res_c3.warning("**패턴 밀도**\n\n탐지율 비교 탭을 통해 특정 공격 키워드가 대용량 데이터셋에서 희석되는지, 혹은 더 명확해지는지 판단이 필요합니다.")