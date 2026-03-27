import streamlit as st

st.set_page_config(page_title="단계별 입력", page_icon="📝")
st.title('📝 단계별 정보 입력')

# ---- 상태 초기화 ----
if 'step' not in st.session_state:
    st.session_state.step = 1
# 데이터를 안전하게 보관할 변수들 초기화
if 'saved_name' not in st.session_state:
    st.session_state.saved_name = ""
if 'saved_age' not in st.session_state:
    st.session_state.saved_age = 20
if 'saved_interests' not in st.session_state:
    st.session_state.saved_interests = []

# ---- 진행률 표시 ----
progress = st.session_state.step / 3
st.progress(progress, text=f'Step {st.session_state.step} / 3')

# ---- Step 1: 기본 정보 ----
if st.session_state.step == 1:
    st.subheader('Step 1: 기본 정보')
    name = st.text_input('이름', value=st.session_state.saved_name)
    age = st.number_input('나이', min_value=1, max_value=100, value=st.session_state.saved_age)

    if st.button('다음 →'):
        if name.strip():
            st.session_state.saved_name = name
            st.session_state.saved_age = age
            st.session_state.step = 2
            st.rerun()
        else:
            st.warning('이름을 입력해주세요.')

# ---- Step 2: 관심 분야 ----
elif st.session_state.step == 2:
    st.subheader('Step 2: 관심 분야')
    # key 대신 default 설정을 사용하여 데이터 유실 방지
    interests = st.multiselect(
        '관심 분야를 선택하세요',
        ['데이터 분석', '웹 개발', 'AI/ML', '모바일', '게임', '보안'],
        default=st.session_state.saved_interests
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button('← 이전'):
            st.session_state.saved_interests = interests # 입력값 저장
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button('다음 →'):
            st.session_state.saved_interests = interests # 입력값 저장
            st.session_state.step = 3
            st.rerun()

# ---- Step 3: 확인 ----
elif st.session_state.step == 3:
    st.subheader('Step 3: 입력 확인')
    # 이제 위젯이 사라져도 saved_interests에 데이터가 남아있어 정상 출력됨
    st.write(f"**이름**: {st.session_state.saved_name}")
    st.write(f"**나이**: {st.session_state.saved_age}")
    st.write(f"**관심 분야**: {', '.join(st.session_state.saved_interests) if st.session_state.saved_interests else '없음'}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button('← 이전'):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button('✅ 제출'):
            st.balloons()
            st.success('제출이 완료되었습니다!')
            # 제출 후 깔끔한 재시작을 위해 모든 데이터 삭제
            for key in list(st.session_state.keys()):
                del st.session_state[key]