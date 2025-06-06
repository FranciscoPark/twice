# streamlit run project_ui.py
import streamlit as st
import pandas as pd
import time
from io import StringIO

# ==== Page configuration ====
st.set_page_config(
    page_title="Three-Panel Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==== Global CSS overrides ====
st.markdown("""
    <style>
      .main .block-container {
        padding: 1rem 2rem;
        max-width: 100% !important;
      }
      .panel {
        background: #F7F9FA;
        padding: 1rem;
        border-radius: 8px;
      }
      .terminal {
        background: #1E1E1E;
        color: #00FF00;
        font-family: monospace;
        padding: 1rem;
        border-radius: 5px;
        height: 300px;
        overflow-y: auto;
      }
    </style>
""", unsafe_allow_html=True)

# ==== Sample DataFrame to show in Results (샘플 데이터) ====
sample_data = {
    "Prompt (EN)": [
        "What is the capital of France?",
        "How many continents are there?",
        "Who wrote 'Hamlet'?"
    ],
    "번역 (KR)": [
        "프랑스의 수도는 무엇인가요?",
        "대륙은 몇 개 있나요?",
        "'햄릿'은 누가 썼나요?"
    ],
    "답변 (KR)": [
        "파리는 프랑스의 수도입니다.",
        "대륙은 총 7개 있습니다.",
        "'햄릿'은 윌리엄 셰익스피어가 썼습니다."
    ]
}


def run(model, language, dataset, cache_opt, double_buff, temperature, log_box):
    """
    선택된 옵션을 인자로 받아서 실제 작업을 수행하고,
    터미널(log_box)에 진행 상황을 출력한 뒤, 결과 DataFrame을 반환합니다.
    """
    buffer = StringIO()

    ########## To DO ##########
    ########## To DO ##########
    ########## To DO ##########
    ########## To DO ##########
    
    # 이 부분에서 실행 시키고,
    # buffer.write 에 log 남기고
    # df로 결과 return

    for step in range(1, 11):
        # 샘플 진행 로그를 생성하는 부분입니다.
        timestamp = time.strftime("%H:%M:%S")
        buffer.write(f"[{timestamp}] Step {step}/10 complete\n")

        # 터미널 화면에 출력
        log_box.markdown(
            f"<div class='terminal'><pre>{buffer.getvalue()}</pre></div>",
            unsafe_allow_html=True
        )

        time.sleep(0.2) 
        
        
    df = pd.DataFrame(sample_data)  # 샘플 결과 return

    ########## To DO ##########
    ########## To DO ##########
    ########## To DO ##########
    ########## To DO ##########
    
    return df

# ==== Top: title aligned left ====
title_col, _ = st.columns([1, 5], gap="small")
with title_col:
    st.markdown("## Team Twice")

st.markdown("---")

# ==== Second row: three equal panels ====
opt_col, log_col, res_col = st.columns(3, gap="large")

# Panel 1: Options
with opt_col:
    st.markdown("### ⚙️ Options")
    st.divider()

    # Model selection
    opt1 = st.selectbox("Models", ["llama", "mistral", "qwen"])
    # Language selection
    opt2 = st.selectbox("Languages", ["english", "chinese", "french", "korean"])
    # Dataset selection
    opt3 = st.selectbox("Datasets", ["MMLU", "hellaswag", "commonsenseqa"])
    # Cache optimization toggle
    opt4 = st.checkbox("Cache Optimization", value=False)
    # Double buffering toggle
    opt5 = st.checkbox("Double Buffering", value=False)
    # Temperature slider
    opt6 = st.slider("Temperature", 0.0, 1.0, 0.7)

    run_button = st.button("🚀 Run", use_container_width=True)

# Panel 2: Live log
with log_col:
    st.markdown("### 🖥️ Execution Log")
    st.divider()
    log_box = st.empty()

# Panel 3: Results
with res_col:
    st.markdown("### 📊 Results")
    st.divider()
    result_box = st.container()

# ==== 실행 및 결과 표시 ====
if run_button:
    # 실제 실행되는 부분 및 결과 리턴  ####################################################################
    df_result = run(
        model=opt1,
        language=opt2,
        dataset=opt3,
        cache_opt=opt4,
        double_buff=opt5,
        temperature=opt6,
        log_box=log_box
    )

    # run() 함수가 반환한 DataFrame을 결과 영역에 표시
    with result_box:
        st.dataframe(df_result)
        st.download_button(
            label="📥 Download Results",
            data=df_result.to_csv(index=False, encoding='utf-8-sig'),
            file_name="sample_results.csv",
            mime="text/csv"
        )
else:
    # Run 버튼을 누르지 않았을 때의 기본 화면
    log_box.info("Press ▶️ Run to start.")
    with result_box:
        st.info("Results will appear here — press Run to generate sample outputs.")
        st.dataframe(pd.DataFrame(sample_data))
        st.download_button(
            label="📥 Download Sample Results",
            data=pd.DataFrame(sample_data).to_csv(index=False, encoding='utf-8-sig'),
            file_name="sample_results.csv",
            mime="text/csv"
        )
