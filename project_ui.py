# streamlit run project_ui.py
import streamlit as st
import pandas as pd
import time
from io import StringIO
from main import run_main_with_args
import os
import contextlib
import threading
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
    # Construct CLI-like arguments
    args_list = [
        "--model", model,
        "--tasks", dataset,
        "--language", language,
        "--num_fewshot", "0",  # or based on language if you want
        "--seed", "42",
        "--batch_size", "4",
        "--output_dir", f"results/{model}_{dataset}_{language}",
    ]

    # You can conditionally append more args:
    if cache_opt:
        args_list.append("--cache_opt")
    if double_buff:
        args_list.append("--double_buffer")
    args_list += ["--temperature", str(temperature)]

    # Optionally show the command in the log box
    log_box.markdown(
        f"<div class='terminal'><pre>Running: {' '.join(args_list)}</pre></div>",
        unsafe_allow_html=True
    )

    # Call main evaluation logic
    df = run_main_with_args(args_list,log_box)

    # Load result CSV (replace with your actual return path)

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
    model_list = [
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B-Instruct",

        "Qwen/Qwen3-0.6B-Base",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B-Base",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B-Base",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B-Base",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B-Base",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",

        "mistralai/Mistral-7B-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-Nemo-Base-2407",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "mistralai/Mistral-Small-24B-Base-2501",
        "mistralai/Mistral-Small-24B-Instruct-2501"
    ]
    # Model selection
    opt1 = st.selectbox("Models", model_list)
    # Language selection
    opt2 = st.selectbox("Languages", ["english", "chinese", "french", "korean"])
    # Dataset selection
    opt3 = st.selectbox("Datasets", ["mmlu", "hellaswag", "openbookqa","arc_easy","arc_challenge","winogrande","custom"])
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
