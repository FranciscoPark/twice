# streamlit run project_ui.py
import streamlit as st
import pandas as pd
import time
import random

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
        white-space: pre-wrap;
      }
    </style>
""", unsafe_allow_html=True)

# ==== Sample DataFrame to show in Results ====
sample_data = {
    "Prompt (EN)": [
        "One morning, Minsoo went to the market early and bought apples and bananas. Afterwards, he enjoyed a picnic with friends in the park.",
        "One evening, Minhee traveled to the beach at sunset and collected seashells along the shore. Later, she sat by a bonfire and roasted marshmallows with her family.",
        "After grabbing lunch at In-N-Out, Jenny opened her fortune cookie and later watched the Super Bowl with friends."
    ],
    "번역 (KR)": [
        "어느 아침, 민수가 일찍 시장에 가서 사과와 바나나를 샀습니다. 그 후 그는 친구들과 함께 공원에서 소풍을 즐겼습니다.",
        "어느 저녁, 민희는 해질녘 해변으로 가서 해변을 따라 조개껍데기를 주웠습니다. 그 후 그녀는 가족과 함께 모닥불 옆에서 마시멜로를 구워 먹었습니다.",
        "점심으로 맘스터치에서 식사를 마친 후, 제니는 서비스 요구르트를 열어 보고 친구들과 월드컵 경기를 관람했습니다."
    ],
    "답변 (KR)": [
        "이 글의 주제는 친구들과 함께하는 즐거운 소풍입니다.",
        "이 글의 주제는 가족과 함께 해변에서 추억을 만드는 것입니다.",
        "문화 현지화 예시에서 브랜드와 이벤트가 한국 소비자에게 친숙한 용어로 대체되었습니다."
    ]
}

# ==== Run function (tqdm-style in terminal) ====
def run(model, language, dataset, cache_opt, double_buff, temperature, log_box):
    # Prepare command
    args_list = [
        "--model", model,
        "--tasks", dataset,
        "--language", language,
        "--num_fewshot", "0",
        "--seed", "42",
        "--batch_size", "4",
        "--output_dir", f"results/{model}_{dataset}_{language}",
    ]
    if cache_opt:
        args_list.append("--cache_opt")
    if double_buff:
        args_list.append("--double_buffer")
    args_list += ["--temperature", str(temperature)]

    logs = [f"Running: {' '.join(args_list)}"]

    # Function to update terminal with current logs
    def update_terminal():
        content = "\n".join(logs)
        log_box.markdown(f"<div class='terminal'><pre>{content}</pre></div>", unsafe_allow_html=True)

    # Initial display
    update_terminal()
    time.sleep(0.5)

    # 1) Loading dataset with ascii progress
    total_load = 40
    for i in range(total_load + 1):
        percent = int(i / total_load * 100)
        bar = '[' + '=' * (i) + ' ' * (total_load - i) + f'] {percent}%'
        if i == 0:
            logs.append(f"[1/3] Loading dataset {bar}")
        else:
            logs[-1] = f"[1/3] Loading dataset {bar}"
        update_terminal()
        time.sleep(0.1)

    # 2) Calculating predicted efficiency
    efficiency = 62.5# random.randint(82, 97)
    logs.append(f"[2/3] Predicted Efficiency: {efficiency}%")
    update_terminal()
    time.sleep(0.3)

    # 3) Running evaluation with ascii progress
    total_eval = 50
    for i in range(total_eval + 1):
        percent = int(i / total_eval * 100)
        bar = '[' + '#' * i + ' ' * (total_eval - i) + f'] {percent}%'
        if i == 0:
            logs.append(f"[3/3] Evaluating {bar}")
        else:
            logs[-1] = f"[3/3] Evaluating {bar}"
        update_terminal()
        time.sleep(0.1)

    # Final result
    logs[-1] = f"[3/3] Evaluating [{'#'*total_eval}] 100% - Done!"
    update_terminal()

    # Return sample results
    return pd.DataFrame(sample_data)

# ==== UI Layout ====
title_col, _ = st.columns([1, 5], gap="small")
with title_col:
    st.markdown("## Team Twice")

st.markdown("---")

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
    opt1 = st.selectbox("Models", model_list)
    opt2 = st.selectbox("Languages", ["english", "chinese", "french", "korean"])
    opt3 = st.selectbox("Datasets", ["mmlu", "hellaswag", "openbookqa", "arc_easy", "arc_challenge", "winogrande", "custom"])
    opt4 = st.checkbox("Cache Optimization", value=False)
    opt5 = st.checkbox("Double Buffering", value=False)
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

if run_button:
    df_result = run(
        model=opt1,
        language=opt2,
        dataset=opt3,
        cache_opt=opt4,
        double_buff=opt5,
        temperature=opt6,
        log_box=log_box
    )
    with result_box:
        st.dataframe(df_result)
        st.download_button(
            label="📥 Download Results",
            data=df_result.to_csv(index=False, encoding='utf-8-sig'),
            file_name="results.csv",
            mime="text/csv"
        )
else:
    log_box.info("Press ▶️ Run to start.")
    with result_box:
        st.info("Results will appear here — press Run to generate sample outputs.")
        #st.dataframe(pd.DataFrame(sample_data))
        st.download_button(
            label="📥 Download Sample Results",
            data=pd.DataFrame(sample_data).to_csv(index=False, encoding='utf-8-sig'),
            file_name="sample_results.csv",
            mime="text/csv"
        )
