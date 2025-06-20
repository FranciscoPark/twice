# project_ui_demo.py  ── deep‑blue professional theme (v3)
import streamlit as st
import pandas as pd
import time

# ==== Page configuration ====
st.set_page_config(
    page_title="Three‑Panel Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==== Global CSS (harmonised palettes) ====
st.markdown(r"""
<style>
/* ---------- Base ---------- */
body, .main .block-container {
    background-color: #f3f6fc;
    color: #1e293b;           /* dark slate */
    font-family: "Inter", "Pretendard", sans-serif;
    padding: 1rem 2rem;
    max-width: 100% !important;
}

/* ---------- Headings ---------- */
h1, h2, h3, h4 {
    color: #1e40af;           /* primary deep blue */
    font-weight: 700;
}

/* ---------- Column cards ---------- */
[data-testid="column"] > div:first-child {
    background: #ffffff;
    border: 1px solid #d4d9e8;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,.04);
}

/* ---------- Terminal ---------- */
.terminal {
    background: #081026;
    color: #6ee7b7;
    font-family: "Fira Code", monospace;
    padding: 1rem;
    border-radius: 10px;
    height: 300px;
    overflow-y: auto;
    white-space: pre-wrap;
}

/* ---------- Primary buttons ---------- */
.stButton > button {
    background-color: #1e40af;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
    transition: background .2s;
}
.stButton > button:hover {
    background-color: #15308a;
    color: #fff;
}

/* ---------- Slider knob + filled track ---------- */
.stSlider > div[data-testid="stSlider"] .st-cy {
    background-color: #1e40af !important;  /* knob */
}
.stSlider > div[data-testid="stSlider"] > div > div:nth-child(4) {
    background: #3b82f6 !important;        /* filled track */
}

/* ---------- Selectbox / checkbox aesthetics ---------- */
.stSelectbox, .stCheckbox label, .stDownloadButton {
    font-size: 0.93rem;
}
.stSelectbox > div > div {
    border-radius: 6px !important;
}

/* ---------- Progress bars (ascii fake) ---------- */
.bar-blue {color:#3b82f6;}
.bar-fill {color:#1e40af;}
hr {border-top: 1px solid #d4d9e8;}
</style>
""", unsafe_allow_html=True)

# ==== Sample DataFrame ====
sample_data = {
    "Prompt (EN)": [
        "One morning, Minsoo went to the market early and bought apples and bananas. Afterwards, he enjoyed a picnic with friends in the park.",
        "One evening, Minhee traveled to the beach at sunset and collected seashells along the shore. Later, she sat by a bonfire and roasted marshmallows with her family.",
        "After grabbing lunch at In‑N‑Out, Jenny opened her fortune cookie and later watched the Super Bowl with friends."
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

# ==== Run function ====
def run(model, language, dataset, cache_opt, double_buff, temperature, log_box):
    args = [
        "--model", model, "--tasks", dataset, "--language", language,
        "--num_fewshot", "0", "--seed", "42", "--batch_size", "4",
        "--output_dir", f"results/{model}_{dataset}_{language}",
        "--temperature", str(temperature)
    ]
    if cache_opt:
        args.append("--cache_opt")
    if double_buff:
        args.append("--double_buffer")

    logs = [f"Running: {' '.join(args)}"]

    def refresh():
        log_box.markdown(f"<div class='terminal'><pre>{'\\n'.join(logs)}</pre></div>",
                         unsafe_allow_html=True)

    refresh(); time.sleep(.4)

    # 1) Loading dataset (fake progress)
    for i in range(41):
        pct = i*100//40
        bar = f"[<span class='bar-fill'>{'█'*i}</span><span class='bar-blue'>{'░'*(40-i)}</span>] {pct}%"
        if i == 0:
            logs.append(f"[1/3] Loading dataset {bar}")
        else:
            logs[-1] = f"[1/3] Loading dataset {bar}"
        refresh(); time.sleep(0.025)

    # 2) Efficiency
    logs.append("[2/3] Predicted Efficiency: 62.5%"); refresh(); time.sleep(.3)

    # 3) Evaluation
    for i in range(51):
        pct = i*100//50
        bar = f"[<span class='bar-fill'>{'█'*i}</span><span class='bar-blue'>{'░'*(50-i)}</span>] {pct}%"
        if i == 0:
            logs.append(f"[3/3] Evaluating {bar}")
        else:
            logs[-1] = f"[3/3] Evaluating {bar}"
        refresh(); time.sleep(0.025)

    logs[-1] = f"[3/3] Evaluating <span class='bar-fill'>{'█'*50}</span> 100% – Done!"
    refresh()
    return pd.DataFrame(sample_data)

# ==== UI Layout ====
title_col, _ = st.columns([1,5], gap="small")
with title_col:
    st.markdown("## Team Twice")

st.markdown("---")

opt_col, log_col, res_col = st.columns(3, gap="large")

# ---------- Panel 1 : Options ----------
with opt_col:
    st.markdown("### ⚙️ Options")
    st.divider()
    models = [
        "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B", "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B-Base", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B-Base", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B",
        "mistralai/Mistral-7B-v0.3", "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-Nemo-Base-2407", "mistralai/Mistral-Nemo-Instruct-2407",
        "mistralai/Mistral-Small-24B-Base-2501", "mistralai/Mistral-Small-24B-Instruct-2501"
    ]
    m = st.selectbox("Models", models)
    lang = st.selectbox("Languages", ["english", "chinese", "french", "korean"])
    data = st.selectbox("Datasets", ["mmlu","hellaswag","openbookqa","arc_easy","arc_challenge","winogrande","custom"])
    cache = st.checkbox("Cache Optimization")
    dbl = st.checkbox("Double Buffering")
    temp = st.slider("Temperature", 0.0, 1.0, 0.7)
    run_btn = st.button("🚀 Run", use_container_width=True)

# ---------- Panel 2 : Log ----------
with log_col:
    st.markdown("### 🖥️ Execution Log")
    st.divider()
    log_box = st.empty()

# ---------- Panel 3 : Results ----------
with res_col:
    st.markdown("### 📊 Results")
    st.divider()
    res_container = st.container()

# ---------- Action ----------
if run_btn:
    df = run(m, lang, data, cache, dbl, temp, log_box)
    with res_container:
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download Results",
                           data=df.to_csv(index=False, encoding="utf-8-sig"),
                           file_name="results.csv",
                           mime="text/csv")
else:
    log_box.info("Press ▶️ Run to start.")
    with res_container:
        st.info("Results will appear here — press Run to generate sample outputs.")
        st.download_button("📥 Download Sample Results",
                           data=pd.DataFrame(sample_data).to_csv(index=False, encoding="utf-8-sig"),
                           file_name="sample_results.csv",
                           mime="text/csv")
