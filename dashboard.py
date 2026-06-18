import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="LLM Benchmark Dashboard",
    page_icon="📊",
    layout="centered"
)

# ----------------------------
# LOAD DATA
# ----------------------------

with open("results/benchmark_results.json", "r") as f:
    data = json.load(f)

rows = []

for entry in data:

    rows.append({
        "Prompt": entry["prompt"],
        "Model": "llama3.2:3b",
        "Latency": entry["llama3.2:3b"]["latency"],
        "TPS": entry["llama3.2:3b"]["tokens_per_second"],
        "Tokens": entry["llama3.2:3b"]["tokens"]
    })

    rows.append({
        "Prompt": entry["prompt"],
        "Model": "mistral:7b",
        "Latency": entry["mistral:7b"]["latency"],
        "TPS": entry["mistral:7b"]["tokens_per_second"],
        "Tokens": entry["mistral:7b"]["tokens"]
    })

df = pd.DataFrame(rows)

avg_metrics = df.groupby("Model")[["Latency", "TPS", "Tokens"]].mean()

# ----------------------------
# TITLE
# ----------------------------

st.title("LLM Benchmark Dashboard")

st.markdown(
    """
Compare performance of locally deployed LLMs using:

- FastAPI
- Ollama
- Pydantic Validation
- Automated Benchmark Suite
- Streamlit Dashboard
"""
)

# ----------------------------
# SUMMARY CARDS
# ----------------------------

st.subheader("Model Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Fastest Model",
        avg_metrics["Latency"].idxmin()
    )

with col2:
    st.metric(
        "Highest TPS",
        avg_metrics["TPS"].idxmax()
    )

with col3:
    st.metric(
        "Most Tokens",
        avg_metrics["Tokens"].idxmax()
    )

# ----------------------------
# AVERAGE METRICS
# ----------------------------

st.subheader("Average Metrics")

st.dataframe(
    avg_metrics.round(2),
    use_container_width=True
)

# ----------------------------
# CHARTS
# ----------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("Latency")

    fig, ax = plt.subplots(figsize=(4, 3))

    avg_metrics["Latency"].plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Seconds")

    st.pyplot(fig)

with col2:

    st.subheader("TPS")

    fig, ax = plt.subplots(figsize=(4, 3))

    avg_metrics["TPS"].plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Tokens/Sec")

    st.pyplot(fig)

# ----------------------------
# TOKENS CHART
# ----------------------------

st.subheader("Average Tokens Generated")

fig, ax = plt.subplots(figsize=(4, 3))

avg_metrics["Tokens"].plot(
    kind="bar",
    ax=ax
)

ax.set_ylabel("Tokens")

st.pyplot(fig)

# ----------------------------
# BENCHMARK TABLE
# ----------------------------

st.subheader("Detailed Benchmark Results")

st.dataframe(
    df,
    use_container_width=True
)

# ----------------------------
# INSIGHTS
# ----------------------------

st.subheader("Key Findings")

fastest_model = avg_metrics["Latency"].idxmin()
best_tps_model = avg_metrics["TPS"].idxmax()

st.success(
    f"""
Fastest Model: {fastest_model}

Highest Throughput (TPS): {best_tps_model}

Based on the benchmark suite, {best_tps_model}
delivered the best performance on the tested hardware.
"""
)