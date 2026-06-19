import json
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import asyncio

from ollama_client import get_installed_models_async, generate_async
from benchmark import measure_generation_async
from quality_evaluator import evaluate_quality_async

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="LLM Evaluation Platform",
    page_icon="📊",
    layout="wide"
)

st.title("📊 LLM Inference & Evaluation Platform")

st.markdown(
    """
Evaluate and profile the quality, speed, and hardware footprint of local open-source LLMs.
"""
)

tab1, tab2 = st.tabs(["📊 Run Analytics", "⚡ Live Evaluator"])

# ----------------------------
# TAB 1: RUN ANALYTICS
# ----------------------------

with tab1:
    results_path = "results/benchmark_results.json"

    if not os.path.exists(results_path):
        st.info(
            f"📊 **No benchmark history found yet.**\n\n"
            f"Please run the benchmark suite to collect history:\n"
            f"```bash\npython benchmark_suite.py\n```\n"
            f"Or proceed to the **Live Evaluator** tab to execute live comparisons."
        )
    else:
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            st.error(f"Failed to load benchmark results: {e}")
            st.stop()

        if not data:
            st.warning("The benchmark results file is empty.")
        else:
            rows = []
            for entry in data:
                prompt = entry.get("prompt", "")
                category = entry.get("category", "General")
                for key, val in entry.items():
                    if key in ("prompt", "category"):
                        continue
                    if isinstance(val, dict) and "latency" in val:
                        rows.append({
                            "Prompt": prompt,
                            "Category": category,
                            "Model": key,
                            "Latency": val.get("latency", 0.0),
                            "TPS": val.get("tokens_per_second", 0.0),
                            "Tokens": val.get("tokens", 0),
                            "RAM (MB)": val.get("memory_mb", 0.0),
                            "Quality Score": val.get("quality_score", 0.0)
                        })

            if not rows:
                st.error("No valid model benchmarks found in the results file.")
            else:
                df = pd.DataFrame(rows)
                avg_metrics = df.groupby("Model")[["Latency", "TPS", "Tokens", "RAM (MB)", "Quality Score"]].mean()

                # SUMMARY CARDS
                st.subheader("Model Summary")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    fastest_model = avg_metrics["Latency"].idxmin() if not avg_metrics.empty else "N/A"
                    st.metric("Fastest Model", fastest_model)

                with col2:
                    highest_tps_model = avg_metrics["TPS"].idxmax() if not avg_metrics.empty else "N/A"
                    st.metric("Highest Throughput (TPS)", highest_tps_model)

                with col3:
                    lowest_ram_model = avg_metrics["RAM (MB)"].idxmin() if not avg_metrics.empty else "N/A"
                    st.metric("Lowest RAM Usage", f"{lowest_ram_model}")

                with col4:
                    highest_quality_model = avg_metrics["Quality Score"].idxmax() if not avg_metrics.empty else "N/A"
                    st.metric("Highest Quality Score", highest_quality_model)

                # AVERAGE METRICS
                st.subheader("Average Performance Metrics")
                st.dataframe(
                    avg_metrics.round(2),
                    use_container_width=True
                )

                # CHARTS
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Speed vs. Quality Trade-off")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    for model_name, group in df.groupby("Model"):
                        ax.scatter(
                            group["TPS"].mean(), 
                            group["Quality Score"].mean(), 
                            s=150, 
                            label=model_name,
                            alpha=0.8,
                            edgecolors="black"
                        )
                    ax.set_xlabel("Throughput (Tokens/Second)")
                    ax.set_ylabel("Quality Score (1-10)")
                    ax.grid(True, linestyle="--", alpha=0.5)
                    ax.legend()
                    st.pyplot(fig)

                with col2:
                    st.subheader("Memory Utilization (Peak RAM)")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    avg_metrics["RAM (MB)"].plot(
                        kind="bar",
                        ax=ax,
                        color="#d62728",
                        edgecolor="black"
                    )
                    ax.set_ylabel("MB")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                col3, col4 = st.columns(2)

                with col3:
                    st.subheader("Average Latency")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    avg_metrics["Latency"].plot(
                        kind="bar",
                        ax=ax,
                        color="#1f77b4",
                        edgecolor="black"
                    )
                    ax.set_ylabel("Seconds")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                with col4:
                    st.subheader("Throughput (TPS)")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    avg_metrics["TPS"].plot(
                        kind="bar",
                        ax=ax,
                        color="#2ca02c",
                        edgecolor="black"
                    )
                    ax.set_ylabel("Tokens/Sec")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                # CATEGORY BREAKDOWN SECTION
                if "Category" in df.columns:
                    st.markdown("---")
                    st.subheader("📚 Performance by Prompt Category")
                    
                    category_metrics = df.groupby(["Model", "Category"])[["Latency", "TPS", "Quality Score"]].mean().reset_index()
                    
                    col_cat1, col_cat2 = st.columns([1, 1])
                    
                    with col_cat1:
                        st.dataframe(
                            category_metrics.round(2),
                            use_container_width=True
                        )
                    
                    with col_cat2:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        pivot_df = category_metrics.pivot(index="Category", columns="Model", values="Quality Score")
                        pivot_df.plot(kind="bar", ax=ax, edgecolor="black")
                        ax.set_ylabel("Quality Score (1-10)")
                        ax.set_xlabel("Category")
                        ax.grid(True, linestyle="--", alpha=0.5)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)

                # BENCHMARK TABLE
                st.subheader("Detailed Benchmark History")
                st.dataframe(
                    df,
                    use_container_width=True
                )

# ----------------------------
# TAB 2: LIVE EVALUATOR
# ----------------------------

with tab2:
    st.subheader("Live LLM Interactive Evaluator")
    st.write("Execute benchmarking runs dynamically with slider tuning and real-time response quality evaluations.")

    # Dynamically query models using async wrapper
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        installed = loop.run_until_complete(get_installed_models_async())
        loop.close()
    except Exception:
        installed = []

    if not installed:
        st.warning("⚠️ Local Ollama registry not detected or offline. Using fallback defaults.")
        installed = ["llama3.2:3b", "mistral:7b", "phi3:latest"]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Configurations")
        selected_models = st.multiselect(
            "Select Models to Compare", 
            options=installed, 
            default=installed[:2] if len(installed) >= 2 else installed
        )
        
        temperature = st.slider(
            "Temperature (Creativity)", 
            min_value=0.0, 
            max_value=1.5, 
            value=0.7, 
            step=0.1
        )
        
        max_tokens = st.slider(
            "Max Tokens (Response size limit)", 
            min_value=10, 
            max_value=2048, 
            value=256, 
            step=10
        )
        
        enable_eval = st.checkbox("Enable LLM-as-a-Judge Grading", value=True)
        judge_model = st.selectbox(
            "Select Judge Model", 
            options=installed, 
            index=0
        )

    with col2:
        st.markdown("### Query Input")
        prompt_input = st.text_area(
            "Enter Test Prompt", 
            value="Compare binary search and sequential search.", 
            height=120
        )
        
        run_btn = st.button("🚀 Run Live Evaluation", type="primary")

    if run_btn:
        if not selected_models:
            st.error("Please select at least one model to benchmark.")
        elif not prompt_input:
            st.error("Please enter a test prompt.")
        else:
            st.info("Benchmarking models in real time...")
            progress_bar = st.progress(0.0)
            
            async def run_live_compare():
                results = {}
                for idx, model in enumerate(selected_models):
                    st.write(f"⏳ Querying **{model}**...")
                    
                    # Core generator function
                    benchmark_result = await measure_generation_async(
                        lambda p, m=model: generate_async(p, m, temperature=temperature, max_tokens=max_tokens),
                        prompt_input
                    )
                    
                    quality_score = 0.0
                    quality_feedback = "N/A"
                    
                    if enable_eval and "error" not in benchmark_result:
                        st.write(f"⚖️ Grading quality of **{model}** using **{judge_model}**...")
                        quality_details = await evaluate_quality_async(
                            prompt_input, 
                            benchmark_result["response"], 
                            judge_model=judge_model
                        )
                        quality_score = quality_details.get("overall_score", 0.0)
                        quality_feedback = quality_details.get("feedback", "None")

                    results[model] = {
                        "latency": benchmark_result["latency"],
                        "tokens_per_second": benchmark_result["tokens_per_second"],
                        "memory_mb": benchmark_result.get("memory_mb", 0.0),
                        "quality_score": quality_score,
                        "feedback": quality_feedback,
                        "response": benchmark_result["response"],
                        "error": benchmark_result.get("error")
                    }
                    progress_bar.progress((idx + 1) / len(selected_models))
                return results

            # Run evaluation loop
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                compare_results = loop.run_until_complete(run_live_compare())
                loop.close()
                st.success("🎉 Benchmark completed successfully!")

                # Render Results Table
                st.markdown("### Performance & Quality Matrix")
                compare_rows = []
                for model, res in compare_results.items():
                    if res.get("error"):
                        compare_rows.append({
                            "Model": model,
                            "Latency (s)": "Error",
                            "TPS": "Error",
                            "RAM (MB)": "Error",
                            "Quality Score": "Error",
                            "Judge Feedback": res["error"]
                        })
                    else:
                        compare_rows.append({
                            "Model": model,
                            "Latency (s)": f"{res['latency']}s",
                            "TPS": res["tokens_per_second"],
                            "RAM (MB)": f"{res['memory_mb']} MB",
                            "Quality Score": f"{res['quality_score']}/10.0",
                            "Judge Feedback": res["feedback"]
                        })
                
                st.dataframe(pd.DataFrame(compare_rows), use_container_width=True)

                # Show responses in expander cards
                st.markdown("### Detailed Outputs")
                for model, res in compare_results.items():
                    with st.expander(f"Show response from {model}"):
                        if res.get("error"):
                            st.error(res["error"])
                        else:
                            st.write(res["response"])

            except Exception as e:
                st.error(f"Failed executing live benchmark: {e}")