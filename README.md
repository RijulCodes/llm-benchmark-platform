# LLM Benchmark Platform

A local LLM evaluation platform built using **FastAPI**, **Ollama**, **Pydantic**, and **Streamlit** for benchmarking and analyzing open-source language models.

---

## Features

* **Asynchronous Web Service**: Fully async routing backend (`async def` using FastAPI & `httpx`) to handle heavy LLM inference queries concurrently without blocking threads.
* **Systems Memory Profiling**: Tracks and logs process-level peak RAM utilization (Resident Set Size via `psutil`) for background model servers during active inference runs.
* **LLM-as-a-Judge Evaluation**: Structured quality grading prompt framework (grading Correctness, Clarity, and Completeness) utilizing local models as judges.
* **Streamlit Interactive UI**: Two-tab dashboard featuring visual run history graphs (Speed vs. Quality scatter plots, RAM usage) and a **Live Evaluator** tab with slider controls for Temperature and Max Tokens.
* **Automated Reports**: Generates a clean markdown summary report (`results/benchmark_report.md`) on benchmark completion.
* **Engine-Level Metrics**: Fetches exact generated token counts (`eval_count`) and precise durations directly from Ollama's response metadata.
* **Robust Software Standards**: Structured JSON mode validations via Pydantic, error handlers on endpoints, a mock unit test suite, and GitHub Actions CI workflow configuration.

---

## Tech Stack

* Python
* FastAPI (Async Web Framework)
* Ollama (Local Model Engine)
* Pydantic (Data Validation Schema)
* Streamlit (Inference Dashboard)
* Matplotlib (Performance Visualizations)
* Pandas (Data Wrangling)
* psutil (Process Memory Profiling)
* httpx (Async HTTP Client)
* pytest & pytest-asyncio (Unit & Integration Tests)

---

## Models Benchmarked

* Llama 3.2 3B
* Mistral 7B

---

## Architecture

```mermaid
graph TD
    User([User Request]) --> FastAPI[FastAPI App app.py]
    FastAPI --> Route1[/generate /summarize /compare /benchmark]
    Route1 --> OllamaClient[Ollama Client ollama_client.py]
    OllamaClient --> LocalEngine[(Local Ollama Engine)]
    LocalEngine --> Response[Model Response]
    Response --> PydanticValidation{Pydantic Schema Validation}
    PydanticValidation -- Valid --> Metrics[Benchmark Profiler benchmark.py]
    PydanticValidation -- Invalid --> Retry[Validation Retry Loop + Prompt Update]
    Retry --> OllamaClient
    Metrics --> ResultsDB[(Results JSON Database)]
    ResultsDB --> Streamlit[Streamlit Dashboard dashboard.py]
    ResultsDB --> MDReport[Markdown Reports benchmark_report.md]
```

---

## Motivation & Performance Insights

### Why Cognitive Quality Benchmarking?
Most open-source LLM benchmarking platforms focus purely on hardware metrics like latency and throughput (tokens per second). However, a model that generates garbage text at 100 TPS isn't actually better than a model that generates precise, correct text at 20 TPS.

To bridge this gap, this platform introduces an **LLM-as-a-Judge Cognitive Quality Evaluator** (`quality_evaluator.py`). It rates responses from 1.0 to 10.0 along three cognitive dimensions:
1. **Correctness**: Technical accuracy and alignment with truth.
2. **Clarity**: Structural readability and tone.
3. **Completeness**: How thoroughly the response answers all constraints in the prompt.

### Summary Metrics

| Model | Avg Latency (s) | Avg Throughput (TPS) | Avg Peak Memory (MB) | Avg Quality Score (1-10) |
| :--- | :---: | :---: | :---: | :---: |
| **phi3:latest** | 34.62s | 9.14 tokens/s | 38.25 MB | 7.31/10.0 |
| **mistral:7b** | 61.39s | 5.58 tokens/s | 41.62 MB | 8.33/10.0 |
| **llama3.2:3b** | 26.37s | 13.31 tokens/s | 47.39 MB | 8.36/10.0 |

### Key Findings
* 🚀 **Throughput Efficiency**: `llama3.2:3b` offers the fastest overall inference speed, running at **13.31 TPS**—nearly **2.4× faster** than `mistral:7b`.
* ⚖️ **Speed vs. Quality Trade-off**: `llama3.2:3b` maintains the highest overall quality score (**8.36/10.0**), while `phi3:latest` performs very well on coding tasks but struggles on long-context algorithmic comparisons (lowering its overall average to **7.31/10.0**).
* 💾 **Hardware Footprint**: `phi3:latest` is highly resource-efficient, demonstrating the **lowest average peak RAM footprint** (**38.25 MB**), which is **19.3% lower** than `llama3.2:3b`.

---

## Dashboard

![Dashboard](assets/dashboard.png)

![Dashboard](assets/dashboard2.png)

The Streamlit dashboard provides:

* Average latency comparison
* Throughput (TPS) comparison
* Token generation statistics
* Model performance summaries
* Detailed benchmark results

---

## API Demonstration

### API Overview

![API Overview](assets/swagger-overview.png)

### Structured Output Validation

![Structured Output](assets/structured-output.png)

### Benchmarking Metrics

![Benchmark Results](assets/benchmark-results.png)

Measures:

* Response latency (seconds)
* Token count
* Tokens per second (TPS)
* Peak memory utilization (MB)
* Cognitive quality alignment (1-10 score)

---

## Installation

### Clone Repository

```bash
git clone https://github.com/RijulCodes/llm-benchmark-platform.git
cd llm-benchmark-platform
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

Install the core application packages:
```bash
pip install -r requirements.txt
```

*(Optional)* Install development and testing dependencies:
```bash
pip install -r requirements-dev.txt
```

---

## Step-by-Step Running Guide

To run the benchmarking platform successfully from scratch:

### 1. Run the FastAPI Server
Launch the local API server using Uvicorn:
```bash
uvicorn app:app --reload
```
You can view the interactive swagger API documentation at:
```text
http://127.0.0.1:8000/docs
```

### 2. Run the Benchmark Suite
> [!NOTE]
> The benchmark results folder `results/` is excluded from git tracking. You **must** run the benchmark suite to generate the results file locally before launching the dashboard.

Execute the automated benchmark run:
```bash
python benchmark_suite.py
```
This queries the local Ollama instance using all detected models, saves raw metrics to `results/benchmark_results.json`, and outputs a polished Markdown analysis report at `results/benchmark_report.md` (which details per-model and per-category speed, RAM, and quality statistics).

### 3. Run the Analytics Dashboard
Launch the Streamlit visualization dashboard:
```bash
streamlit run dashboard.py
```

---

## Running Unit Tests

To execute the test suite (endpoints testing with mocked LLM queries, and math validation for the benchmark timings):
```bash
pytest
```

---

## Author

**Rijul Yadav**

