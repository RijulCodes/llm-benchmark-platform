# LLM Benchmark Platform

A local LLM evaluation platform built using **FastAPI**, **Ollama**, **Pydantic**, and **Streamlit** for benchmarking and analyzing open-source language models.

---

## Features

* Local LLM inference using Ollama
* FastAPI REST API
* Structured JSON generation
* Pydantic schema validation
* Automatic retry mechanism for invalid outputs
* Multi-model benchmarking
* Automated benchmark suite
* Streamlit analytics dashboard
* Performance metrics collection:

  * Latency
  * Token Count
  * Tokens Per Second (TPS)

---

## Tech Stack

* Python
* FastAPI
* Ollama
* Pydantic
* Streamlit
* Matplotlib
* Pandas

---

## Models Benchmarked

* Llama 3.2 3B
* Mistral 7B

---

## Architecture

```text
User Request
      │
      ▼
 FastAPI Endpoint
      │
      ▼
 Ollama Client
      │
      ▼
 Local LLM
      │
      ▼
 JSON Validation
 (Pydantic)
      │
      ▼
 Retry Logic
      │
      ▼
 Benchmark Metrics
      │
      ▼
 Streamlit Dashboard
```

---

## Benchmark Results

| Model        | Avg Latency (s) | Avg TPS |
| ------------ | --------------- | ------- |
| Llama 3.2 3B | 42.32           | 8.69    |
| Mistral 7B   | 50.12           | 4.38    |

### Key Findings

* Llama 3.2 3B achieved lower latency on the tested hardware.
* Llama 3.2 3B achieved nearly 2× higher throughput (TPS).
* Mistral 7B generated shorter responses but was slower overall.
* Local deployment allows benchmarking without relying on external APIs.

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

* Response latency
* Token count
* Tokens per second (TPS)

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

```bash
pip install -r requirements.txt
```

---

## Running the API

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Running the Dashboard

```bash
streamlit run dashboard.py
```


---

## Author

**Rijul Yadav**

