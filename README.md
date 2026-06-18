# LLM Benchmark Platform

A local LLM evaluation platform built using FastAPI and Ollama.

## Features

- Local LLM inference using Ollama
- FastAPI REST API
- Structured JSON outputs
- Pydantic validation
- Automatic retry mechanism
- Benchmarking metrics
  - Latency
  - Token count
  - Tokens per second

## Tech Stack

- Python
- FastAPI
- Ollama
- Pydantic

## Models

- Llama 3.2 3B

## Demo

### API Overview

![API Overview](assets/swagger-overview.png)

### Structured Output Validation

![Structured Output](assets/structured-output.png)

### Benchmarking Metrics

![Benchmark Results](assets/benchmark-results.png)
Measures response latency, approximate token count, and throughput (tokens/sec) for local LLM inference