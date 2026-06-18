from fastapi import FastAPI
from ollama_client import generate
from schemas import ConceptSummary
from benchmark import measure_generation
import json

app = FastAPI()

@app.get("/benchmark")
def benchmark(
    prompt: str,
    model: str = "llama3.2:3b"
):

    result = measure_generation(
        lambda p: generate(p, model),
        prompt
    )

    return result

@app.get("/")
def home():
    return {"status": "running"}


@app.get("/generate")
def ask(prompt: str):
    answer = generate(prompt)

    return {
        "response": answer
    }


@app.get("/summarize")
def summarize(topic: str):

    prompt = f"""
You are a JSON API.

Return ONLY valid JSON.

Required schema:

{{
    "title":"",
    "summary":"",
    "difficulty":"easy|medium|hard"
}}

Topic:
{topic}
"""

    for attempt in range(3):

        response = generate(prompt)

        print(f"\nATTEMPT {attempt+1}")
        print(response)

        try:
            data = json.loads(response)

            result = ConceptSummary.model_validate(data)

            return {
                "success": True,
                "attempt": attempt + 1,
                "data": result.model_dump()
            }

        except Exception as e:

            prompt += """

Your previous response failed validation.

Return ONLY JSON matching:

{
    "title":"",
    "summary":"",
    "difficulty":"easy|medium|hard"
}
"""

    return {
        "success": False,
        "error": "Failed after 3 attempts"
    }

@app.get("/compare")
def compare(prompt: str):

    models = [
        "llama3.2:3b",
        "mistral:7b"
    ]

    results = {}

    for model in models:

        benchmark_result = measure_generation(
            lambda p: generate(p, model),
            prompt
        )

        results[model] = {
            "latency": benchmark_result["latency"],
            "tokens": benchmark_result["tokens"],
            "tokens_per_second": benchmark_result["tokens_per_second"]
        }

    return results