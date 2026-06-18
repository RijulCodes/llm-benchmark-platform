from fastapi import FastAPI
from ollama_client import generate
from schemas import ConceptSummary
from benchmark import measure_generation
import json

app = FastAPI()

@app.get("/benchmark")
def benchmark(prompt: str):

    result = measure_generation(generate, prompt)

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