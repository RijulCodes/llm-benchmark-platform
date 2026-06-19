# pyrefly: ignore [missing-import]
from fastapi import FastAPI, Query
from ollama_client import generate_async, get_installed_models_async
from schemas import ConceptSummary
from benchmark import measure_generation_async
from quality_evaluator import evaluate_quality_async
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

async def get_default_model():
    installed = await get_installed_models_async()
    return installed[0] if installed else "llama3.2:3b"

@app.get("/benchmark")
async def benchmark(
    prompt: str,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    evaluate: bool = False,
    judge_model: str = None
):
    if not model:
        model = await get_default_model()
    result = await measure_generation_async(
        lambda p: generate_async(p, model, temperature=temperature, max_tokens=max_tokens),
        prompt
    )
    
    if evaluate:
        eval_judge = judge_model if judge_model else model
        quality = await evaluate_quality_async(prompt, result["response"], judge_model=eval_judge)
        result["quality"] = quality
        
    return result

@app.get("/")
async def home():
    installed = await get_installed_models_async()
    return {"status": "running", "detected_models": installed}

@app.get("/generate")
async def ask(
    prompt: str,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None
):
    if not model:
        model = await get_default_model()
    try:
        response_payload = await generate_async(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
        answer = response_payload["response"]
    except Exception as e:
        answer = f"Error: {str(e)}"

    return {
        "response": answer
    }

@app.get("/summarize")
async def summarize(
    topic: str,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None
):
    if not model:
        model = await get_default_model()

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
        try:
            # Query Ollama with native format constraint
            response_payload = await generate_async(
                prompt, model=model, format="json",
                temperature=temperature, max_tokens=max_tokens
            )
            response_text = response_payload["response"]

            logger.info("Attempt %d response: %s", attempt + 1, response_text)

            data = json.loads(response_text)
            result = ConceptSummary.model_validate(data)

            return {
                "success": True,
                "attempt": attempt + 1,
                "data": result.model_dump()
            }
        except Exception as e:
            logger.warning("Attempt %d failed validation: %s", attempt + 1, e)
            prompt += f"\n\nYour previous response failed validation with error: {str(e)}. Return ONLY JSON matching the schema."

    return {
        "success": False,
        "error": "Failed after 3 attempts"
    }

@app.get("/compare")
async def compare(
    prompt: str,
    models: list[str] = Query(None, description="List of models to compare (repeated param or comma-separated)"),
    temperature: float = None,
    max_tokens: int = None,
    evaluate: bool = False,
    judge_model: str = None
):
    # Support both list and comma-separated string formatting
    resolved_models = []
    if models:
        for m in models:
            if "," in m:
                resolved_models.extend([item.strip() for item in m.split(",") if item.strip()])
            else:
                resolved_models.append(m.strip())
    else:
        resolved_models = await get_installed_models_async()
        if not resolved_models:
            resolved_models = ["llama3.2:3b", "mistral:7b"]

    results = {}

    for model in resolved_models:
        benchmark_result = await measure_generation_async(
            lambda p: generate_async(p, model, temperature=temperature, max_tokens=max_tokens),
            prompt
        )

        if evaluate and "error" not in benchmark_result:
            eval_judge = judge_model if judge_model else model
            quality = await evaluate_quality_async(prompt, benchmark_result["response"], judge_model=eval_judge)
            benchmark_result["quality"] = quality

        if "error" in benchmark_result:
            results[model] = {
                "error": benchmark_result["error"],
                "latency": 0.0,
                "tokens": 0,
                "tokens_per_second": 0.0,
                "memory_mb": 0.0
            }
        else:
            results[model] = {
                "latency": benchmark_result["latency"],
                "tokens": benchmark_result["tokens"],
                "tokens_per_second": benchmark_result["tokens_per_second"],
                "memory_mb": benchmark_result.get("memory_mb", 0.0)
            }
            if "quality" in benchmark_result:
                results[model]["quality"] = benchmark_result["quality"]

    return results