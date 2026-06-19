import json
from ollama_client import generate, generate_async

EVAL_PROMPT_TEMPLATE = """
You are an expert AI evaluator. Rate the quality of the AI response to the user's prompt on three dimensions: Correctness, Clarity, and Completeness.
Each dimension must be scored on a scale from 1.0 to 10.0 (where 10.0 is perfect).

User Prompt:
{prompt}

AI Response to evaluate:
{response}

Return ONLY a valid JSON object matching this schema:
{{
  "correctness": 8.5,
  "clarity": 9.0,
  "completeness": 7.5,
  "overall_score": 8.33,
  "feedback": "A short sentence explaining the score reasoning."
}}
"""

def evaluate_quality(prompt, response_text, judge_model="llama3.2:3b"):
    eval_prompt = EVAL_PROMPT_TEMPLATE.format(prompt=prompt, response=response_text)
    try:
        # Enforce json output format
        result_payload = generate(eval_prompt, model=judge_model, format="json")
        data = json.loads(result_payload["response"])
        return {
            "correctness": float(data.get("correctness", 0.0)),
            "clarity": float(data.get("clarity", 0.0)),
            "completeness": float(data.get("completeness", 0.0)),
            "overall_score": float(data.get("overall_score", 0.0)),
            "feedback": str(data.get("feedback", ""))
        }
    except Exception as e:
        return {
            "correctness": 0.0,
            "clarity": 0.0,
            "completeness": 0.0,
            "overall_score": 0.0,
            "feedback": f"Evaluation failed: {str(e)}"
        }

async def evaluate_quality_async(prompt, response_text, judge_model="llama3.2:3b"):
    eval_prompt = EVAL_PROMPT_TEMPLATE.format(prompt=prompt, response=response_text)
    try:
        result_payload = await generate_async(eval_prompt, model=judge_model, format="json")
        data = json.loads(result_payload["response"])
        return {
            "correctness": float(data.get("correctness", 0.0)),
            "clarity": float(data.get("clarity", 0.0)),
            "completeness": float(data.get("completeness", 0.0)),
            "overall_score": float(data.get("overall_score", 0.0)),
            "feedback": str(data.get("feedback", ""))
        }
    except Exception as e:
        return {
            "correctness": 0.0,
            "clarity": 0.0,
            "completeness": 0.0,
            "overall_score": 0.0,
            "feedback": f"Evaluation failed: {str(e)}"
        }
