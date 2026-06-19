import pytest
from benchmark import measure_generation, measure_generation_async
from quality_evaluator import evaluate_quality, evaluate_quality_async
from unittest.mock import patch

def test_measure_generation_sync_success():
    def mock_generate(prompt):
        return {
            "response": "Hello world from Ollama",
            "total_duration": 1500000000,  # 1.5s
            "eval_duration": 1000000000,   # 1.0s
            "eval_count": 10
        }
    
    result = measure_generation(mock_generate, "Test prompt")
    assert result["response"] == "Hello world from Ollama"
    assert result["latency"] == 1.5
    assert result["tokens"] == 10
    assert result["tokens_per_second"] == 10.0
    assert "memory_mb" in result

def test_measure_generation_sync_fallback():
    def mock_generate(prompt):
        return "This is a simple text response"
    
    result = measure_generation(mock_generate, "Test prompt")
    assert result["response"] == "This is a simple text response"
    assert result["latency"] >= 0.0
    assert result["tokens"] == 6
    assert result["tokens_per_second"] >= 0
    assert "memory_mb" in result

@pytest.mark.asyncio
async def test_measure_generation_async_success():
    async def mock_generate_async(prompt):
        return {
            "response": "Async response works!",
            "total_duration": 2000000000,  # 2.0s
            "eval_duration": 500000000,    # 0.5s
            "eval_count": 5
        }
    
    result = await measure_generation_async(mock_generate_async, "Test prompt")
    assert result["response"] == "Async response works!"
    assert result["latency"] == 2.0
    assert result["tokens"] == 5
    assert result["tokens_per_second"] == 10.0
    assert "memory_mb" in result

@patch("quality_evaluator.generate")
def test_evaluate_quality_success(mock_generate):
    mock_generate.return_value = {
        "response": '{"correctness": 9.0, "clarity": 8.5, "completeness": 9.5, "overall_score": 9.0, "feedback": "Excellent response."}'
    }
    res = evaluate_quality("What is 2+2?", "4", judge_model="llama")
    assert res["overall_score"] == 9.0
    assert res["feedback"] == "Excellent response."

@pytest.mark.asyncio
@patch("quality_evaluator.generate_async")
async def test_evaluate_quality_async_success(mock_generate_async):
    mock_generate_async.return_value = {
        "response": '{"correctness": 8.0, "clarity": 8.0, "completeness": 8.0, "overall_score": 8.0, "feedback": "Good response."}'
    }
    res = await evaluate_quality_async("What is 3+3?", "6", judge_model="llama")
    assert res["overall_score"] == 8.0
    assert res["feedback"] == "Good response."
