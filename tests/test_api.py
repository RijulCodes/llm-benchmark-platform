import pytest
from fastapi.testclient import TestClient
from app import app
from unittest.mock import AsyncMock, patch, ANY

client = TestClient(app)

@patch("app.get_installed_models_async")
def test_home(mock_get_models):
    mock_get_models.return_value = ["llama3.2:3b", "mistral:7b"]
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "running", "detected_models": ["llama3.2:3b", "mistral:7b"]}

@patch("app.get_installed_models_async")
@patch("app.generate_async")
def test_generate(mock_generate, mock_get_models):
    mock_get_models.return_value = ["llama3.2:3b"]
    mock_generate.return_value = {"response": "This is generated text"}
    
    response = client.get("/generate?prompt=hello&temperature=0.5&max_tokens=100")
    assert response.status_code == 200
    assert response.json() == {"response": "This is generated text"}
    mock_generate.assert_called_once_with("hello", model="llama3.2:3b", temperature=0.5, max_tokens=100)

@patch("app.get_installed_models_async")
@patch("app.generate_async")
def test_summarize(mock_generate, mock_get_models):
    mock_get_models.return_value = ["llama3.2:3b"]
    mock_generate.return_value = {
        "response": '{"title": "Quantum Physics", "summary": "Study of tiny particles", "difficulty": "hard"}'
    }
    
    response = client.get("/summarize?topic=quantum&temperature=0.0")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["title"] == "Quantum Physics"
    assert data["data"]["difficulty"] == "hard"
    mock_generate.assert_called_with(
        ANY, model="llama3.2:3b", format="json", temperature=0.0, max_tokens=None
    )

@patch("app.get_installed_models_async")
@patch("app.measure_generation_async")
def test_compare(mock_measure, mock_get_models):
    mock_get_models.return_value = ["llama3.2:3b", "mistral:7b"]
    mock_measure.side_effect = [
        {"latency": 1.2, "tokens": 10, "tokens_per_second": 8.33, "memory_mb": 150.0},
        {"latency": 2.5, "tokens": 12, "tokens_per_second": 4.8, "memory_mb": 250.0}
    ]
    
    response = client.get("/compare?prompt=hello&models=llama3.2:3b,mistral:7b&temperature=0.8")
    assert response.status_code == 200
    data = response.json()
    assert "llama3.2:3b" in data
    assert "mistral:7b" in data
    assert data["llama3.2:3b"]["latency"] == 1.2
    assert data["llama3.2:3b"]["memory_mb"] == 150.0
    assert data["mistral:7b"]["latency"] == 2.5
    assert data["mistral:7b"]["memory_mb"] == 250.0
