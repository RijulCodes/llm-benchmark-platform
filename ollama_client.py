# pyrefly: ignore [missing-import]
import httpx

def get_installed_models():
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []

async def get_installed_models_async():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []

def generate(prompt, model="llama3.2:3b", format=None, temperature=None, max_tokens=None):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    if format:
        payload["format"] = format

    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    if options:
        payload["options"] = options

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            "http://localhost:11434/api/generate",
            json=payload
        )
        response.raise_for_status()
        return response.json()

async def generate_async(prompt, model="llama3.2:3b", format=None, temperature=None, max_tokens=None):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    if format:
        payload["format"] = format

    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    if options:
        payload["options"] = options

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json=payload
        )
        response.raise_for_status()
        return response.json()