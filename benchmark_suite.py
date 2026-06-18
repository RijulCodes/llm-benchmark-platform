import json

from ollama_client import generate
from benchmark import measure_generation


MODELS = [
    "llama3.2:3b",
    "mistral:7b"
]


with open("prompts/test_prompts.json", "r") as f:
    prompts = json.load(f)


results = []


for i, prompt in enumerate(prompts, start=1):

    print(f"\nPrompt {i}/{len(prompts)}")
    print(prompt)

    prompt_result = {
        "prompt": prompt
    }

    for model in MODELS:

        print(f"Testing {model}...")

        benchmark = measure_generation(
            lambda p: generate(p, model),
            prompt
        )
        prompt_result[model] = {
            "latency": benchmark["latency"],
            "tokens": benchmark["tokens"],
            "tokens_per_second": benchmark["tokens_per_second"]
        }

    results.append(prompt_result)


with open("results/benchmark_results.json", "w") as f:
    json.dump(results, f, indent=4)


print("Benchmark completed.")