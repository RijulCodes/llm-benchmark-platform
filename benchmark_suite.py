import json
import os
import time

from ollama_client import generate, get_installed_models
from benchmark import measure_generation
from quality_evaluator import evaluate_quality

def main():
    installed = get_installed_models()
    MODELS = installed if installed else ["llama3.2:3b", "mistral:7b"]
    print(f"Detected models: {installed}")
    print(f"Beginning benchmark with models: {MODELS}\n")

    # Use Llama 3.2 3b as the default judge, or fall back to the first available model
    judge_model = "llama3.2:3b" if "llama3.2:3b" in MODELS else MODELS[0]
    print(f"Using judge model for quality evaluation: {judge_model}")

    # Try to load prompts, fallback to basic list if not found
    try:
        with open("prompts/test_prompts.json", "r") as f:
            prompts = json.load(f)
    except FileNotFoundError:
        print("Warning: prompts/test_prompts.json not found. Using fallback prompts.")
        prompts = [
            "Explain quantum computing in one sentence.",
            "What is the capital of France?",
            "Write a quick python function to calculate fibonacci numbers."
        ]

    results = []

    for i, prompt in enumerate(prompts, start=1):
        print(f"\nPrompt {i}/{len(prompts)}")
        print(f"Prompt content: {prompt}")

        prompt_result = {
            "prompt": prompt
        }

        for model in MODELS:
            print(f"Testing {model}...")

            benchmark = measure_generation(
                lambda p, m=model: generate(p, m),
                prompt
            )

            if "error" in benchmark:
                print(f"  [ERROR] {model} failed: {benchmark['error']}")
                quality_score = 0.0
                quality_details = {}
            else:
                print(f"  [SUCCESS] {model} - Latency: {benchmark['latency']}s, TPS: {benchmark['tokens_per_second']}, Peak RAM: {benchmark.get('memory_mb', 0.0)} MB")
                
                # Evaluate the response quality using our judge model
                print(f"  Evaluating quality of {model} response using {judge_model}...")
                quality_details = evaluate_quality(prompt, benchmark["response"], judge_model=judge_model)
                quality_score = quality_details.get("overall_score", 0.0)
                print(f"  [QUALITY] Score: {quality_score}/10.0 (Reason: {quality_details.get('feedback', 'None')})")

            prompt_result[model] = {
                "latency": benchmark["latency"],
                "tokens": benchmark["tokens"],
                "tokens_per_second": benchmark["tokens_per_second"],
                "memory_mb": benchmark.get("memory_mb", 0.0),
                "quality_score": quality_score,
                "quality_details": quality_details
            }

        results.append(prompt_result)

    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nBenchmark completed and results saved to results/benchmark_results.json.")

    # ----------------------------
    # GENERATE MARKDOWN REPORT (Option D)
    # ----------------------------
    averages = {}
    for model in MODELS:
        latencies = [r[model]["latency"] for r in results if model in r and "latency" in r[model]]
        tps_vals = [r[model]["tokens_per_second"] for r in results if model in r and "tokens_per_second" in r[model]]
        rams = [r[model]["memory_mb"] for r in results if model in r and "memory_mb" in r[model]]
        qualities = [r[model]["quality_score"] for r in results if model in r and "quality_score" in r[model]]
        
        if latencies:
            averages[model] = {
                "avg_latency": round(sum(latencies) / len(latencies), 2),
                "avg_tps": round(sum(tps_vals) / len(tps_vals), 2),
                "avg_ram": round(sum(rams) / len(rams), 2),
                "avg_quality": round(sum(qualities) / len(qualities), 2)
            }

    if averages:
        fastest_model = min(averages.keys(), key=lambda k: averages[k]["avg_latency"])
        best_tps_model = max(averages.keys(), key=lambda k: averages[k]["avg_tps"])
        lowest_ram_model = min(averages.keys(), key=lambda k: averages[k]["avg_ram"])
        best_quality_model = max(averages.keys(), key=lambda k: averages[k]["avg_quality"])
        
        report_time = time.strftime("%Y-%m-%d %H:%M:%S")
        report_content = f"""# LLM Benchmark & Evaluation Report

Generated on: `{report_time}`

## Summary Performance Matrix

| Model | Avg Latency (s) | Avg Throughput (TPS) | Avg Peak Memory (MB) | Avg Quality Score (1-10) |
| :--- | :---: | :---: | :---: | :---: |
"""
        for model, metrics in averages.items():
            report_content += f"| **{model}** | {metrics['avg_latency']}s | {metrics['avg_tps']} tokens/s | {metrics['avg_ram']} MB | {metrics['avg_quality']}/10.0 |\n"
            
        report_content += f"""
## Key Insights

* 🚀 **Fastest Inference**: `{fastest_model}` averaged `{averages[fastest_model]['avg_latency']}s` latency.
* 📈 **Highest Throughput**: `{best_tps_model}` generated `{averages[best_tps_model]['avg_tps']} tokens/s`.
* 💾 **Lowest Memory Footprint**: `{lowest_ram_model}` consumed peak `{averages[lowest_ram_model]['avg_ram']} MB` RAM.
* ⚖️ **Best Output Alignment**: `{best_quality_model}` received a quality grade of `{averages[best_quality_model]['avg_quality']}/10.0` from the LLM Judge (`{judge_model}`).
"""
    else:
        report_content = "# LLM Benchmark Report\n\nNo benchmark data found."

    with open("results/benchmark_report.md", "w") as f:
        f.write(report_content)

    print("Markdown evaluation report generated at results/benchmark_report.md.")

if __name__ == "__main__":
    main()