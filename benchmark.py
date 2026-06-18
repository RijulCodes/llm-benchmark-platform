import time

def measure_generation(generate_func, prompt):

    start = time.time()

    response = generate_func(prompt)

    end = time.time()

    latency = round(end - start, 2)

    token_count = len(response.split())

    tps = round(token_count / latency, 2) if latency > 0 else 0

    return {
        "response": response,
        "latency": latency,
        "tokens": token_count,
        "tokens_per_second": tps
    }