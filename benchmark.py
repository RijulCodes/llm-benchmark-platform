import time
import psutil

def get_ollama_memory_mb():
    try:
        mem_bytes = 0
        for proc in psutil.process_iter(['name', 'memory_info']):
            try:
                name = (proc.info['name'] or '').lower()
                if 'ollama' in name:
                    mem_bytes += proc.info['memory_info'].rss
            except Exception:
                continue
        return round(mem_bytes / (1024 * 1024), 2)
    except Exception:
        return 0.0

def measure_generation(generate_func, prompt):
    start = time.time()
    try:
        mem_before = get_ollama_memory_mb()
        response_data = generate_func(prompt)
        end = time.time()
        mem_after = get_ollama_memory_mb()
        peak_mem = max(mem_before, mem_after)

        if isinstance(response_data, dict):
            response_text = response_data.get("response", "")
            
            # Ollama returns durations in nanoseconds (1s = 1e9 ns)
            total_duration_ns = response_data.get("total_duration", 0)
            eval_duration_ns = response_data.get("eval_duration", 0)
            eval_count = response_data.get("eval_count", 0)

            if total_duration_ns > 0:
                latency = round(total_duration_ns / 1e9, 2)
            else:
                latency = round(end - start, 2)

            if eval_count > 0:
                token_count = eval_count
                if eval_duration_ns > 0:
                    tps = round(token_count / (eval_duration_ns / 1e9), 2)
                else:
                    tps = round(token_count / latency, 2) if latency > 0 else 0.0
            else:
                token_count = len(response_text.split())
                tps = round(token_count / latency, 2) if latency > 0 else 0.0
        else:
            response_text = str(response_data)
            latency = round(end - start, 2)
            token_count = len(response_text.split())
            tps = round(token_count / latency, 2) if latency > 0 else 0.0

        return {
            "response": response_text,
            "latency": latency,
            "tokens": token_count,
            "tokens_per_second": tps,
            "memory_mb": peak_mem
        }
    except Exception as e:
        return {
            "response": f"Error: {str(e)}",
            "latency": 0.0,
            "tokens": 0,
            "tokens_per_second": 0.0,
            "memory_mb": 0.0,
            "error": str(e)
        }

async def measure_generation_async(generate_func_async, prompt):
    start = time.time()
    try:
        mem_before = get_ollama_memory_mb()
        response_data = await generate_func_async(prompt)
        end = time.time()
        mem_after = get_ollama_memory_mb()
        peak_mem = max(mem_before, mem_after)

        if isinstance(response_data, dict):
            response_text = response_data.get("response", "")
            
            # Ollama returns durations in nanoseconds (1s = 1e9 ns)
            total_duration_ns = response_data.get("total_duration", 0)
            eval_duration_ns = response_data.get("eval_duration", 0)
            eval_count = response_data.get("eval_count", 0)

            if total_duration_ns > 0:
                latency = round(total_duration_ns / 1e9, 2)
            else:
                latency = round(end - start, 2)

            if eval_count > 0:
                token_count = eval_count
                if eval_duration_ns > 0:
                    tps = round(token_count / (eval_duration_ns / 1e9), 2)
                else:
                    tps = round(token_count / latency, 2) if latency > 0 else 0.0
            else:
                token_count = len(response_text.split())
                tps = round(token_count / latency, 2) if latency > 0 else 0.0
        else:
            response_text = str(response_data)
            latency = round(end - start, 2)
            token_count = len(response_text.split())
            tps = round(token_count / latency, 2) if latency > 0 else 0.0

        return {
            "response": response_text,
            "latency": latency,
            "tokens": token_count,
            "tokens_per_second": tps,
            "memory_mb": peak_mem
        }
    except Exception as e:
        return {
            "response": f"Error: {str(e)}",
            "latency": 0.0,
            "tokens": 0,
            "tokens_per_second": 0.0,
            "memory_mb": 0.0,
            "error": str(e)
        }