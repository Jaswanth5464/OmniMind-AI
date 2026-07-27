import time
from langchain_ollama import OllamaLLM

_global_llm = None
_llm_initialized_at = None


def init_llm():

    global _global_llm, _llm_initialized_at

    if _global_llm is not None:
        return _global_llm

    try:
        print("→ Attempting to initialize Ollama LLM (tinyllama:1.1b)...")
        _global_llm = OllamaLLM(model="tinyllama:1.1b")
        _llm_initialized_at = time.time()
        print("Ollama initialized.")
        return _global_llm
    except Exception as e:
        print("Could not initialize Ollama LLM:", e)
        _global_llm = None
        return None


def _safe_llm_call(llm_obj, prompt: str) -> str:

    if llm_obj is None:
        raise RuntimeError("LLM not available")

    try:
        resp = llm_obj.invoke(prompt)
    except Exception as e:
        # Fallback for unexpected API changes
        print(f"LLM Error: {e}")
        resp = str(e)

    if isinstance(resp, str):
        return resp
    if hasattr(resp, "text"):
        return resp.text
    if isinstance(resp, dict) and "text" in resp:
        return resp["text"]

    return str(resp)
