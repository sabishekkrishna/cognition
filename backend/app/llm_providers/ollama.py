import httpx
import time
import json
from typing import List, Dict, Any, AsyncGenerator, Optional, Tuple
from .base import BaseLLMProvider, LLMProviderResponse, join_url_path

class OllamaProvider(BaseLLMProvider):
    """
    Adapter for Ollama local LLM daemon (default: http://127.0.0.1:11434).
    Uses Ollama native REST API endpoints:
    - GET /api/tags: Lists installed models
    - POST /api/generate: Prompt completion (streaming or non-streaming)
    """
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout_sec: float = 45.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_sec

    def set_base_url(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def get_provider_name(self) -> str:
        return "ollama"

    async def check_health(self) -> Tuple[bool, str]:
        url = join_url_path(self.base_url, "/api/tags")
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(url)
                content_type = res.headers.get("content-type", "").lower()
                body_start = res.text[:200].strip().lower()

                if "text/html" in content_type or body_start.startswith("<!doctype") or body_start.startswith("<html"):
                    return (
                        False,
                        f"Server at {self.base_url} is serving frontend HTML ('text/html'), not an Ollama inference API. "
                        "It appears to be a web/frontend development server. To connect to an LLM, point to an inference daemon "
                        "(e.g., Ollama on port 11434) or configure an LLM proxy."
                    )

                if res.status_code == 200:
                    return True, "Ollama daemon connected and operational."
                return False, f"Ollama returned HTTP status {res.status_code}."
        except httpx.ConnectError:
            return False, f"Connection refused at {self.base_url}. Ensure 'ollama serve' is running locally."
        except Exception as e:
            return False, f"Ollama connection error: {str(e)}"

    async def list_models(self) -> List[str]:
        url = join_url_path(self.base_url, "/api/tags")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(url)
                content_type = res.headers.get("content-type", "").lower()
                if "text/html" in content_type:
                    return []
                if res.status_code == 200:
                    data = res.json()
                    models = [m.get("name", "unknown") for m in data.get("models", [])]
                    return models
                return []
        except Exception:
            return []

    async def generate(self,
                       prompt: str,
                       model: str,
                       system_prompt: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: int = 512) -> LLMProviderResponse:
        url = join_url_path(self.base_url, "/api/generate")
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, json=payload)
                if res.status_code != 200:
                    raise RuntimeError(f"Ollama error {res.status_code}: {res.text}")
                data = res.json()
                duration_ms = (time.time() - start_time) * 1000.0
                return LLMProviderResponse(
                    provider="ollama",
                    model=model,
                    prompt=prompt,
                    response_text=data.get("response", ""),
                    duration_ms=round(duration_ms, 1),
                    total_tokens=data.get("eval_count", 0)
                )
        except httpx.ConnectError:
            raise ConnectionError(f"Could not connect to Ollama at {self.base_url}. Verify the local server is running.")

    async def generate_stream(self,
                              prompt: str,
                              model: str,
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 512) -> AsyncGenerator[str, None]:
        url = join_url_path(self.base_url, "/api/generate")
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        raise RuntimeError(f"Ollama stream error {response.status_code}")
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            if chunk:
                                yield chunk
                            if data.get("done", False):
                                break
        except httpx.ConnectError:
            yield f"[Connection Error: Ollama is unreachable at {self.base_url}]"
