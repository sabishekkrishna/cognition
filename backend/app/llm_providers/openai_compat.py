import httpx
import time
import json
from typing import List, Dict, Any, AsyncGenerator, Optional, Tuple
from .base import BaseLLMProvider, LLMProviderResponse

class OpenAICompatProvider(BaseLLMProvider):
    """
    Adapter for LM Studio, llama.cpp server, or OpenAI-compatible local servers.
    Default endpoints:
      - LM Studio: http://127.0.0.1:1234/v1
      - llama.cpp: http://127.0.0.1:8080/v1
    """
    def __init__(self,
                 provider_id: str = "lm_studio",
                 base_url: str = "http://127.0.0.1:1234/v1",
                 timeout_sec: float = 45.0):
        self.provider_id = provider_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_sec

    def get_provider_name(self) -> str:
        return self.provider_id

    async def check_health(self) -> Tuple[bool, str]:
        url = f"{self.base_url}/models"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(url)
                if res.status_code == 200:
                    return True, f"{self.provider_id} server connected and operational."
                return False, f"Server returned HTTP status {res.status_code}."
        except httpx.ConnectError:
            return False, f"Connection refused at {self.base_url}."
        except Exception as e:
            return False, f"Connection error: {str(e)}"

    async def list_models(self) -> List[str]:
        url = f"{self.base_url}/models"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(url)
                if res.status_code == 200:
                    data = res.json()
                    models = [m.get("id", "unknown") for m in data.get("data", [])]
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
        url = f"{self.base_url}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, json=payload)
                if res.status_code != 200:
                    raise RuntimeError(f"Server error {res.status_code}: {res.text}")
                data = res.json()
                duration_ms = (time.time() - start_time) * 1000.0
                content = data["choices"][0]["message"]["content"]
                return LLMProviderResponse(
                    provider=self.provider_id,
                    model=model,
                    prompt=prompt,
                    response_text=content,
                    duration_ms=round(duration_ms, 1)
                )
        except httpx.ConnectError:
            raise ConnectionError(f"Could not connect to {self.provider_id} at {self.base_url}.")

    async def generate_stream(self,
                              prompt: str,
                              model: str,
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 512) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        raise RuntimeError(f"Stream error {response.status_code}")
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0].get("delta", {}).get("content", "")
                                if delta:
                                    yield delta
                            except json.JSONDecodeError:
                                pass
        except httpx.ConnectError:
            yield f"[Connection Error: Server is unreachable at {self.base_url}]"
