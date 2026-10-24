import httpx
import time
import json
from typing import List, Dict, Any, AsyncGenerator, Optional, Tuple
from .base import BaseLLMProvider, LLMProviderResponse, join_url_path

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

    def set_base_url(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def get_provider_name(self) -> str:
        return self.provider_id

    def _get_models_endpoint(self) -> str:
        if self.base_url.endswith("/v1") or "/v1" in self.base_url:
            return join_url_path(self.base_url, "/models")
        return join_url_path(self.base_url, "/v1/models")

    def _get_chat_endpoint(self) -> str:
        if self.base_url.endswith("/v1") or "/v1" in self.base_url:
            return join_url_path(self.base_url, "/chat/completions")
        return join_url_path(self.base_url, "/v1/chat/completions")

    async def check_health(self) -> Tuple[bool, str]:
        endpoint = self._get_models_endpoint()
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(endpoint)
                content_type = res.headers.get("content-type", "").lower()
                body_start = res.text[:200].strip().lower()

                if "text/html" in content_type or body_start.startswith("<!doctype") or body_start.startswith("<html"):
                    return (
                        False,
                        f"Server at {self.base_url} is serving frontend HTML ('text/html'), not an OpenAI-compatible API. "
                        "It appears to be a web/frontend development server. To connect to an LLM, point to an inference daemon "
                        "(e.g., LM Studio on port 1234 or llama.cpp on port 8080)."
                    )

                if res.status_code == 404:
                    # Try alternative path without /v1 or with /v1
                    alt_endpoint = join_url_path(self.base_url, "/models")
                    res2 = await client.get(alt_endpoint)
                    if res2.status_code == 200:
                        return True, f"{self.provider_id} server connected and operational."

                if res.status_code == 200:
                    return True, f"{self.provider_id} server connected and operational."
                return False, f"Server returned HTTP status {res.status_code}."
        except httpx.ConnectError:
            return False, f"Connection refused at {self.base_url}."
        except Exception as e:
            return False, f"Connection error: {str(e)}"

    async def list_models(self) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                endpoint = self._get_models_endpoint()
                res = await client.get(endpoint)
                if res.status_code == 404:
                    endpoint = join_url_path(self.base_url, "/models")
                    res = await client.get(endpoint)

                content_type = res.headers.get("content-type", "").lower()
                if "text/html" in content_type:
                    return []

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
        url = self._get_chat_endpoint()
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
                if res.status_code == 404 and "/v1/" in url:
                    # Fallback to without /v1
                    alt_url = join_url_path(self.base_url, "/chat/completions")
                    res = await client.post(alt_url, json=payload)

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
        url = self._get_chat_endpoint()
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
