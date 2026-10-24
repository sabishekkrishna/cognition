import httpx
import os
import time
import json
from typing import List, Dict, Any, AsyncGenerator, Optional, Tuple
from .base import BaseLLMProvider, LLMProviderResponse, join_url_path

class CustomLocalProvider(BaseLLMProvider):
    """
    Configurable adapter for custom local HTTP inference servers, proxies, and gateways.
    Supports:
      - Custom base URLs (e.g. http://localhost:8080, http://127.0.0.1:8000, http://localhost:3000)
      - Configurable base paths (e.g. /v1, /api, /llm, or root)
      - Configurable adapter modes: 'auto', 'nova', 'openai', 'ollama', 'generic'
      - Manual model specification when model discovery (/models, /api/tags) is not implemented
    """
    def __init__(self,
                 base_url: Optional[str] = None,
                 api_path: str = "",
                 adapter_type: str = "auto",
                 fallback_model: str = "custom-model",
                 timeout_sec: float = 45.0):
        if not base_url or base_url == "http://127.0.0.1:3000":
            base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080")
        self.base_url = base_url.rstrip("/")
        self.api_path = api_path.strip().strip("/")
        self.adapter_type = adapter_type or "auto"
        self.fallback_model = fallback_model or "custom-model"
        self.timeout = timeout_sec

    def set_base_url(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def set_api_path(self, api_path: str):
        self.api_path = (api_path or "").strip().strip("/")

    def set_adapter_type(self, adapter_type: str):
        self.adapter_type = adapter_type or "auto"

    def set_fallback_model(self, model: str):
        if model:
            self.fallback_model = model

    def get_provider_name(self) -> str:
        return "custom_http"

    def _resolve_url(self, endpoint: str) -> str:
        """Builds target URL honoring optional api_path and preventing duplicated segments."""
        effective_base = self.base_url
        if self.api_path:
            effective_base = join_url_path(self.base_url, f"/{self.api_path}")
        return join_url_path(effective_base, endpoint)

    async def _detect_adapter(self, client: httpx.AsyncClient) -> str:
        if self.adapter_type != "auto":
            return self.adapter_type

        # Check Custom Ollama-based LLM (e.g. NOVA AI on port 8080)
        try:
            res = await client.get(self._resolve_url("/api/health"), timeout=5.0)
            ct = res.headers.get("content-type", "").lower()
            if res.status_code == 200 and "application/json" in ct:
                data = res.json()
                if isinstance(data, dict) and ("app_name" in data or "ollama_connected" in data):
                    return "nova"
        except Exception:
            pass

        try:
            res = await client.get(self._resolve_url("/api/models"), timeout=5.0)
            ct = res.headers.get("content-type", "").lower()
            if res.status_code == 200 and "application/json" in ct:
                data = res.json()
                if isinstance(data, dict) and "models" in data:
                    return "nova"
        except Exception:
            pass

        # Check OpenAI /v1/models or /models
        for path in ["/v1/models", "/models"]:
            try:
                res = await client.get(self._resolve_url(path), timeout=5.0)
                ct = res.headers.get("content-type", "").lower()
                if res.status_code == 200 and "application/json" in ct:
                    return "openai"
            except Exception:
                pass

        # Check Ollama /api/tags
        try:
            res = await client.get(self._resolve_url("/api/tags"), timeout=5.0)
            ct = res.headers.get("content-type", "").lower()
            if res.status_code == 200 and "application/json" in ct:
                return "ollama"
        except Exception:
            pass

        return "generic"

    async def check_health(self) -> Tuple[bool, str]:
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                mode = await self._detect_adapter(client)
                if mode != "generic":
                    return True, f"Custom HTTP provider reachable at {self.base_url} (mode: {mode})."

                probe_url = self.base_url
                res = await client.get(probe_url)
                content_type = res.headers.get("content-type", "").lower()
                body_start = res.text[:200].strip().lower()

                if "text/html" in content_type or body_start.startswith("<!doctype") or body_start.startswith("<html"):
                    return (
                        False,
                        f"Server at {self.base_url} is serving frontend HTML ('text/html'), not an LLM inference API. "
                        "It appears to be a web/frontend development server. To connect to an LLM, point to an inference daemon "
                        "(e.g., Ollama on port 11434 or LM Studio on port 1234) or configure a local proxy that exposes LLM endpoints."
                    )

                return True, f"Custom HTTP provider reachable at {self.base_url} (mode: {mode})."
        except httpx.ConnectError:
            return False, f"Connection refused at {self.base_url}. Verify the custom local server is running."
        except Exception as e:
            return False, f"Custom server connection error: {str(e)}"

    async def list_models(self) -> List[str]:
        models: List[str] = []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                mode = await self._detect_adapter(client)
                if mode == "nova":
                    try:
                        res = await client.get(self._resolve_url("/api/models"), timeout=5.0)
                        ct = res.headers.get("content-type", "").lower()
                        if res.status_code == 200 and "application/json" in ct:
                            data = res.json()
                            m_list = data.get("models", [])
                            discovered = [
                                (m.get("name") or m.get("model"))
                                for m in m_list
                                if (m.get("name") or m.get("model"))
                            ]
                            if discovered:
                                return discovered
                    except Exception:
                        pass
                elif mode == "openai":
                    for path in ["/v1/models", "/models"]:
                        try:
                            res = await client.get(self._resolve_url(path))
                            ct = res.headers.get("content-type", "").lower()
                            if res.status_code == 200 and "text/html" not in ct:
                                data = res.json()
                                if isinstance(data, dict) and "data" in data:
                                    discovered = [m.get("id", "") for m in data["data"] if m.get("id")]
                                    if discovered:
                                        return discovered
                        except Exception:
                            continue
                elif mode == "ollama":
                    res = await client.get(self._resolve_url("/api/tags"))
                    ct = res.headers.get("content-type", "").lower()
                    if res.status_code == 200 and "text/html" not in ct:
                        data = res.json()
                        discovered = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                        if discovered:
                            return discovered
        except Exception:
            pass

        # If discovery endpoint is not available or returned no models, fallback to configured model
        if self.fallback_model:
            return [self.fallback_model]
        return ["custom-model"]

    async def generate(self,
                       prompt: str,
                       model: str,
                       system_prompt: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: int = 512) -> LLMProviderResponse:
        start_time = time.time()
        effective_model = model or self.fallback_model or "custom-model"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            mode = await self._detect_adapter(client)

            if mode == "nova":
                endpoint = self._resolve_url("/api/chat")
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                payload = {
                    "model": effective_model,
                    "messages": messages,
                    "stream": False
                }
                res = await client.post(endpoint, json=payload, timeout=self.timeout)
                if res.status_code != 200:
                    raise RuntimeError(f"Custom LLM server error {res.status_code}: {res.text}")
                data = res.json()
                duration_ms = (time.time() - start_time) * 1000.0
                return LLMProviderResponse(
                    provider="custom_http",
                    model=effective_model,
                    prompt=prompt,
                    response_text=data.get("response", ""),
                    duration_ms=round(duration_ms, 1)
                )

            elif mode == "openai":
                endpoint = self._resolve_url("/chat/completions")
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                payload = {
                    "model": effective_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                }
                res = await client.post(endpoint, json=payload)
                if res.status_code != 200:
                    raise RuntimeError(f"OpenAI-compat server error {res.status_code}: {res.text}")
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                duration_ms = (time.time() - start_time) * 1000.0
                return LLMProviderResponse(
                    provider="custom_http",
                    model=effective_model,
                    prompt=prompt,
                    response_text=content,
                    duration_ms=round(duration_ms, 1)
                )

            elif mode == "ollama":
                endpoint = self._resolve_url("/api/generate")
                payload = {
                    "model": effective_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens}
                }
                if system_prompt:
                    payload["system"] = system_prompt
                res = await client.post(endpoint, json=payload)
                if res.status_code != 200:
                    raise RuntimeError(f"Ollama server error {res.status_code}: {res.text}")
                data = res.json()
                duration_ms = (time.time() - start_time) * 1000.0
                return LLMProviderResponse(
                    provider="custom_http",
                    model=effective_model,
                    prompt=prompt,
                    response_text=data.get("response", ""),
                    duration_ms=round(duration_ms, 1)
                )

            else: # generic adapter
                # Try POST to /generate or base endpoint
                endpoint = self._resolve_url("/generate")
                payload = {
                    "model": effective_model,
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                try:
                    res = await client.post(endpoint, json=payload)
                except Exception:
                    endpoint = self.base_url
                    res = await client.post(endpoint, json=payload)

                if res.status_code != 200:
                    raise RuntimeError(f"Custom LLM server returned HTTP {res.status_code}: {res.text}")

                try:
                    data = res.json()
                    if isinstance(data, dict):
                        content = data.get("response") or data.get("text") or data.get("content") or data.get("output") or json.dumps(data)
                    else:
                        content = str(data)
                except Exception:
                    content = res.text

                duration_ms = (time.time() - start_time) * 1000.0
                return LLMProviderResponse(
                    provider="custom_http",
                    model=effective_model,
                    prompt=prompt,
                    response_text=content,
                    duration_ms=round(duration_ms, 1)
                )

    async def generate_stream(self,
                              prompt: str,
                              model: str,
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 512) -> AsyncGenerator[str, None]:
        effective_model = model or self.fallback_model or "custom-model"
        mode = self.adapter_type
        if mode == "auto":
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    mode = await self._detect_adapter(client)
            except Exception:
                mode = "generic"

        if mode == "nova":
            endpoint = self._resolve_url("/api/chat")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": effective_model,
                "messages": messages,
                "stream": True
            }
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream("POST", endpoint, json=payload) as response:
                        if response.status_code == 200:
                            async for chunk in response.aiter_text():
                                if chunk:
                                    yield chunk
                            return
            except Exception:
                pass

        # Fallback to standard generation and stream chunks
        resp = await self.generate(prompt, model, system_prompt, temperature, max_tokens)
        words = resp.response_text.split(" ")
        for i, word in enumerate(words):
            yield (word + " " if i < len(words) - 1 else word)
