from typing import Dict, Any, Optional, List
from .base import BaseLLMProvider
from .ollama import OllamaProvider
from .openai_compat import OpenAICompatProvider
from .mock_local import OfflineMockProvider
from ..storage.database import get_all_settings

class ProviderRegistry:
    """
    Central registry for local LLM inference engines.
    Manages active provider instance, connection configuration, and failover.
    """
    def __init__(self):
        settings = get_all_settings()
        ollama_url = settings.get("ollama_url", "http://127.0.0.1:11434")
        openai_url = settings.get("openai_compat_url", "http://127.0.0.1:1234/v1")

        self._providers: Dict[str, BaseLLMProvider] = {
            "ollama": OllamaProvider(base_url=ollama_url),
            "lm_studio": OpenAICompatProvider(provider_id="lm_studio", base_url=openai_url),
            "llama_cpp": OpenAICompatProvider(provider_id="llama_cpp", base_url="http://127.0.0.1:8080/v1"),
            "mock": OfflineMockProvider()
        }
        self._active_key = settings.get("llm_provider", "mock")

    def register_provider(self, key: str, provider: BaseLLMProvider):
        self._providers[key] = provider

    def get_provider(self, key: Optional[str] = None) -> BaseLLMProvider:
        target_key = key or self._active_key
        if target_key not in self._providers:
            # Fallback to mock
            return self._providers["mock"]
        return self._providers[target_key]

    def set_active_provider(self, key: str):
        if key in self._providers:
            self._active_key = key
        else:
            raise ValueError(f"Unknown provider: {key}")

    def list_available_providers(self) -> List[Dict[str, str]]:
        return [
            {"id": "ollama", "name": "Ollama (http://127.0.0.1:11434)", "type": "local_daemon"},
            {"id": "lm_studio", "name": "LM Studio (http://127.0.0.1:1234/v1)", "type": "openai_compat"},
            {"id": "llama_cpp", "name": "llama.cpp Server (http://127.0.0.1:8080/v1)", "type": "openai_compat"},
            {"id": "mock", "name": "Offline Local Demo Provider (Synthetic)", "type": "local_mock"}
        ]

_registry_instance = None

def get_provider_registry() -> ProviderRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ProviderRegistry()
    return _registry_instance
