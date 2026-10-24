from typing import Dict, Any, Optional, List
from .base import BaseLLMProvider
from .ollama import OllamaProvider
from .openai_compat import OpenAICompatProvider
from .custom_http import CustomLocalProvider
from .mock_local import OfflineMockProvider
from ..storage.database import get_all_settings

class ProviderRegistry:
    """
    Central registry for local LLM inference engines.
    Manages active provider instance, connection configuration, dynamic URL updates, and failover.
    """
    def __init__(self):
        settings = get_all_settings()
        ollama_url = settings.get("ollama_url", "http://127.0.0.1:11434")
        openai_url = settings.get("openai_compat_url", "http://127.0.0.1:1234/v1")
        llama_url = settings.get("llama_cpp_url", "http://127.0.0.1:8080/v1")
        custom_url = settings.get("custom_llm_url", "http://127.0.0.1:3000")
        custom_path = settings.get("custom_llm_path", "")
        custom_type = settings.get("custom_llm_type", "auto")
        custom_model = settings.get("custom_llm_model", "custom-model")

        self._providers: Dict[str, BaseLLMProvider] = {
            "ollama": OllamaProvider(base_url=ollama_url),
            "lm_studio": OpenAICompatProvider(provider_id="lm_studio", base_url=openai_url),
            "llama_cpp": OpenAICompatProvider(provider_id="llama_cpp", base_url=llama_url),
            "custom_http": CustomLocalProvider(
                base_url=custom_url,
                api_path=custom_path,
                adapter_type=custom_type,
                fallback_model=custom_model
            ),
            "mock": OfflineMockProvider()
        }
        self._active_key = settings.get("llm_provider", "mock")

    def sync_from_settings(self):
        """Refreshes URLs and active provider from database settings."""
        settings = get_all_settings()
        if "ollama" in self._providers and "ollama_url" in settings:
            self.update_provider_url("ollama", settings["ollama_url"])
        if "lm_studio" in self._providers and "openai_compat_url" in settings:
            self.update_provider_url("lm_studio", settings["openai_compat_url"])
        if "llama_cpp" in self._providers and "llama_cpp_url" in settings:
            self.update_provider_url("llama_cpp", settings["llama_cpp_url"])
        if "custom_http" in self._providers:
            custom_prov = self._providers["custom_http"]
            if isinstance(custom_prov, CustomLocalProvider):
                if "custom_llm_url" in settings:
                    custom_prov.set_base_url(settings["custom_llm_url"])
                if "custom_llm_path" in settings:
                    custom_prov.set_api_path(settings["custom_llm_path"])
                if "custom_llm_type" in settings:
                    custom_prov.set_adapter_type(settings["custom_llm_type"])
                if "custom_llm_model" in settings:
                    custom_prov.set_fallback_model(settings["custom_llm_model"])
        if "llm_provider" in settings:
            self._active_key = settings["llm_provider"]

    def update_provider_url(self, key: str, url: str):
        """Dynamically updates the base URL of a registered provider."""
        if key in self._providers:
            prov = self._providers[key]
            if hasattr(prov, "set_base_url"):
                prov.set_base_url(url)

    def update_custom_provider_config(self,
                                      base_url: Optional[str] = None,
                                      api_path: Optional[str] = None,
                                      adapter_type: Optional[str] = None,
                                      fallback_model: Optional[str] = None):
        if "custom_http" in self._providers:
            custom_prov = self._providers["custom_http"]
            if isinstance(custom_prov, CustomLocalProvider):
                if base_url:
                    custom_prov.set_base_url(base_url)
                if api_path is not None:
                    custom_prov.set_api_path(api_path)
                if adapter_type:
                    custom_prov.set_adapter_type(adapter_type)
                if fallback_model:
                    custom_prov.set_fallback_model(fallback_model)

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
        ollama_url = getattr(self._providers.get("ollama"), "base_url", "http://127.0.0.1:11434")
        lm_url = getattr(self._providers.get("lm_studio"), "base_url", "http://127.0.0.1:1234/v1")
        llama_url = getattr(self._providers.get("llama_cpp"), "base_url", "http://127.0.0.1:8080/v1")
        custom_url = getattr(self._providers.get("custom_http"), "base_url", "http://127.0.0.1:3000")

        return [
            {"id": "ollama", "name": f"Ollama ({ollama_url})", "type": "local_daemon", "base_url": ollama_url},
            {"id": "lm_studio", "name": f"LM Studio ({lm_url})", "type": "openai_compat", "base_url": lm_url},
            {"id": "llama_cpp", "name": f"llama.cpp Server ({llama_url})", "type": "openai_compat", "base_url": llama_url},
            {"id": "custom_http", "name": f"Custom Local HTTP API ({custom_url})", "type": "custom_http", "base_url": custom_url},
            {"id": "mock", "name": "Offline Local Demo Provider (Synthetic)", "type": "local_mock", "base_url": "internal://mock"}
        ]

_registry_instance = None

def get_provider_registry() -> ProviderRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ProviderRegistry()
    return _registry_instance
