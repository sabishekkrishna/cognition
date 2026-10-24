from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator, Optional, Tuple
from pydantic import BaseModel

class LLMProviderResponse(BaseModel):
    provider: str
    model: str
    prompt: str
    response_text: str
    duration_ms: float
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = "stop"

class BaseLLMProvider(ABC):
    """
    Abstract interface for local LLM inference engines (Ollama, LM Studio, llama.cpp, etc.)
    Strictly local operation: only loopback connections are permitted.
    """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Returns the identifier name of the provider."""
        pass

    @abstractmethod
    async def check_health(self) -> Tuple[bool, str]:
        """Checks if the local server daemon is reachable and responding."""
        pass

    @abstractmethod
    async def list_models(self) -> List[str]:
        """Discovers available local models hosted by this provider."""
        pass

    @abstractmethod
    async def generate(self,
                       prompt: str,
                       model: str,
                       system_prompt: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: int = 512) -> LLMProviderResponse:
        """Executes a single prompt completion."""
        pass

    @abstractmethod
    async def generate_stream(self,
                              prompt: str,
                              model: str,
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 512) -> AsyncGenerator[str, None]:
        """Streams response tokens as they are produced."""
        pass
