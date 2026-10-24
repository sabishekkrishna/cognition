import asyncio
import time
from typing import List, Dict, Any, AsyncGenerator, Optional, Tuple
from .base import BaseLLMProvider, LLMProviderResponse

class OfflineMockProvider(BaseLLMProvider):
    """
    Offline Local Demo Provider.
    Clearly labeled synthetic demonstration provider for evaluating monitor behavior,
    streaming, uncertainty detection, and user rating feedback when an external Ollama daemon
    has not yet been booted.
    """
    def __init__(self):
        self.models = [
            "demo-llama3-8b-instruct (Synthetic)",
            "demo-mistral-7b (Synthetic)",
            "demo-phi3-mini (Synthetic)"
        ]

    def get_provider_name(self) -> str:
        return "mock_local"

    async def check_health(self) -> Tuple[bool, str]:
        return True, "Offline Local Demo Provider active (Local test mode)."

    async def list_models(self) -> List[str]:
        return self.models

    async def generate(self,
                       prompt: str,
                       model: str,
                       system_prompt: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: int = 512) -> LLMProviderResponse:
        start_time = time.time()
        await asyncio.sleep(0.3) # realistic local generation latency

        lower = prompt.lower()
        if "blurry" in lower or "dark" in lower or "noisy" in lower:
            response_text = (
                "Based on the input image provided, the fine visual details are unclear and heavily degraded. "
                "I am not certain about the exact object category due to blur and poor exposure. "
                "Please verify or capture a clearer image under steady lighting before relying on this analysis."
            )
        elif "audio" in lower or "sound" in lower:
            response_text = (
                "The acoustic speech segment was received. The primary utterance appears to be an instructional command, "
                "though background noise may have masked the final word phonemes."
            )
        else:
            response_text = (
                f"Local analysis of '{prompt[:60]}...': The request has been evaluated within local memory. "
                "Input reliability monitoring verified that upstream sensors and features conform to normal operating parameters."
            )

        duration_ms = (time.time() - start_time) * 1000.0
        return LLMProviderResponse(
            provider="mock_local",
            model=model,
            prompt=prompt,
            response_text=response_text,
            duration_ms=round(duration_ms, 1),
            total_tokens=len(response_text.split())
        )

    async def generate_stream(self,
                              prompt: str,
                              model: str,
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 512) -> AsyncGenerator[str, None]:
        res = await self.generate(prompt, model, system_prompt, temperature, max_tokens)
        words = res.response_text.split(" ")
        for w in words:
            await asyncio.sleep(0.04)
            yield w + " "
