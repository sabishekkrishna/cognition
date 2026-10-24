from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator, Optional, Tuple
import urllib.parse
import ipaddress
from pydantic import BaseModel

def parse_and_validate_local_endpoint(
    raw_input: str,
    default_port: int = 11434,
    allow_remote: bool = False
) -> Tuple[bool, str, str, str, int]:
    """
    Parses and validates local inference server endpoint with SSRF protection.
    Accepts:
      - 'http://127.0.0.1:11434'
      - '127.0.0.1:11434'
      - 'localhost:8080'
      - 'http://localhost:1234/v1'

    Returns:
      (is_valid, canonical_url, host, error_message, port)
    """
    raw = (raw_input or "").strip()
    if not raw:
        return False, "", "", "Endpoint address cannot be empty.", 0

    if not (raw.startswith("http://") or raw.startswith("https://")):
        raw = "http://" + raw

    try:
        parsed = urllib.parse.urlsplit(raw)
    except Exception as e:
        return False, "", "", f"Invalid URL syntax: {e}", 0

    scheme = parsed.scheme.lower()
    if scheme not in ["http", "https"]:
        return False, "", "", f"Unsupported protocol '{scheme}'. Use 'http' or 'https'.", 0

    host = parsed.hostname
    if not host:
        return False, "", "", "Host name or IP address is required.", 0

    try:
        port = parsed.port or default_port
    except ValueError:
        return False, "", host, "Invalid port number. Port must be an integer between 1 and 65535.", 0

    if not (1 <= port <= 65535):
        return False, "", host, f"Port {port} is out of valid range (1-65535).", port

    # SSRF Protection: verify if host is local loopback
    is_loopback = False
    if host.lower() in ["localhost", "127.0.0.1", "::1"]:
        is_loopback = True
    else:
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_loopback:
                is_loopback = True
        except ValueError:
            # Host is a domain name other than localhost
            pass

    if not is_loopback and not allow_remote:
        return (
            False, "", host,
            f"SSRF Security Violation: Host '{host}' is not a local loopback address (127.0.0.1 / localhost). "
            "To prevent Server-Side Request Forgery attacks against internal networks, external IPs are restricted. "
            "Enable 'allow_remote_llm' in Settings if you explicitly trust this target.",
            port
        )

    # Clean canonical URL
    path = parsed.path.rstrip("/")
    canonical_url = f"{scheme}://{host}:{port}{path}"
    return True, canonical_url, host, "", port

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
