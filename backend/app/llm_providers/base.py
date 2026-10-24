from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator, Optional, Tuple
import urllib.parse
import ipaddress
from pydantic import BaseModel

def join_url_path(base_url: str, endpoint: str) -> str:
    """
    Safely joins a base URL with an endpoint path without duplicating path segments
    such as '/v1' or '/api'.
    Example:
      join_url_path("http://localhost:3000/v1", "/v1/chat/completions") -> "http://localhost:3000/v1/chat/completions"
      join_url_path("http://localhost:3000/api", "/api/generate") -> "http://localhost:3000/api/generate"
      join_url_path("http://localhost:11434", "/api/tags") -> "http://localhost:11434/api/tags"
    """
    base = (base_url or "").rstrip("/")
    ep = (endpoint or "").strip()
    if not ep:
        return base
    if not ep.startswith("/"):
        ep = "/" + ep

    try:
        parsed_base = urllib.parse.urlsplit(base)
        base_path = parsed_base.path.rstrip("/")

        if base_path:
            # Check if the endpoint starts with base_path to avoid duplication
            if ep == base_path:
                ep = ""
            elif ep.startswith(base_path + "/"):
                ep = ep[len(base_path):]
            elif base_path.endswith("/api") and ep.startswith("/api/"):
                ep = ep[4:]
            elif base_path.endswith("/v1") and ep.startswith("/v1/"):
                ep = ep[3:]

        if parsed_base.scheme and parsed_base.netloc:
            new_path = (base_path + ep) if ep.startswith("/") else (f"{base_path}/{ep}" if ep else base_path)
            if not new_path.startswith("/"):
                new_path = "/" + new_path
            return urllib.parse.urlunsplit((
                parsed_base.scheme,
                parsed_base.netloc,
                new_path,
                parsed_base.query,
                parsed_base.fragment
            ))
        else:
            return base + ep
    except Exception:
        return base.rstrip("/") + "/" + ep.lstrip("/")

def parse_and_validate_local_endpoint(
    raw_input: str,
    default_port: Optional[int] = 11434,
    allow_remote: bool = False
) -> Tuple[bool, str, str, str, int]:
    """
    Parses and validates local inference server endpoint with SSRF protection.
    Accepts:
      - 'http://127.0.0.1:11434'
      - 'http://localhost:3000'
      - '127.0.0.1:3000'
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
        if parsed.port is not None:
            port = parsed.port
        elif default_port is not None:
            port = default_port
        else:
            port = 443 if scheme == "https" else 80
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

    # Clean canonical URL, preserving port and base path
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
