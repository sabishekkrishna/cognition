import pytest
from starlette.testclient import TestClient
from backend.app.main import app
from backend.app.llm_providers.base import parse_and_validate_local_endpoint, join_url_path
from backend.app.llm_providers.custom_http import CustomLocalProvider
from backend.app.llm_providers.registry import get_provider_registry
from backend.app.storage.database import set_setting, get_all_settings
import httpx
import json

client = TestClient(app)

def test_join_url_path_prevents_duplicate_segments():
    """Verify join_url_path prevents duplicate /v1, /api, or multiple slashes."""
    # OpenAI duplicate /v1 prevention
    assert join_url_path("http://localhost:3000/v1", "/v1/chat/completions") == "http://localhost:3000/v1/chat/completions"
    assert join_url_path("http://localhost:3000/v1/", "v1/chat/completions") == "http://localhost:3000/v1/chat/completions"
    assert join_url_path("http://localhost:3000", "/v1/chat/completions") == "http://localhost:3000/v1/chat/completions"
    assert join_url_path("http://localhost:3000", "/chat/completions") == "http://localhost:3000/chat/completions"

    # Ollama duplicate /api prevention
    assert join_url_path("http://localhost:11434/api", "/api/tags") == "http://localhost:11434/api/tags"
    assert join_url_path("http://localhost:11434/api/", "api/tags") == "http://localhost:11434/api/tags"
    assert join_url_path("http://localhost:11434", "/api/tags") == "http://localhost:11434/api/tags"
    assert join_url_path("http://localhost:11434", "/api/generate") == "http://localhost:11434/api/generate"

    # Port 3000 custom endpoints
    assert join_url_path("http://localhost:3000", "/generate") == "http://localhost:3000/generate"
    assert join_url_path("http://127.0.0.1:3000/llm", "/generate") == "http://127.0.0.1:3000/llm/generate"

def test_parse_and_validate_local_endpoint_preserves_custom_ports():
    """Verify custom ports such as 3000, 11434, 1234, 8080, 5000 are strictly preserved and not overwritten."""
    # Port 3000
    valid, canon, host, err, port = parse_and_validate_local_endpoint("http://localhost:3000")
    assert valid is True
    assert port == 3000
    assert canon == "http://localhost:3000"
    assert host == "localhost"

    valid, canon, host, err, port = parse_and_validate_local_endpoint("http://127.0.0.1:3000")
    assert valid is True
    assert port == 3000
    assert canon == "http://127.0.0.1:3000"

    # Default ports
    valid, canon, host, err, port = parse_and_validate_local_endpoint("http://localhost:11434")
    assert valid is True
    assert port == 11434

    valid, canon, host, err, port = parse_and_validate_local_endpoint("http://127.0.0.1:1234/v1")
    assert valid is True
    assert port == 1234
    assert canon == "http://127.0.0.1:1234/v1"

    # Custom arbitrary port e.g. 5000 (never silently overwritten with 11434)
    valid, canon, host, err, port = parse_and_validate_local_endpoint("http://localhost:5000", default_port=11434)
    assert valid is True
    assert port == 5000
    assert canon == "http://localhost:5000"

def test_parse_and_validate_port_ranges_and_syntax():
    """Verify port range validation (1-65535) and rejection of invalid ports."""
    valid, _, _, err, _ = parse_and_validate_local_endpoint("http://localhost:0")
    assert valid is False
    assert "65535" in err

    valid, _, _, err, _ = parse_and_validate_local_endpoint("http://localhost:70000")
    assert valid is False
    assert "65535" in err

    valid, _, _, err, _ = parse_and_validate_local_endpoint("http://localhost:invalidport")
    assert valid is False
    assert "Invalid port" in err or "65535" in err

def test_ssrf_protection_loopback_and_override():
    """Verify SSRF protection permits loopback hosts and rejects external targets unless allowed."""
    # Loopback targets permitted
    valid, _, _, _, _ = parse_and_validate_local_endpoint("http://localhost:3000")
    assert valid is True

    valid, _, _, _, _ = parse_and_validate_local_endpoint("http://127.0.0.1:3000")
    assert valid is True

    valid, _, _, _, _ = parse_and_validate_local_endpoint("http://[::1]:3000")
    assert valid is True

    # External targets blocked by default
    valid, _, _, err, _ = parse_and_validate_local_endpoint("http://192.168.1.100:8000", allow_remote=False)
    assert valid is False
    assert "SSRF Security Violation" in err

    # Permitted when allow_remote is True
    valid, _, _, _, _ = parse_and_validate_local_endpoint("http://192.168.1.100:8000", allow_remote=True)
    assert valid is True

def test_test_connection_diagnostics_html_response_handling(monkeypatch):
    """
    Simulate testing http://localhost:3000 when it serves frontend HTML ('text/html').
    Verify it fails Step 3 cleanly with an explicit explanation instead of crashing with JSONDecodeError.
    """
    html_content = "<!doctype html><html><head><title>Frontend App</title></head><body>Vite React</body></html>"

    class MockHTMLResponse:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        text = html_content
        def json(self):
            raise json.decoder.JSONDecodeError("Expecting value", html_content, 0)

    async def mock_get(self, url, *args, **kwargs):
        return MockHTMLResponse()

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    res = client.post("/api/llm/test-connection", json={
        "provider": "ollama",
        "base_url": "http://localhost:3000"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["overall_status"] == "failed"
    assert len(data["steps"]) >= 3

    # Step 1 passed
    assert data["steps"][0]["step_number"] == 1
    assert data["steps"][0]["status"] == "pass"

    # Step 2 passed
    assert data["steps"][1]["step_number"] == 2
    assert data["steps"][1]["status"] == "pass"

    # Step 3 failed with clear diagnostic message about frontend HTML
    step3 = data["steps"][2]
    assert step3["step_number"] == 3
    assert step3["status"] == "fail"
    assert "serving frontend HTML ('text/html'), not an LLM inference API" in step3["message"]
    assert "Port 3000 is serving the web application UI" in step3["remedy"] or "frontend" in step3["remedy"].lower()

@pytest.mark.anyio
async def test_custom_local_provider_openai_mode(monkeypatch):
    """Verify CustomLocalProvider works with OpenAI format."""
    provider = CustomLocalProvider(
        base_url="http://localhost:3000",
        api_path="/v1",
        adapter_type="openai",
        fallback_model="custom-gpt"
    )

    class MockOpenAIResponse:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = '{"choices": [{"message": {"content": "Custom OpenAI response"}}]}'
        def json(self):
            return {"choices": [{"message": {"content": "Custom OpenAI response"}}]}

    class MockModelsResponse:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = '{"data": [{"id": "model-a"}, {"id": "model-b"}]}'
        def json(self):
            return {"data": [{"id": "model-a"}, {"id": "model-b"}]}

    async def mock_post(self, url, *args, **kwargs):
        assert "chat/completions" in url
        return MockOpenAIResponse()

    async def mock_get(self, url, *args, **kwargs):
        return MockModelsResponse()

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    models = await provider.list_models()
    assert models == ["model-a", "model-b"]

    resp = await provider.generate(prompt="Hello", model="model-a")
    assert resp.provider == "custom_http"
    assert resp.response_text == "Custom OpenAI response"
    assert resp.model == "model-a"

@pytest.mark.anyio
async def test_custom_local_provider_fallback_model_when_discovery_unavailable(monkeypatch):
    """Verify CustomLocalProvider falls back to specified model when discovery returns 404."""
    provider = CustomLocalProvider(
        base_url="http://localhost:3000",
        adapter_type="generic",
        fallback_model="my-local-model"
    )

    class Mock404Response:
        status_code = 404
        headers = {"content-type": "application/json"}
        text = "Not found"
        def json(self):
            return {"error": "not found"}

    async def mock_get(self, url, *args, **kwargs):
        return Mock404Response()

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    models = await provider.list_models()
    assert models == ["my-local-model"]

def test_save_and_persist_custom_url_configuration():
    """Verify configuring and saving custom URL updates database and runtime registry."""
    # Update settings with custom URL on port 3000
    res = client.post("/api/settings", json={
        "custom_llm_url": "http://localhost:3000",
        "custom_llm_path": "/api",
        "custom_llm_model": "yolo-llm",
        "llm_provider": "custom_http"
    })
    assert res.status_code == 200

    # Verify settings persisted in SQLite
    settings = get_all_settings()
    assert settings.get("custom_llm_url") == "http://localhost:3000"
    assert settings.get("custom_llm_path") == "/api"
    assert settings.get("custom_llm_model") == "yolo-llm"
    assert settings.get("llm_provider") == "custom_http"

    # Verify runtime registry reflects the saved custom URL
    registry = get_provider_registry()
    providers = registry.list_available_providers()
    custom_entry = next((p for p in providers if p["id"] == "custom_http"), None)
    assert custom_entry is not None
    assert custom_entry["base_url"] == "http://localhost:3000"

    # Clean up back to mock provider for other tests
    client.post("/api/settings", json={"llm_provider": "mock"})
