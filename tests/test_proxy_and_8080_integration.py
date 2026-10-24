import os
import json
import pytest
from starlette.testclient import TestClient
from backend.app.main import app
from backend.app.llm_providers.response_monitor import ResponseMonitor
from backend.app.llm_providers.custom_http import CustomLocalProvider
from backend.app.storage.database import set_setting, get_all_settings, get_recent_events

client = TestClient(app)

def test_proxy_chat_endpoint_non_streaming():
    """
    Verify Client -> AI Monitor (8000) -> Custom LLM (8080) transparent proxying:
    - Forwards request body, model, messages, parameters.
    - Captures input and response.
    - Evaluates exchange through reliability pipeline.
    - Attaches X-Monitor-* headers.
    - Returns genuine LLM response.
    """
    payload = {
        "model": "gemma2:2b",
        "messages": [
            {"role": "user", "content": "Respond with the single word: READY"}
        ],
        "stream": False
    }

    res = client.post("/api/chat", json=payload, headers={"X-Input-Risk-Score": "15.0", "X-Input-Subsystem": "image"})
    assert res.status_code == 200, f"Expected 200 but got {res.status_code}: {res.text}"

    data = res.json()
    assert "response" in data
    assert len(data["response"]) > 0
    assert "model" in data

    # Verify custom monitor headers
    assert "x-monitor-risk-score" in res.headers
    assert "x-monitor-interaction-id" in res.headers
    assert "x-monitor-relevance-Score".lower() in [k.lower() for k in res.headers.keys()]
    assert "x-monitor-evidence-status" in res.headers

    # Verify interaction ID was recorded in database
    interaction_id = int(res.headers["x-monitor-interaction-id"])
    interactions = ResponseMonitor.list_interactions(limit=10)
    matched = [i for i in interactions if i["id"] == interaction_id]
    assert len(matched) == 1
    assert "READY" in matched[0]["prompt"]
    assert matched[0]["input_risk_score"] == 15.0
    assert matched[0]["input_subsystem"] == "image"

def test_proxy_root_chat_and_proxy_chat_endpoints():
    """Verify both /chat and /api/llm/proxy/chat function identically as proxy endpoints."""
    payload = {
        "model": "gemma2:2b",
        "messages": [
            {"role": "user", "content": "Say OK"}
        ],
        "stream": False
    }

    # Test /chat
    res_root = client.post("/chat", json=payload)
    assert res_root.status_code == 200
    assert "response" in res_root.json()
    assert "x-monitor-risk-score" in res_root.headers

    # Test /api/llm/proxy/chat
    res_proxy = client.post("/api/llm/proxy/chat", json=payload)
    assert res_proxy.status_code == 200
    assert "response" in res_proxy.json()
    assert "x-monitor-risk-score" in res_proxy.headers

def test_proxy_chat_streaming():
    """
    Verify streamed responses (stream: true):
    - Yields text chunks in real-time.
    - Buffers chunks and evaluates after completion.
    - Records interaction to database without breaking the client stream.
    """
    payload = {
        "model": "gemma2:2b",
        "messages": [
            {"role": "user", "content": "Say hello in one brief sentence."}
        ],
        "stream": True
    }

    initial_interactions = ResponseMonitor.list_interactions(limit=1)
    initial_latest_id = initial_interactions[0]["id"] if initial_interactions else 0

    with client.stream("POST", "/api/chat", json=payload) as response:
        assert response.status_code == 200
        chunks = [chunk for chunk in response.iter_text()]
        content = "".join(chunks)
        assert len(content.strip()) > 0

    # Verify stream post-evaluation recorded interaction in database
    updated_interactions = ResponseMonitor.list_interactions(limit=5)
    assert len(updated_interactions) > 0
    newest = updated_interactions[0]
    assert newest["id"] > initial_latest_id
    assert "hello" in newest["prompt"].lower()

def test_proxy_models_endpoint():
    """Verify GET /api/models and GET /models forward to custom LLM and return model list."""
    res_api = client.get("/api/models")
    assert res_api.status_code == 200
    data_api = res_api.json()
    assert "models" in data_api
    assert isinstance(data_api["models"], list)

    res_root = client.get("/models")
    assert res_root.status_code == 200
    data_root = res_root.json()
    assert "models" in data_root

def test_loop_prevention_on_port_8000():
    """Verify infinite proxy loops are detected and rejected when LLM_BASE_URL points to monitor port (8000)."""
    orig_env = os.environ.get("LLM_BASE_URL")
    try:
        os.environ["LLM_BASE_URL"] = "http://localhost:8000"
        payload = {"messages": [{"role": "user", "content": "test"}]}
        res = client.post("/api/chat", json=payload)
        assert res.status_code == 400
        assert "Infinite proxy loop detected" in res.json()["detail"]

        os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8000"
        res2 = client.post("/api/chat", json=payload)
        assert res2.status_code == 400
        assert "Infinite proxy loop detected" in res2.json()["detail"]
    finally:
        if orig_env:
            os.environ["LLM_BASE_URL"] = orig_env
        else:
            os.environ.pop("LLM_BASE_URL", None)

def test_upstream_unreachable_bad_gateway():
    """Verify upstream connection failures return HTTP 502 with clear diagnostic explanation."""
    orig_env = os.environ.get("LLM_BASE_URL")
    try:
        # Point to unused port where nothing is listening
        os.environ["LLM_BASE_URL"] = "http://127.0.0.1:58791"
        payload = {"messages": [{"role": "user", "content": "test"}]}
        res = client.post("/api/chat", json=payload)
        assert res.status_code == 502
        assert "Bad Gateway" in res.json()["detail"]
    finally:
        if orig_env:
            os.environ["LLM_BASE_URL"] = orig_env
        else:
            os.environ.pop("LLM_BASE_URL", None)

def test_evaluation_pipeline_relevance_and_evidence():
    """
    Verify evaluation metrics:
    - Lexical/topical relevance.
    - Explicit unavailable evidence reporting for open-domain queries (no fabricated evidence).
    - Grounded factual consistency against local datasets.
    """
    # 1. On-topic response has high relevance
    rel_high, w1 = ResponseMonitor.calculate_relevance("Explain neural network backpropagation", "Backpropagation computes the gradient of the loss function with respect to weights in a neural network.")
    assert rel_high >= 50.0
    assert len(w1) == 0

    # 2. Off-topic response triggers warning
    rel_low, w2 = ResponseMonitor.calculate_relevance("Calculate the square root of 144", "Tomorrow it will rain in Paris.")
    assert rel_low < 30.0
    assert len(w2) > 0
    assert "Low relevance" in w2[0]

    # 3. Open-domain prompt reporting unavailable evidence explicitly
    fact_score, ev_status, w3 = ResponseMonitor.check_factual_consistency("What is quantum entanglement?", "Quantum entanglement occurs when particles share physical states.")
    assert "unavailable" in ev_status.lower()
    assert "Ground truth evidence is not present in local datasets" in ev_status or "unavailable" in ev_status
    assert fact_score is None

    # 4. Detailed evaluation dictionary structure
    eval_res = ResponseMonitor.evaluate_response_detailed(
        prompt="Tell me about AI reliability",
        response_text="AI reliability ensures systems behave safely under degraded conditions.",
        input_risk_score=20.0,
        input_subsystem="image"
    )
    assert "risk_score" in eval_res
    assert "relevance_score" in eval_res
    assert "evidence_status" in eval_res
    assert "summary" in eval_res
    assert eval_res["relevance_score"] > 0

def test_interactions_list_endpoint():
    """Verify GET /api/llm/interactions returns captured records with all evaluation metrics."""
    res = client.get("/api/llm/interactions?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "interactions" in data
    assert "count" in data
    assert isinstance(data["interactions"], list)

def test_system_status_reflects_llm_state():
    """Verify GET /api/status displays LLM state and provider coupling."""
    res = client.get("/api/status")
    assert res.status_code == 200
    status = res.json()
    assert "llm_state" in status
    assert status["llm_state"] in ["normal", "caution", "warning", "ready", "unavailable"]

@pytest.mark.anyio
async def test_custom_local_provider_nova_mode():
    """Verify CustomLocalProvider auto-detects nova adapter on port 8080 and lists models."""
    prov = CustomLocalProvider(base_url="http://127.0.0.1:8080")
    is_ok, msg = await prov.check_health()
    assert is_ok is True
    assert "nova" in msg or "8080" in msg

    models = await prov.list_models()
    assert len(models) > 0
    assert any("gemma" in m or "qwen" in m for m in models)
