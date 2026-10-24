import pytest
import asyncio
from backend.app.llm_providers.registry import get_provider_registry
from backend.app.llm_providers.mock_local import OfflineMockProvider
from backend.app.llm_providers.response_monitor import ResponseMonitor

def test_mock_local_provider_generation():
    async def _run():
        provider = OfflineMockProvider()
        ok, msg = await provider.check_health()
        assert ok is True

        models = await provider.list_models()
        assert len(models) > 0

        res = await provider.generate(prompt="Describe this component", model=models[0])
        assert len(res.response_text) > 0
        assert res.provider == "mock_local"

    asyncio.run(_run())

def test_mock_local_provider_streaming():
    async def _run():
        provider = OfflineMockProvider()
        models = await provider.list_models()
        chunks = []
        async for chunk in provider.generate_stream(prompt="Test prompt", model=models[0]):
            chunks.append(chunk)

        full_text = "".join(chunks)
        assert len(full_text) > 0

    asyncio.run(_run())

def test_response_monitor_uncertainty_and_input_risk():
    # Model expressing explicit uncertainty
    uncertain_response = "I am not certain about the answer because the image details are blurry and unclear."
    risk, warnings, is_uncert, is_empty = ResponseMonitor.evaluate_response(
        prompt="What is this?",
        response_text=uncertain_response,
        input_risk_score=75.0,
        input_subsystem="image"
    )

    assert is_uncert is True
    assert is_empty is False
    assert risk > 50.0
    assert any("Uncertainty detected" in w for w in warnings)
    assert any("Input hazard warning" in w for w in warnings)

def test_response_monitor_empty_response():
    empty_response = "   "
    risk, warnings, is_uncert, is_empty = ResponseMonitor.evaluate_response(
        prompt="Tell me something",
        response_text=empty_response
    )

    assert is_empty is True
    assert risk >= 80.0
