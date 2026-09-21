import asyncio

import pytest
from medical_agent.core.config import get_settings
from medical_agent.graph import entry as entry_module


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preenche as env vars que Settings() exige, sem tocar rede real."""
    monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-test')
    monkeypatch.setenv('GOOGLE_SERVICE_ACCOUNT_JSON', '{}')
    monkeypatch.setenv('ADMIN_API_KEY', 'test-admin-key')
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_graph_factory_builds_a_compiled_graph_for_langgraph_dev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`langgraph dev` (via langgraph.json) chama esta fábrica sem
    argumentos -- ela precisa devolver o grafo já compilado, igual ao
    que `main.build_context()` monta, só que sem fechar nada sozinha.
    """

    class _FakeGoogleCalendarClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr(
        entry_module, 'GoogleCalendarClient', _FakeGoogleCalendarClient
    )

    compiled_graph = asyncio.run(entry_module.graph())

    assert hasattr(compiled_graph, 'ainvoke')
