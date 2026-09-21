import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from medical_agent import main as main_module
from medical_agent.core.config import get_settings
from medical_agent.exceptions import UpstreamProviderError
from medical_agent.graph.factory import build_graph
from medical_agent.main import AppContext, app
from medical_agent.prompts.v1.identify_intent import Intent
from medical_agent.schemas.chat import ChatResponse
from medical_agent.services.appointment_service import AppointmentService
from medical_agent.services.base import BaseLLMService
from medical_agent.services.conversation_state import ConversationStateStore
from medical_agent.services.fake_calendar_client import FakeCalendarClient

from tests._fakes import FakeLLMService

ADMIN_KEY = 'test-admin-key'


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preenche as env vars que Settings() exige, sem tocar rede real."""
    monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-test')
    monkeypatch.setenv('GOOGLE_SERVICE_ACCOUNT_JSON', '{}')
    monkeypatch.setenv('ADMIN_API_KEY', ADMIN_KEY)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _fake_build_context(
    tmp_path: Path, intents: list[Intent], chat_content: str = 'ok'
):
    @asynccontextmanager
    async def build_context() -> AsyncIterator[AppContext]:
        appointment_service = AppointmentService(FakeCalendarClient())
        store = ConversationStateStore(str(tmp_path / 'state.db'))
        llm = FakeLLMService(intents=intents, chat_content=chat_content)
        graph = build_graph(llm, appointment_service, store)
        yield AppContext(
            graph=graph,
            appointment_service=appointment_service,
            conversation_state_store=store,
        )

    return build_context


class _RateLimitedLLMService(BaseLLMService):
    """Simula o provedor de LLM respondendo 429 (limite de requisições)."""

    async def generate(  # noqa: PLR6301
        self, prompt: str, system_prompt: str | None = None
    ) -> ChatResponse:
        raise UpstreamProviderError(
            "OpenRouter request failed: Client error '429 Too Many "
            "Requests'"
        )

    async def generate_structured(  # type: ignore[override] # noqa: PLR6301
        self, system_prompt: str, user_prompt: str, schema: type
    ) -> object:
        raise UpstreamProviderError(
            "OpenRouter request failed: Client error '429 Too Many "
            "Requests'"
        )


def _fake_build_context_with_failing_llm(tmp_path: Path):
    @asynccontextmanager
    async def build_context() -> AsyncIterator[AppContext]:
        appointment_service = AppointmentService(FakeCalendarClient())
        store = ConversationStateStore(str(tmp_path / 'state.db'))
        graph = build_graph(
            _RateLimitedLLMService(), appointment_service, store
        )
        yield AppContext(
            graph=graph,
            appointment_service=appointment_service,
            conversation_state_store=store,
        )

    return build_context


def test_chat_endpoint_returns_structured_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main_module,
        'build_context',
        _fake_build_context(
            tmp_path,
            [Intent(intent='check', patient_name='Maria Santos')],
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            '/chat',
            json={
                'question': 'tenho consulta marcada?',
                'session_id': 'session-1',
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body['intent'] == 'check'
    assert body['error'] == ('No upcoming appointment found for this patient.')


def test_chat_endpoint_remembers_missing_fields_across_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main_module,
        'build_context',
        _fake_build_context(
            tmp_path,
            [
                Intent(intent='cancel', professional_name='John Doe'),
                Intent(intent='unknown', patient_name='Maria Santos'),
            ],
        ),
    )

    with TestClient(app) as client:
        first = client.post(
            '/chat',
            json={
                'question': 'quero cancelar minha consulta',
                'session_id': 'session-1',
            },
        )
        second = client.post(
            '/chat',
            json={
                'question': 'sou a Maria Santos',
                'session_id': 'session-1',
            },
        )

    # A primeira mensagem falha por falta do nome do paciente...
    assert 'seu nome' in str(first.json()['error'])
    # ...e a segunda, mesmo sem repetir o profissional, completa o pedido
    # porque o estado da sessao anterior foi reaproveitado.
    assert second.json()['intent'] == 'cancel'


def test_chat_endpoint_returns_a_clean_error_when_the_llm_is_rate_limited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main_module,
        'build_context',
        _fake_build_context_with_failing_llm(tmp_path),
    )

    with TestClient(app) as client:
        response = client.post(
            '/chat',
            json={'question': 'quero marcar uma consulta', 'session_id': 's1'},
        )

    assert response.status_code == 502
    assert '429' in response.json()['detail']


def test_admin_register_and_remove_professional_are_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main_module,
        'build_context',
        _fake_build_context(tmp_path, [Intent(intent='unknown')]),
    )

    with TestClient(app) as client:
        calendar_id = 'fake-calendar-john'
        headers = {'x-admin-key': ADMIN_KEY}

        first_register = client.post(
            f'/admin/professionals/{calendar_id}', headers=headers
        )
        second_register = client.post(
            f'/admin/professionals/{calendar_id}', headers=headers
        )
        first_remove = client.delete(
            f'/admin/professionals/{calendar_id}', headers=headers
        )
        second_remove = client.delete(
            f'/admin/professionals/{calendar_id}', headers=headers
        )

    assert first_register.json() == {'status': 'registered'}
    assert second_register.json() == {'status': 'already_registered'}
    assert first_remove.json() == {'status': 'removed'}
    assert second_remove.json() == {'status': 'not_registered'}


def test_admin_endpoint_rejects_wrong_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main_module,
        'build_context',
        _fake_build_context(tmp_path, [Intent(intent='unknown')]),
    )

    with TestClient(app) as client:
        response = client.post(
            '/admin/professionals/fake-calendar-john',
            headers={'x-admin-key': 'wrong-key'},
        )

    assert response.status_code == 401


def test_run_cli_processes_one_message_then_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        main_module,
        'build_context',
        _fake_build_context(
            tmp_path, [Intent(intent='unknown')], chat_content='Oi!'
        ),
    )
    inputs = iter(['ola'])

    def fake_input(prompt: str = '') -> str:
        try:
            return next(inputs)
        except StopIteration:
            raise KeyboardInterrupt from None

    monkeypatch.setattr('builtins.input', fake_input)

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(main_module.run_cli())

    captured = capsys.readouterr()
    assert 'Agente: Oi!' in captured.out


def test_run_cli_survives_an_upstream_error_and_keeps_the_repl_alive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        main_module,
        'build_context',
        _fake_build_context_with_failing_llm(tmp_path),
    )
    inputs = iter(['quero marcar uma consulta'])

    def fake_input(prompt: str = '') -> str:
        try:
            return next(inputs)
        except StopIteration:
            raise KeyboardInterrupt from None

    monkeypatch.setattr('builtins.input', fake_input)

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(main_module.run_cli())

    captured = capsys.readouterr()
    assert 'problema para falar com o provedor de IA' in captured.out
    assert '429' in captured.out


def test_build_context_wires_the_real_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """So mocka o GoogleCalendarClient (ele fala com a rede no __init__).

    O OpenRouterService nao chama rede na construcao -- so no primeiro
    generate()/generate_structured() -- entao pode ser o real aqui.
    """

    class _FakeGoogleCalendarClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr(
        main_module, 'GoogleCalendarClient', _FakeGoogleCalendarClient
    )

    async def _run() -> None:
        async with main_module.build_context() as context:
            assert isinstance(context, AppContext)
            assert context.appointment_service is not None
            assert context.conversation_state_store is not None

    asyncio.run(_run())
