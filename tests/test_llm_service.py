import asyncio
import json

import httpx
import pytest
from medical_agent.core.config import Settings
from medical_agent.exceptions import UpstreamProviderError
from medical_agent.prompts.v1.identify_intent import Intent
from medical_agent.services.llm import OpenRouterService


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        # _env_file=None isola o teste do .env real da maquina; sem isso
        # o Settings le o arquivo e valores locais vazam para dentro dos
        # testes (o suite passa aqui e falha no CI, ou o contrario).
        '_env_file': None,
        'OPENROUTER_API_KEY': 'sk-test',
        'GOOGLE_SERVICE_ACCOUNT_JSON': '{}',
        'ADMIN_API_KEY': 'admin-key',
    }
    values.update(overrides)
    return Settings(**values)


def _service(handler, settings: Settings | None = None) -> OpenRouterService:
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    return OpenRouterService(settings or _settings(), http_client)


def _capturing_handler(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured['headers'] = request.headers
        captured['url'] = str(request.url)
        return httpx.Response(
            200,
            json={'model': 'm', 'choices': [{'message': {'content': 'ok'}}]},
        )

    return handler


def test_generate_returns_a_chat_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                'model': 'nvidia/nemotron',
                'choices': [{'message': {'content': 'Ola!'}}],
            },
        )

    service = _service(handler)

    response = asyncio.run(service.generate('oi', system_prompt='seja gentil'))

    assert response.model == 'nvidia/nemotron'
    assert response.content == 'Ola!'


def test_generate_sends_the_expected_headers_and_payload() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured['headers'] = request.headers
        captured['body'] = json.loads(request.content)
        return httpx.Response(
            200,
            json={'model': 'm', 'choices': [{'message': {'content': 'ok'}}]},
        )

    service = _service(handler)
    asyncio.run(service.generate('oi'))

    assert captured['headers']['authorization'] == 'Bearer sk-test'
    assert captured['body']['messages'][1] == {
        'role': 'user',
        'content': 'oi',
    }
    assert captured['body']['stream'] is False


def test_generate_structured_returns_the_parsed_schema() -> None:
    payload = {
        'intent': 'schedule',
        'patient_name': 'Maria Santos',
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                'model': 'm',
                'choices': [{'message': {'content': json.dumps(payload)}}],
            },
        )

    service = _service(handler)

    intent = asyncio.run(
        service.generate_structured(
            system_prompt='s', user_prompt='u', schema=Intent
        )
    )

    assert intent.intent == 'schedule'
    assert intent.patient_name == 'Maria Santos'


def test_generate_structured_raises_when_content_is_not_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                'model': 'm',
                'choices': [{'message': {'content': 'not json at all'}}],
            },
        )

    service = _service(handler)

    with pytest.raises(UpstreamProviderError):
        asyncio.run(
            service.generate_structured(
                system_prompt='s', user_prompt='u', schema=Intent
            )
        )


def test_generate_structured_raises_when_schema_does_not_match() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                'model': 'm',
                'choices': [
                    {'message': {'content': json.dumps({'intent': 'nope'})}}
                ],
            },
        )

    service = _service(handler)

    with pytest.raises(UpstreamProviderError):
        asyncio.run(
            service.generate_structured(
                system_prompt='s', user_prompt='u', schema=Intent
            )
        )


def test_call_raises_upstream_error_on_http_status_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={'error': {'message': 'boom'}})

    service = _service(handler)

    with pytest.raises(UpstreamProviderError):
        asyncio.run(service.generate('oi'))


def test_call_raises_upstream_error_when_response_has_no_choices() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'error': {'message': 'sem creditos'}})

    service = _service(handler)

    with pytest.raises(UpstreamProviderError, match='sem creditos'):
        asyncio.run(service.generate('oi'))


def test_adds_the_gateway_header_when_a_token_is_configured() -> None:
    captured: dict = {}
    settings = _settings(CF_AIG_TOKEN='aig-token')

    service = _service(_capturing_handler(captured), settings)
    asyncio.run(service.generate('oi'))

    assert captured['headers']['cf-aig-authorization'] == 'Bearer aig-token'


def test_omits_the_gateway_header_when_no_token_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sem token, a requisicao sai identica a de antes do gateway.

    E o teste que protege a rota de rollback: apagar a variavel de
    ambiente basta para voltar a falar direto com o OpenRouter.

    O delenv e necessario porque main.py roda load_dotenv() no import,
    o que despeja o .env dentro de os.environ antes dos testes rodarem;
    _env_file=None desliga a leitura do arquivo, nao das variaveis.
    """
    monkeypatch.delenv('CF_AIG_TOKEN', raising=False)
    captured: dict = {}

    service = _service(_capturing_handler(captured))
    asyncio.run(service.generate('oi'))

    assert 'cf-aig-authorization' not in captured['headers']


def test_builds_the_chat_url_from_the_configured_base_url() -> None:
    """A barra final na base nao pode virar // no caminho."""
    captured: dict = {}
    settings = _settings(OPENROUTER_BASE_URL='https://gw.example/openrouter/')

    service = _service(_capturing_handler(captured), settings)
    asyncio.run(service.generate('oi'))

    assert captured['url'] == (
        'https://gw.example/openrouter/chat/completions'
    )


def test_defaults_to_calling_openrouter_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('OPENROUTER_BASE_URL', raising=False)
    captured: dict = {}

    service = _service(_capturing_handler(captured))
    asyncio.run(service.generate('oi'))

    assert captured['url'] == ('https://openrouter.ai/api/v1/chat/completions')
