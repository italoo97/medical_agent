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
