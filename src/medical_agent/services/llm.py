import json
from typing import Any, Final, TypeVar, cast

import httpx
from langsmith import traceable
from pydantic import BaseModel, ValidationError

from medical_agent.core.config import Settings
from medical_agent.exceptions import UpstreamProviderError
from medical_agent.schemas.chat import ChatResponse
from medical_agent.services.base import BaseLLMService

OPENROUTER_CHAT_URL: Final[str] = (
    'https://openrouter.ai/api/v1/chat/completions'
)

SchemaT = TypeVar('SchemaT', bound=BaseModel)


class OpenRouterService(BaseLLMService):
    """Roteia prompts para o OpenRouter."""

    def __init__(
        self, settings: Settings, http_client: httpx.AsyncClient
    ) -> None:
        self._settings = settings
        self._http_client = http_client

    @traceable(name='openrouter_generate')
    async def generate(self, prompt: str) -> ChatResponse:
        data = await self._call([
            {'role': 'system', 'content': self._settings.system_prompt},
            {'role': 'user', 'content': prompt},
        ])
        content = str(data['choices'][0]['message']['content'])
        return ChatResponse(model=data['model'], content=content)

    @traceable(name='openrouter_generate_structured')
    async def generate_structured(  # type: ignore[override]
        self, system_prompt: str, user_prompt: str, schema: type[SchemaT]
    ) -> SchemaT:
        data = await self._call(
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            response_format={'type': 'json_object'},
        )
        raw_content = str(data['choices'][0]['message']['content'])

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as error:
            raise UpstreamProviderError(
                f'Model did not return valid JSON: {error}'
            ) from error

        try:
            return schema.model_validate(parsed)
        except ValidationError as error:
            raise UpstreamProviderError(
                f'Model JSON did not match expected schema: {error}'
            ) from error

    async def _call(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._build_payload(messages, response_format)

        try:
            response = await self._http_client.post(
                OPENROUTER_CHAT_URL,
                headers=self._build_headers(),
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise UpstreamProviderError(
                f'OpenRouter request failed: {error}'
            ) from error

        return cast(dict[str, Any], response.json())

    def _build_headers(self) -> dict[str, str]:
        api_key = self._settings.openrouter_api_key.get_secret_value()
        return {
            'Authorization': f'Bearer {api_key}',
            'HTTP-Referer': self._settings.http_referer,
            'X-Title': self._settings.x_title,
        }

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'models': self._settings.models,
            'messages': messages,
            'stream': False,
            'temperature': self._settings.temperature,
            'max_tokens': self._settings.max_tokens,
            'provider': self._settings.provider.model_dump(),
        }
        if response_format is not None:
            payload['response_format'] = response_format
        return payload
