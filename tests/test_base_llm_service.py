import asyncio

import pytest
from medical_agent.schemas.chat import ChatResponse
from medical_agent.services.base import BaseLLMService


class _IncompleteLLMService(BaseLLMService):
    """So existe para exercitar o `raise NotImplementedError` da base."""

    async def generate(
        self, prompt: str, system_prompt: str | None = None
    ) -> ChatResponse:
        return await BaseLLMService.generate(self, prompt, system_prompt)

    async def generate_structured(self, system_prompt, user_prompt, schema):  # type: ignore[override]
        return await BaseLLMService.generate_structured(
            self, system_prompt, user_prompt, schema
        )


def test_generate_is_not_implemented_on_the_base_class() -> None:
    service = _IncompleteLLMService()

    with pytest.raises(NotImplementedError):
        asyncio.run(service.generate('oi'))


def test_generate_structured_is_not_implemented_on_the_base_class() -> None:
    service = _IncompleteLLMService()

    with pytest.raises(NotImplementedError):
        asyncio.run(service.generate_structured('s', 'u', ChatResponse))
