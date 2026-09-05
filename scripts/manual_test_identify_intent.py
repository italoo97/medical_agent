"""Teste manual, ponta a ponta, contra o OpenRouter de verdade.

Este script NAO faz parte da suite de testes automatizados (pytest) --
ele faz uma chamada de rede real, custa tokens de verdade, e serve
apenas para voce validar visualmente:
  1) que o OpenRouterService consegue conversar com o modelo e obter
     um Intent estruturado valido;
  2) que o trace aparece no dashboard do LangSmith.

Rode com:
    poetry run python scripts/manual_test_identify_intent.py
"""

import asyncio

import httpx
from dotenv import load_dotenv
from medical_agent.core.config import get_settings
from medical_agent.prompts.v1.identify_intent import (
    Intent,
    build_system_prompt,
    build_user_prompt,
)
from medical_agent.services.base import BaseLLMService
from medical_agent.services.calendar_client import Professional
from medical_agent.services.llm import OpenRouterService

load_dotenv()


async def main() -> None:
    settings = get_settings()

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        llm_service: BaseLLMService = OpenRouterService(settings, http_client)

        professionals = [
            Professional('cal-john', 'John Doe', 'Cardiology'),
            Professional('cal-jane', 'Jane Doe', 'Dermatology'),
            Professional('cal-richard', 'Richard Roe', 'Neurology'),
        ]

        user_message = (
            'Oi, queria cancelar minha consulta com o dr jhon, sou a '
            'Maria Santos'
        )

        intent = await llm_service.generate_structured(
            system_prompt=build_system_prompt(),
            user_prompt=build_user_prompt(user_message, professionals),
            schema=Intent,
        )

        print('--- Intent extraido ---')
        print(intent.model_dump_json(indent=2))


if __name__ == '__main__':
    asyncio.run(main())
