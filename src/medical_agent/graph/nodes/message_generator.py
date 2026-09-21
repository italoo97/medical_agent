from collections.abc import Callable, Coroutine
from typing import Any

from medical_agent.graph.state import AppointmentState
from medical_agent.prompts.v1.message_generator import (
    build_system_prompt,
    build_user_prompt,
)
from medical_agent.services.base import BaseLLMService

_DEFAULT_LANGUAGE = 'Portuguese'


def make_message_generator_node(
    llm_service: BaseLLMService,
) -> Callable[[AppointmentState], Coroutine[Any, Any, dict[str, str]]]:
    async def message_generator(
        state: AppointmentState,
    ) -> dict[str, str]:
        professional_name = None
        calendar_id = state['calendar_id']
        if calendar_id is not None:
            for professional in state['professionals']:
                if professional.calendar_id == calendar_id:
                    professional_name = professional.name
                    break

        intent = state['intent']
        language = (
            intent.language
            if intent is not None and intent.language
            else _DEFAULT_LANGUAGE
        )

        user_prompt = build_user_prompt(
            original_message=state['user_message'],
            intent=intent,
            professional_name=professional_name,
            error=state['error'],
            appointment_datetime=state['appointment_datetime'],
            language=language,
        )

        chat_response = await llm_service.generate(
            user_prompt, system_prompt=build_system_prompt()
        )

        return {'response': chat_response.content}

    return message_generator
