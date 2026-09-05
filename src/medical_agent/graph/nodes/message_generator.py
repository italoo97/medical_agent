from collections.abc import Callable, Coroutine
from typing import Any

from medical_agent.graph.state import AppointmentState
from medical_agent.prompts.v1.message_generator import (
    build_system_prompt,
    build_user_prompt,
)
from medical_agent.services.base import BaseLLMService


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

        user_prompt = build_user_prompt(
            original_message=state['user_message'],
            intent=state['intent'],
            professional_name=professional_name,
            error=state['error'],
        )

        chat_response = await llm_service.generate(
            user_prompt, system_prompt=build_system_prompt()
        )

        return {'response': chat_response.content}

    return message_generator
