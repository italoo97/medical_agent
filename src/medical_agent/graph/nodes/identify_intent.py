from collections.abc import Callable, Coroutine
from typing import Any

from medical_agent.graph.state import AppointmentState
from medical_agent.prompts.v1.identify_intent import (
    Intent,
    build_system_prompt,
    build_user_prompt,
)
from medical_agent.services.base import BaseLLMService


def make_identify_intent_node(
    llm_service: BaseLLMService,
) -> Callable[[AppointmentState], Coroutine[Any, Any, dict[str, Intent]]]:
    async def identify_intent(
        state: AppointmentState,
    ) -> dict[str, Intent]:
        intent = await llm_service.generate_structured(
            system_prompt=build_system_prompt(),
            user_prompt=build_user_prompt(state['user_message']),
            schema=Intent,
        )
        return {'intent': intent}

    return identify_intent
