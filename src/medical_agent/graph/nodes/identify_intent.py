from collections.abc import Callable, Coroutine
from typing import Any

from medical_agent.graph.state import AppointmentState
from medical_agent.prompts.v1.identify_intent import (
    Intent,
    build_system_prompt,
    build_user_prompt,
)
from medical_agent.services.base import BaseLLMService
from medical_agent.services.conversation_state import (
    ConversationStateStore,
    merge_intent,
)


def make_identify_intent_node(
    llm_service: BaseLLMService,
    conversation_state_store: ConversationStateStore,
) -> Callable[[AppointmentState], Coroutine[Any, Any, dict[str, Intent]]]:
    async def identify_intent(
        state: AppointmentState,
    ) -> dict[str, Intent]:
        new_intent = await llm_service.generate_structured(
            system_prompt=build_system_prompt(),
            user_prompt=build_user_prompt(
                state['user_message'], state['professionals']
            ),
            schema=Intent,
        )

        saved_intent = conversation_state_store.load(state['session_id'])
        if saved_intent is None and new_intent.patient_name:
            saved_intent = conversation_state_store.load_by_patient_name(
                new_intent.patient_name
            )

        merged_intent = merge_intent(saved_intent, new_intent)
        return {'intent': merged_intent}

    return identify_intent
