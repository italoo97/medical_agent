import asyncio

from medical_agent.graph.nodes.message_generator import (
    make_message_generator_node,
)
from medical_agent.graph.state import AppointmentState
from medical_agent.prompts.v1.identify_intent import Intent
from medical_agent.services.calendar_client import Professional

from tests._fakes import FakeLLMService


def _state(**overrides: object) -> AppointmentState:
    base: AppointmentState = {
        'session_id': 'session-1',
        'user_message': 'quero marcar uma consulta',
        'intent': Intent(intent='schedule'),
        'professionals': [],
        'calendar_id': None,
        'appointment_datetime': None,
        'response': '',
        'error': None,
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def test_returns_the_llm_generated_response() -> None:
    llm = FakeLLMService(chat_content='Consulta confirmada!')
    node = make_message_generator_node(llm)

    result = asyncio.run(node(_state()))

    assert result == {'response': 'Consulta confirmada!'}


def test_resolves_the_professional_name_from_the_calendar_id() -> None:
    llm = FakeLLMService(chat_content='ok')
    node = make_message_generator_node(llm)
    professionals = [
        Professional('cal-john', 'John Doe', 'Cardiology'),
        Professional('cal-jane', 'Jane Doe', 'Dermatology'),
    ]

    asyncio.run(
        node(_state(professionals=professionals, calendar_id='cal-jane'))
    )

    assert 'Jane Doe' in llm.received_user_prompts[0]


def test_leaves_professional_name_unresolved_when_calendar_id_is_unknown() -> (
    None
):
    llm = FakeLLMService(chat_content='ok')
    node = make_message_generator_node(llm)
    professionals = [Professional('cal-john', 'John Doe', 'Cardiology')]

    asyncio.run(
        node(_state(professionals=professionals, calendar_id='cal-unknown'))
    )

    assert 'not resolved' in llm.received_user_prompts[0]


def test_uses_the_language_detected_on_the_intent() -> None:
    llm = FakeLLMService(chat_content='ok')
    node = make_message_generator_node(llm)

    asyncio.run(
        node(_state(intent=Intent(intent='schedule', language='English')))
    )

    assert 'Reply language: English' in llm.received_user_prompts[0]


def test_defaults_to_portuguese_when_the_intent_has_no_language() -> None:
    llm = FakeLLMService(chat_content='ok')
    node = make_message_generator_node(llm)

    asyncio.run(node(_state(intent=Intent(intent='schedule', language=None))))

    assert 'Reply language: Portuguese' in llm.received_user_prompts[0]


def test_defaults_to_portuguese_when_there_is_no_intent_at_all() -> None:
    llm = FakeLLMService(chat_content='ok')
    node = make_message_generator_node(llm)

    asyncio.run(node(_state(intent=None)))

    assert 'Reply language: Portuguese' in llm.received_user_prompts[0]
