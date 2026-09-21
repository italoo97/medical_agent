import asyncio
from pathlib import Path

from medical_agent.graph.nodes.identify_intent import (
    make_identify_intent_node,
)
from medical_agent.graph.state import AppointmentState
from medical_agent.prompts.v1.identify_intent import Intent
from medical_agent.services.conversation_state import ConversationStateStore

from tests._fakes import FakeLLMService


def _state(session_id: str, user_message: str = 'oi') -> AppointmentState:
    return {
        'session_id': session_id,
        'user_message': user_message,
        'intent': None,
        'professionals': [],
        'calendar_id': None,
        'appointment_datetime': None,
        'response': '',
        'error': None,
    }


def test_returns_llm_intent_when_nothing_was_saved(tmp_path: Path) -> None:
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    llm = FakeLLMService(
        intents=[Intent(intent='schedule', patient_name='Maria')]
    )
    node = make_identify_intent_node(llm, store)

    result = asyncio.run(node(_state('session-1')))

    assert result['intent'].patient_name == 'Maria'


def test_merges_with_state_saved_under_the_same_session(
    tmp_path: Path,
) -> None:
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    store.save(
        'session-1',
        Intent(
            intent='cancel',
            professional_name='John Doe',
            patient_name='Maria',
        ),
    )
    llm = FakeLLMService(intents=[Intent(intent='unknown', time='16:00')])
    node = make_identify_intent_node(llm, store)

    result = asyncio.run(node(_state('session-1')))

    merged = result['intent']
    assert merged.intent == 'cancel'
    assert merged.professional_name == 'John Doe'
    assert merged.patient_name == 'Maria'
    assert merged.time == '16:00'


def test_falls_back_to_patient_name_when_session_has_nothing_saved(
    tmp_path: Path,
) -> None:
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    store.save(
        'other-session',
        Intent(
            intent='schedule',
            professional_name='John Doe',
            patient_name='Maria',
        ),
    )
    llm = FakeLLMService(
        intents=[Intent(intent='unknown', patient_name='Maria')]
    )
    node = make_identify_intent_node(llm, store)

    result = asyncio.run(node(_state('brand-new-session')))

    merged = result['intent']
    assert merged.intent == 'schedule'
    assert merged.professional_name == 'John Doe'


def test_does_not_fall_back_when_new_message_has_no_patient_name(
    tmp_path: Path,
) -> None:
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    store.save(
        'other-session',
        Intent(intent='schedule', patient_name='Maria'),
    )
    llm = FakeLLMService(intents=[Intent(intent='unknown')])
    node = make_identify_intent_node(llm, store)

    result = asyncio.run(node(_state('brand-new-session')))

    assert result['intent'].patient_name is None
