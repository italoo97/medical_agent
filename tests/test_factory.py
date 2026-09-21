import asyncio
from datetime import datetime
from pathlib import Path

from medical_agent.graph.factory import build_graph
from medical_agent.graph.state import AppointmentState
from medical_agent.prompts.v1.identify_intent import Intent
from medical_agent.services.appointment_service import AppointmentService
from medical_agent.services.conversation_state import ConversationStateStore
from medical_agent.services.fake_calendar_client import FakeCalendarClient

from tests._fakes import FakeLLMService


def _build(
    tmp_path: Path, intents: list[Intent], chat_content: str = 'ok'
) -> tuple:
    appointment_service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    llm = FakeLLMService(intents=intents, chat_content=chat_content)
    graph = build_graph(llm, appointment_service, store)
    return graph, appointment_service, store


def _initial_state(
    session_id: str, professionals: list, user_message: str = 'oi'
) -> AppointmentState:
    return {
        'session_id': session_id,
        'user_message': user_message,
        'intent': None,
        'professionals': professionals,
        'calendar_id': None,
        'appointment_datetime': None,
        'response': '',
        'error': None,
    }


def test_schedule_intent_routes_through_the_scheduler(tmp_path: Path) -> None:
    graph, appointment_service, _store = _build(
        tmp_path,
        [
            Intent(
                intent='schedule',
                professional_name='John Doe',
                patient_name='Maria Santos',
                date='2026-09-10',
                time='16:00',
            )
        ],
        chat_content='Consulta marcada!',
    )
    professionals = appointment_service.list_professionals()

    final_state = asyncio.run(
        graph.ainvoke(_initial_state('session-1', professionals))
    )

    assert final_state['error'] is None
    assert final_state['response'] == 'Consulta marcada!'


def test_cancel_intent_routes_through_the_canceller(tmp_path: Path) -> None:
    graph, appointment_service, _store = _build(
        tmp_path,
        [
            Intent(
                intent='cancel',
                professional_name='John Doe',
                patient_name='Maria Santos',
                date='2026-09-10',
                time='16:00',
            )
        ],
    )
    professionals = appointment_service.list_professionals()
    appointment_service.book_appointment(
        professionals[0].calendar_id,
        datetime(2026, 9, 10, 16, 0),
        'Maria Santos',
        'check-up',
    )

    final_state = asyncio.run(
        graph.ainvoke(_initial_state('session-1', professionals))
    )

    assert final_state['error'] is None


def test_check_intent_routes_through_the_checker(tmp_path: Path) -> None:
    graph, appointment_service, _store = _build(
        tmp_path, [Intent(intent='check', patient_name='Maria Santos')]
    )
    professionals = appointment_service.list_professionals()

    final_state = asyncio.run(
        graph.ainvoke(_initial_state('session-1', professionals))
    )

    assert final_state['error'] == (
        'No upcoming appointment found for this patient.'
    )


def test_unknown_intent_skips_straight_to_the_message_generator(
    tmp_path: Path,
) -> None:
    graph, appointment_service, _store = _build(
        tmp_path, [Intent(intent='unknown')], chat_content='Nao entendi.'
    )
    professionals = appointment_service.list_professionals()

    final_state = asyncio.run(
        graph.ainvoke(_initial_state('session-1', professionals))
    )

    assert final_state['response'] == 'Nao entendi.'
    assert final_state['error'] is None


def test_cancels_without_repeating_any_detail_in_the_same_session(
    tmp_path: Path,
) -> None:
    graph, appointment_service, _store = _build(
        tmp_path,
        [
            Intent(
                intent='schedule',
                professional_name='John Doe',
                patient_name='Maria Santos',
                date='2026-09-10',
                time='16:00',
            ),
            Intent(intent='cancel'),
        ],
        chat_content='ok',
    )
    professionals = appointment_service.list_professionals()

    schedule_result = asyncio.run(
        graph.ainvoke(
            _initial_state(
                'session-1', professionals, 'quero marcar com o John Doe'
            )
        )
    )
    assert schedule_result['error'] is None

    # Segunda mensagem, mesma sessao, sem repetir nome nem profissional.
    cancel_result = asyncio.run(
        graph.ainvoke(
            _initial_state('session-1', professionals, 'cancela por favor')
        )
    )

    assert cancel_result['error'] is None
    assert cancel_result['calendar_id'] == schedule_result['calendar_id']
