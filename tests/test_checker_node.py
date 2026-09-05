import asyncio
from datetime import datetime

from medical_agent.graph.nodes.checker import make_checker_node
from medical_agent.graph.state import AppointmentState
from medical_agent.prompts.v1.identify_intent import Intent
from medical_agent.services.appointment_service import AppointmentService
from medical_agent.services.fake_calendar_client import FakeCalendarClient


def _calendar_id_for(service: AppointmentService, name: str) -> str:
    for professional in service.list_professionals():
        if name in professional.name:
            return professional.calendar_id
    raise AssertionError(f'no fixture professional matches "{name}"')


def _build_state(
    service: AppointmentService, intent: Intent
) -> AppointmentState:
    return {
        'user_message': '',
        'intent': intent,
        'professionals': service.list_professionals(),
        'calendar_id': None,
        'appointment_datetime': None,
        'response': '',
        'error': None,
    }


def test_finds_upcoming_appointment_for_patient() -> None:
    service = AppointmentService(FakeCalendarClient())
    calendar_id = _calendar_id_for(service, 'Jane Doe')
    service.book_appointment(
        calendar_id,
        datetime(2026, 9, 10, 15, 0),
        'Maria Santos',
        'check-up',
    )
    intent = Intent(intent='check', patient_name='Maria Santos')
    state = _build_state(service, intent)
    checker = make_checker_node(service)

    result = asyncio.run(checker(state))

    assert result['error'] is None
    assert result['calendar_id'] == calendar_id
    assert (
        result['appointment_datetime']
        == datetime(2026, 9, 10, 15, 0).isoformat()
    )


def test_returns_error_when_patient_name_is_missing() -> None:
    service = AppointmentService(FakeCalendarClient())
    intent = Intent(intent='check')
    state = _build_state(service, intent)
    checker = make_checker_node(service)

    result = asyncio.run(checker(state))

    assert result['error'] == (
        'Não consegui verificar sua consulta porque faltou '
        'informar: seu nome.'
    )


def test_returns_error_when_no_appointment_found() -> None:
    service = AppointmentService(FakeCalendarClient())
    intent = Intent(intent='check', patient_name='Ninguem Marcou')
    state = _build_state(service, intent)
    checker = make_checker_node(service)

    result = asyncio.run(checker(state))

    assert result['error'] == 'No upcoming appointment found for this patient.'
