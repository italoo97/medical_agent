import asyncio
from datetime import datetime

from medical_agent.graph.nodes.canceller import make_canceller_node
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
        'response': '',
        'error': None,
    }


def test_cancels_existing_appointment() -> None:
    service = AppointmentService(FakeCalendarClient())
    calendar_id = _calendar_id_for(service, 'Jane Doe')
    service.book_appointment(
        calendar_id,
        datetime(2026, 9, 10, 15, 0),
        'Maria Santos',
        'check-up',
    )
    intent = Intent(
        intent='cancel',
        professional_name='Jane Doe',
        patient_name='Maria Santos',
        date='2026-09-10',
        time='15:00',
    )
    state = _build_state(service, intent)
    canceller = make_canceller_node(service)

    result = asyncio.run(canceller(state))

    assert result['error'] is None
    assert result['calendar_id'] == calendar_id


def test_returns_error_when_date_is_missing() -> None:
    service = AppointmentService(FakeCalendarClient())
    intent = Intent(intent='cancel', professional_name='Jane Doe')
    state = _build_state(service, intent)
    canceller = make_canceller_node(service)

    result = asyncio.run(canceller(state))

    assert result['error'] == 'Missing date or time to cancel.'


def test_returns_error_when_professional_not_found() -> None:
    service = AppointmentService(FakeCalendarClient())
    intent = Intent(
        intent='cancel',
        professional_name='Nonexistent Doctor',
        date='2026-09-10',
        time='15:00',
    )
    state = _build_state(service, intent)
    canceller = make_canceller_node(service)

    result = asyncio.run(canceller(state))

    assert result['error'] == 'No matching professional found.'


def test_returns_error_when_appointment_not_found() -> None:
    service = AppointmentService(FakeCalendarClient())
    intent = Intent(
        intent='cancel',
        professional_name='Jane Doe',
        patient_name='Maria Santos',
        date='2026-09-10',
        time='15:00',
    )
    state = _build_state(service, intent)
    canceller = make_canceller_node(service)

    result = asyncio.run(canceller(state))

    assert result['error'] is not None
    assert result['error'] != 'No matching professional found.'
