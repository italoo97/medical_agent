import asyncio

from medical_agent.graph.nodes.scheduler import make_scheduler_node
from medical_agent.graph.state import AppointmentState
from medical_agent.prompts.v1.identify_intent import Intent
from medical_agent.services.appointment_service import AppointmentService
from medical_agent.services.fake_calendar_client import FakeCalendarClient


def _build_state(
    intent: Intent,
) -> tuple[AppointmentState, AppointmentService]:
    service = AppointmentService(FakeCalendarClient())
    state: AppointmentState = {
        'user_message': '',
        'intent': intent,
        'professionals': service.list_professionals(),
        'calendar_id': None,
        'response': '',
        'error': None,
    }
    return state, service


def test_books_appointment_when_slot_is_available() -> None:
    intent = Intent(
        intent='schedule',
        professional_name='John Doe',
        patient_name='Maria Santos',
        date='2026-09-10',
        time='14:00',
        reason='check-up',
    )
    state, service = _build_state(intent)
    scheduler = make_scheduler_node(service)

    result = asyncio.run(scheduler(state))

    assert result['error'] is None
    assert result['calendar_id'] is not None


def test_returns_error_when_date_is_missing() -> None:
    intent = Intent(intent='schedule', professional_name='John Doe')
    state, service = _build_state(intent)
    scheduler = make_scheduler_node(service)

    result = asyncio.run(scheduler(state))

    assert result['error'] == 'Missing date or time to schedule.'


def test_returns_error_when_professional_not_found() -> None:
    intent = Intent(
        intent='schedule',
        professional_name='Nonexistent Doctor',
        date='2026-09-10',
        time='14:00',
    )
    state, service = _build_state(intent)
    scheduler = make_scheduler_node(service)

    result = asyncio.run(scheduler(state))

    assert result['error'] == 'No matching professional found.'


def test_returns_error_when_slot_is_already_booked() -> None:
    intent = Intent(
        intent='schedule',
        professional_name='John Doe',
        patient_name='Maria Santos',
        date='2026-09-10',
        time='14:00',
        reason='check-up',
    )
    state, service = _build_state(intent)
    scheduler = make_scheduler_node(service)

    asyncio.run(scheduler(state))
    result = asyncio.run(scheduler(state))

    assert result['error'] is not None
    assert result['error'] != 'No matching professional found.'
