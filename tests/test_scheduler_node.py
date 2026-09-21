import asyncio
from pathlib import Path

from medical_agent.graph.nodes.scheduler import make_scheduler_node
from medical_agent.graph.state import AppointmentState
from medical_agent.prompts.v1.identify_intent import Intent
from medical_agent.services.appointment_service import AppointmentService
from medical_agent.services.conversation_state import ConversationStateStore
from medical_agent.services.fake_calendar_client import FakeCalendarClient


def _build_state(
    intent: Intent | None,
    tmp_path: Path,
    session_id: str = 'session-1',
) -> tuple[AppointmentState, AppointmentService, ConversationStateStore]:
    service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    state: AppointmentState = {
        'session_id': session_id,
        'user_message': '',
        'intent': intent,
        'professionals': service.list_professionals(),
        'calendar_id': None,
        'appointment_datetime': None,
        'response': '',
        'error': None,
    }
    return state, service, store


def test_books_appointment_when_slot_is_available(tmp_path: Path) -> None:
    intent = Intent(
        intent='schedule',
        professional_name='John Doe',
        patient_name='Maria Santos',
        date='2026-09-10',
        time='14:00',
        reason='check-up',
    )
    state, service, store = _build_state(intent, tmp_path)
    scheduler = make_scheduler_node(service, store)

    result = asyncio.run(scheduler(state))

    assert result['error'] is None
    assert result['calendar_id'] is not None


def test_remembers_the_booking_for_this_session(tmp_path: Path) -> None:
    intent = Intent(
        intent='schedule',
        professional_name='John Doe',
        patient_name='Maria Santos',
        date='2026-09-10',
        time='14:00',
        reason='check-up',
    )
    state, service, store = _build_state(intent, tmp_path)
    scheduler = make_scheduler_node(service, store)

    result = asyncio.run(scheduler(state))

    booking = store.load_last_booking('session-1')
    assert booking is not None
    calendar_id, _start, patient_name = booking
    assert calendar_id == result['calendar_id']
    assert patient_name == 'Maria Santos'


def test_returns_error_listing_every_missing_field(tmp_path: Path) -> None:
    intent = Intent(intent='schedule', professional_name='John Doe')
    state, service, store = _build_state(intent, tmp_path)
    scheduler = make_scheduler_node(service, store)

    result = asyncio.run(scheduler(state))

    assert result['error'] == (
        'Não consegui agendar a consulta porque faltou informar: '
        'a data, o horário, seu nome.'
    )


def test_returns_error_when_only_patient_name_is_missing(
    tmp_path: Path,
) -> None:
    intent = Intent(
        intent='schedule',
        professional_name='John Doe',
        date='2026-09-10',
        time='14:00',
    )
    state, service, store = _build_state(intent, tmp_path)
    scheduler = make_scheduler_node(service, store)

    result = asyncio.run(scheduler(state))

    assert result['error'] == (
        'Não consegui agendar a consulta porque faltou informar: seu nome.'
    )


def test_returns_error_when_professional_not_found(tmp_path: Path) -> None:
    intent = Intent(
        intent='schedule',
        professional_name='Nonexistent Doctor',
        patient_name='Maria Santos',
        date='2026-09-10',
        time='14:00',
    )
    state, service, store = _build_state(intent, tmp_path)
    scheduler = make_scheduler_node(service, store)

    result = asyncio.run(scheduler(state))

    assert result['error'] == 'No matching professional found.'


def test_returns_error_when_slot_is_already_booked(tmp_path: Path) -> None:
    intent = Intent(
        intent='schedule',
        professional_name='John Doe',
        patient_name='Maria Santos',
        date='2026-09-10',
        time='14:00',
        reason='check-up',
    )
    state, service, store = _build_state(intent, tmp_path)
    scheduler = make_scheduler_node(service, store)

    asyncio.run(scheduler(state))
    result = asyncio.run(scheduler(state))

    assert result['error'] is not None
    assert result['error'] != 'No matching professional found.'


def test_returns_error_when_intent_is_none(tmp_path: Path) -> None:
    state, service, store = _build_state(None, tmp_path)
    scheduler = make_scheduler_node(service, store)

    result = asyncio.run(scheduler(state))

    assert result['error'] == (
        'Não consegui agendar a consulta: não entendi seu pedido.'
    )


def test_matches_professional_by_specialty_when_name_not_given(
    tmp_path: Path,
) -> None:
    intent = Intent(
        intent='schedule',
        specialty='Dermatology',
        patient_name='Maria Santos',
        date='2026-09-10',
        time='14:00',
    )
    state, service, store = _build_state(intent, tmp_path)
    scheduler = make_scheduler_node(service, store)

    result = asyncio.run(scheduler(state))

    assert result['error'] is None


def test_matches_professional_despite_a_full_name_typo(
    tmp_path: Path,
) -> None:
    intent = Intent(
        intent='schedule',
        professional_name='Jhon Doe',  # fixture esta como 'John Doe'
        patient_name='Maria Santos',
        date='2026-09-10',
        time='14:00',
    )
    state, service, store = _build_state(intent, tmp_path)
    scheduler = make_scheduler_node(service, store)

    result = asyncio.run(scheduler(state))

    assert result['error'] is None


def test_matches_professional_from_a_bare_first_name_typo(
    tmp_path: Path,
) -> None:
    intent = Intent(
        intent='schedule',
        professional_name='Jhon',
        patient_name='Maria Santos',
        date='2026-09-10',
        time='14:00',
    )
    state, service, store = _build_state(intent, tmp_path)
    scheduler = make_scheduler_node(service, store)

    result = asyncio.run(scheduler(state))

    assert result['error'] is None


def test_does_not_fuzzy_match_a_different_professionals_surname(
    tmp_path: Path,
) -> None:
    # 'Richard' bate com o Dr. Richard Roe, mas 'Doe' nao -- nao deve
    # confundir com nenhum profissional cadastrado.
    intent = Intent(
        intent='schedule',
        professional_name='Richard Doe',
        patient_name='Maria Santos',
        date='2026-09-10',
        time='14:00',
    )
    state, service, store = _build_state(intent, tmp_path)
    scheduler = make_scheduler_node(service, store)

    result = asyncio.run(scheduler(state))

    assert result['error'] == 'No matching professional found.'


def test_does_not_match_when_only_a_title_is_given(tmp_path: Path) -> None:
    intent = Intent(
        intent='schedule',
        professional_name='Dra.',
        patient_name='Maria Santos',
        date='2026-09-10',
        time='14:00',
    )
    state, service, store = _build_state(intent, tmp_path)
    scheduler = make_scheduler_node(service, store)

    result = asyncio.run(scheduler(state))

    assert result['error'] == 'No matching professional found.'
