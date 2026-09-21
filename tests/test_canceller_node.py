import asyncio
from datetime import datetime
from pathlib import Path

from medical_agent.graph.nodes.canceller import make_canceller_node
from medical_agent.graph.state import AppointmentState
from medical_agent.prompts.v1.identify_intent import Intent
from medical_agent.services.appointment_service import AppointmentService
from medical_agent.services.conversation_state import ConversationStateStore
from medical_agent.services.fake_calendar_client import FakeCalendarClient


def _calendar_id_for(service: AppointmentService, name: str) -> str:
    for professional in service.list_professionals():
        if name in professional.name:
            return professional.calendar_id
    raise AssertionError(f'no fixture professional matches "{name}"')


def _build_state(
    service: AppointmentService,
    intent: Intent | None,
    session_id: str = 'session-1',
) -> AppointmentState:
    return {
        'session_id': session_id,
        'user_message': '',
        'intent': intent,
        'professionals': service.list_professionals(),
        'calendar_id': None,
        'appointment_datetime': None,
        'response': '',
        'error': None,
    }


def test_cancels_existing_appointment(tmp_path: Path) -> None:
    service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
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
    canceller = make_canceller_node(service, store)

    result = asyncio.run(canceller(state))

    assert result['error'] is None
    assert result['calendar_id'] == calendar_id


def test_returns_error_when_patient_name_is_missing(tmp_path: Path) -> None:
    service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    intent = Intent(
        intent='cancel',
        professional_name='Jane Doe',
        date='2026-09-10',
        time='15:00',
    )
    state = _build_state(service, intent)
    canceller = make_canceller_node(service, store)

    result = asyncio.run(canceller(state))

    assert result['error'] == (
        'Não consegui cancelar a consulta porque faltou informar: seu nome.'
    )


def test_returns_error_when_professional_not_found(tmp_path: Path) -> None:
    service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    intent = Intent(
        intent='cancel',
        professional_name='Nonexistent Doctor',
        patient_name='Maria Santos',
        date='2026-09-10',
        time='15:00',
    )
    state = _build_state(service, intent)
    canceller = make_canceller_node(service, store)

    result = asyncio.run(canceller(state))

    assert result['error'] == 'No matching professional found.'


def test_returns_error_when_appointment_not_found(tmp_path: Path) -> None:
    service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    intent = Intent(
        intent='cancel',
        professional_name='Jane Doe',
        patient_name='Maria Santos',
        date='2026-09-10',
        time='15:00',
    )
    state = _build_state(service, intent)
    canceller = make_canceller_node(service, store)

    result = asyncio.run(canceller(state))

    assert result['error'] is not None
    assert result['error'] != 'No matching professional found.'


def test_finds_professional_by_appointment_when_not_named(
    tmp_path: Path,
) -> None:
    service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    calendar_id = _calendar_id_for(service, 'Jane Doe')
    service.book_appointment(
        calendar_id,
        datetime(2026, 9, 10, 15, 0),
        'Maria Santos',
        'check-up',
    )
    intent = Intent(
        intent='cancel',
        patient_name='Maria Santos',
        date='2026-09-10',
        time='15:00',
    )
    state = _build_state(service, intent)
    canceller = make_canceller_node(service, store)

    result = asyncio.run(canceller(state))

    assert result['error'] is None
    assert result['calendar_id'] == calendar_id


def test_returns_error_when_no_professional_has_matching_appointment(
    tmp_path: Path,
) -> None:
    service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    intent = Intent(
        intent='cancel',
        patient_name='Ninguem Marcou',
        date='2026-09-10',
        time='15:00',
    )
    state = _build_state(service, intent)
    canceller = make_canceller_node(service, store)

    result = asyncio.run(canceller(state))

    assert result['error'] == 'No matching professional found.'


def test_cancels_upcoming_appointment_when_date_and_time_not_given(
    tmp_path: Path,
) -> None:
    service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    calendar_id = _calendar_id_for(service, 'Jane Doe')
    service.book_appointment(
        calendar_id,
        datetime(2026, 9, 10, 15, 0),
        'Maria Santos',
        'check-up',
    )
    intent = Intent(intent='cancel', patient_name='Maria Santos')
    state = _build_state(service, intent)
    canceller = make_canceller_node(service, store)

    result = asyncio.run(canceller(state))

    assert result['error'] is None
    assert result['calendar_id'] == calendar_id
    assert (
        result['appointment_datetime']
        == datetime(2026, 9, 10, 15, 0).isoformat()
    )


def test_cancels_upcoming_appointment_scoped_to_named_professional(
    tmp_path: Path,
) -> None:
    service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    jane_calendar_id = _calendar_id_for(service, 'Jane Doe')
    service.book_appointment(
        jane_calendar_id,
        datetime(2026, 9, 10, 15, 0),
        'Maria Santos',
        'check-up',
    )
    intent = Intent(
        intent='cancel',
        professional_name='Jane Doe',
        patient_name='Maria Santos',
    )
    state = _build_state(service, intent)
    canceller = make_canceller_node(service, store)

    result = asyncio.run(canceller(state))

    assert result['error'] is None
    assert result['calendar_id'] == jane_calendar_id


def test_does_not_leak_appointment_from_a_different_professional(
    tmp_path: Path,
) -> None:
    service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    john_calendar_id = _calendar_id_for(service, 'John Doe')
    service.book_appointment(
        john_calendar_id,
        datetime(2026, 9, 10, 15, 0),
        'Maria Santos',
        'check-up',
    )
    # Maria so tem consulta com o Dr. John, mas pede para cancelar citando
    # a Dra. Jane -- a busca deve ficar restrita a agenda da Jane, e nao
    # encontrar (nem cancelar) a consulta que esta com o John.
    intent = Intent(
        intent='cancel',
        professional_name='Jane Doe',
        patient_name='Maria Santos',
    )
    state = _build_state(service, intent)
    canceller = make_canceller_node(service, store)

    result = asyncio.run(canceller(state))

    assert result['error'] == 'No matching professional found.'


def test_returns_error_when_no_upcoming_appointment_exists_anywhere(
    tmp_path: Path,
) -> None:
    service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    intent = Intent(intent='cancel', patient_name='Ninguem Marcou')
    state = _build_state(service, intent)
    canceller = make_canceller_node(service, store)

    result = asyncio.run(canceller(state))

    assert result['error'] == 'No matching professional found.'


def test_uses_the_sessions_last_booking_when_nothing_else_is_given(
    tmp_path: Path,
) -> None:
    service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    calendar_id = _calendar_id_for(service, 'John Doe')
    start = datetime(2026, 9, 10, 15, 0)
    service.book_appointment(calendar_id, start, 'Italo', 'check-up')
    store.save_last_booking('session-1', calendar_id, start, 'Italo')
    # A mensagem nao diz absolutamente nada -- nem nome, nem profissional.
    intent = Intent(intent='cancel')
    state = _build_state(service, intent)
    canceller = make_canceller_node(service, store)

    result = asyncio.run(canceller(state))

    assert result['error'] is None
    assert result['calendar_id'] == calendar_id
    assert result['appointment_datetime'] == start.isoformat()


def test_session_booking_works_even_when_intent_is_none(
    tmp_path: Path,
) -> None:
    service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    calendar_id = _calendar_id_for(service, 'John Doe')
    start = datetime(2026, 9, 10, 15, 0)
    service.book_appointment(calendar_id, start, 'Italo', 'check-up')
    store.save_last_booking('session-1', calendar_id, start, 'Italo')
    state = _build_state(service, None)
    canceller = make_canceller_node(service, store)

    result = asyncio.run(canceller(state))

    assert result['error'] is None
    assert result['calendar_id'] == calendar_id


def test_session_booking_is_cleared_after_a_successful_cancel(
    tmp_path: Path,
) -> None:
    service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    calendar_id = _calendar_id_for(service, 'John Doe')
    start = datetime(2026, 9, 10, 15, 0)
    service.book_appointment(calendar_id, start, 'Italo', 'check-up')
    store.save_last_booking('session-1', calendar_id, start, 'Italo')
    intent = Intent(intent='cancel')
    state = _build_state(service, intent)
    canceller = make_canceller_node(service, store)

    asyncio.run(canceller(state))

    assert store.load_last_booking('session-1') is None


def test_session_booking_is_ignored_when_a_different_professional_named(
    tmp_path: Path,
) -> None:
    service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    john_calendar_id = _calendar_id_for(service, 'John Doe')
    start = datetime(2026, 9, 10, 15, 0)
    service.book_appointment(john_calendar_id, start, 'Italo', 'check-up')
    store.save_last_booking('session-1', john_calendar_id, start, 'Italo')
    # Explicitamente pede para cancelar com a Jane -- diferente do que a
    # sessao tem guardado -- entao a memoria da sessao deve ser ignorada.
    intent = Intent(intent='cancel', professional_name='Jane Doe')
    state = _build_state(service, intent)
    canceller = make_canceller_node(service, store)

    result = asyncio.run(canceller(state))

    assert result['error'] == (
        'Não consegui cancelar a consulta porque faltou informar: seu nome.'
    )


def test_session_booking_is_used_when_the_named_professional_matches(
    tmp_path: Path,
) -> None:
    service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    john_calendar_id = _calendar_id_for(service, 'John Doe')
    start = datetime(2026, 9, 10, 15, 0)
    service.book_appointment(john_calendar_id, start, 'Italo', 'check-up')
    store.save_last_booking('session-1', john_calendar_id, start, 'Italo')
    intent = Intent(intent='cancel', professional_name='John Doe')
    state = _build_state(service, intent)
    canceller = make_canceller_node(service, store)

    result = asyncio.run(canceller(state))

    assert result['error'] is None
    assert result['calendar_id'] == john_calendar_id


def test_falls_back_to_name_search_when_session_has_no_booking(
    tmp_path: Path,
) -> None:
    service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    calendar_id = _calendar_id_for(service, 'Jane Doe')
    service.book_appointment(
        calendar_id, datetime(2026, 9, 10, 15, 0), 'Maria Santos', 'check-up'
    )
    # Sessao "brand-new-session" nunca agendou nada -- deve cair no
    # fluxo antigo de busca por nome.
    intent = Intent(intent='cancel', patient_name='Maria Santos')
    state = _build_state(service, intent, session_id='brand-new-session')
    canceller = make_canceller_node(service, store)

    result = asyncio.run(canceller(state))

    assert result['error'] is None
    assert result['calendar_id'] == calendar_id


def test_ignores_session_booking_for_a_professional_no_longer_registered(
    tmp_path: Path,
) -> None:
    service = AppointmentService(FakeCalendarClient())
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    start = datetime(2026, 9, 10, 15, 0)
    # A sessao lembra de uma agenda que nao esta mais na lista de
    # profissionais atual (por exemplo, foi desregistrada depois).
    store.save_last_booking('session-1', 'calendar-removed', start, 'Italo')
    intent = Intent(intent='cancel', patient_name='Italo')
    state = _build_state(service, intent)
    canceller = make_canceller_node(service, store)

    result = asyncio.run(canceller(state))

    assert result['error'] == 'No matching professional found.'
