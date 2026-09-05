from datetime import datetime

import pytest
from medical_agent.services.appointment_service import (
    AppointmentNotFoundError,
    AppointmentService,
    SlotUnavailableError,
)
from medical_agent.services.fake_calendar_client import FakeCalendarClient


def _build_service() -> tuple[AppointmentService, str]:
    service = AppointmentService(FakeCalendarClient())
    calendar_id = service.list_professionals()[0].calendar_id
    return service, calendar_id


def test_books_appointment_successfully() -> None:
    service, calendar_id = _build_service()

    event_id = service.book_appointment(
        calendar_id, datetime(2026, 9, 10, 14, 0), 'Maria Santos', 'check-up'
    )

    assert event_id


def test_raises_when_slot_is_already_booked() -> None:
    service, calendar_id = _build_service()
    start = datetime(2026, 9, 10, 14, 0)

    service.book_appointment(calendar_id, start, 'Maria Santos', 'check-up')

    with pytest.raises(SlotUnavailableError):
        service.book_appointment(
            calendar_id, start, 'Joao da Silva', 'follow-up'
        )


def test_cancels_appointment_successfully() -> None:
    service, calendar_id = _build_service()
    start = datetime(2026, 9, 10, 14, 0)

    service.book_appointment(calendar_id, start, 'Maria Santos', 'check-up')
    service.cancel_appointment(calendar_id, 'Maria Santos', start)


def test_raises_when_cancelling_nonexistent_appointment() -> None:
    service, calendar_id = _build_service()

    with pytest.raises(AppointmentNotFoundError):
        service.cancel_appointment(
            calendar_id, 'Ninguem', datetime(2026, 9, 10, 14, 0)
        )


def test_register_professional_returns_true_when_new() -> None:
    service, calendar_id = _build_service()

    assert service.register_professional(calendar_id) is True


def test_register_professional_returns_false_when_already_registered() -> None:
    service, calendar_id = _build_service()

    service.register_professional(calendar_id)

    assert service.register_professional(calendar_id) is False


def test_remove_professional_returns_true_when_registered() -> None:
    service, calendar_id = _build_service()
    service.register_professional(calendar_id)

    assert service.remove_professional(calendar_id) is True


def test_remove_professional_returns_false_when_not_registered() -> None:
    service, calendar_id = _build_service()

    assert service.remove_professional(calendar_id) is False


def test_finds_professional_with_matching_appointment() -> None:
    service = AppointmentService(FakeCalendarClient())
    professionals = service.list_professionals()
    target = professionals[1]
    start = datetime(2026, 9, 10, 14, 0)
    service.book_appointment(
        target.calendar_id, start, 'Maria Santos', 'check-up'
    )

    found = service.find_professional_with_appointment(
        professionals, 'Maria Santos', start
    )

    assert found == target


def test_find_professional_with_appointment_returns_none_when_no_match() -> (
    None
):
    service = AppointmentService(FakeCalendarClient())
    professionals = service.list_professionals()

    found = service.find_professional_with_appointment(
        professionals, 'Ninguem', datetime(2026, 9, 10, 14, 0)
    )

    assert found is None
