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
