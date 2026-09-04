from dataclasses import dataclass
from datetime import datetime

from medical_agent.services.calendar_client import Professional


@dataclass
class _FakeEvent:
    calendar_id: str
    start: datetime
    end: datetime
    patient_name: str
    reason: str


class FakeCalendarClient:
    """CalendarClient em memória — só para os testes, sem chamada de rede."""

    def __init__(self) -> None:
        self._events: list[_FakeEvent] = []
        self._professionals = [
            Professional('fake-calendar-john', 'Dr. John Doe', 'Cardiology'),
            Professional('fake-calendar-jane', 'Dr. Jane Doe', 'Dermatology'),
            Professional(
                'fake-calendar-richard', 'Dr. Richard Roe', 'Neurology'
            ),
        ]

    def list_professionals(self) -> list[Professional]:
        return list(self._professionals)

    def is_available(
        self, calendar_id: str, start: datetime, end: datetime
    ) -> bool:
        return not any(
            event.calendar_id == calendar_id and event.start == start
            for event in self._events
        )

    def create_event(
        self,
        calendar_id: str,
        start: datetime,
        end: datetime,
        patient_name: str,
        reason: str,
    ) -> str:
        event = _FakeEvent(calendar_id, start, end, patient_name, reason)
        self._events.append(event)
        return f'fake-event-{len(self._events)}'

    def cancel_event(
        self, calendar_id: str, patient_name: str, start: datetime
    ) -> None:
        for event in self._events:
            if (
                event.calendar_id == calendar_id
                and event.patient_name == patient_name
                and event.start == start
            ):
                self._events.remove(event)
                return

        raise LookupError('No matching appointment found to cancel')
