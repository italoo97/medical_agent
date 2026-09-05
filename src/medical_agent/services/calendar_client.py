from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class Professional:
    calendar_id: str
    name: str
    specialty: str


class CalendarClient(Protocol):
    def list_professionals(self) -> list[Professional]:
        """Lista os profissionais com calendário compartilhado."""
        ...

    def is_available(
        self, calendar_id: str, start: datetime, end: datetime
    ) -> bool: ...

    def create_event(
        self,
        calendar_id: str,
        start: datetime,
        end: datetime,
        patient_name: str,
        reason: str,
    ) -> str: ...

    def cancel_event(
        self, calendar_id: str, patient_name: str, start: datetime
    ) -> None: ...

    def has_appointment(
        self, calendar_id: str, patient_name: str, start: datetime
    ) -> bool:
        """Existe uma consulta desse paciente nesse horario, nessa agenda?"""
        ...

    def find_upcoming_appointment(
        self, calendar_id: str, patient_name: str
    ) -> datetime | None:
        """Proxima consulta futura desse paciente nessa agenda, se houver."""
        ...

    def register_calendar(self, calendar_id: str) -> bool:
        """Assina a agenda na calendar list. True se registrou agora."""
        ...

    def unregister_calendar(self, calendar_id: str) -> bool:
        """Remove a agenda da calendar list. True se removeu agora."""
        ...
