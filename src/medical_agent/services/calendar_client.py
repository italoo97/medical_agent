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
