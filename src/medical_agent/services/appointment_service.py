from datetime import datetime, timedelta

from medical_agent.services.calendar_client import CalendarClient, Professional

APPOINTMENT_DURATION = timedelta(minutes=30)


class SlotUnavailableError(Exception):
    """O profissional já tem consulta marcada nesse horário."""


class AppointmentNotFoundError(Exception):
    """Não existe consulta com esses dados para cancelar."""


class AppointmentService:
    def __init__(self, calendar_client: CalendarClient) -> None:
        self._calendar_client = calendar_client

    def list_professionals(self) -> list[Professional]:
        return self._calendar_client.list_professionals()

    def book_appointment(
        self,
        calendar_id: str,
        start: datetime,
        patient_name: str,
        reason: str,
    ) -> str:
        end = start + APPOINTMENT_DURATION

        if not self._calendar_client.is_available(calendar_id, start, end):
            raise SlotUnavailableError(
                'This professional already has an appointment at that time'
            )

        return self._calendar_client.create_event(
            calendar_id, start, end, patient_name, reason
        )

    def cancel_appointment(
        self, calendar_id: str, patient_name: str, start: datetime
    ) -> None:
        try:
            self._calendar_client.cancel_event(
                calendar_id, patient_name, start
            )
        except LookupError as exc:
            raise AppointmentNotFoundError(str(exc)) from exc
