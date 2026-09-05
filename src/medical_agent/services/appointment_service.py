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

    def register_professional(self, calendar_id: str) -> bool:
        return self._calendar_client.register_calendar(calendar_id)

    def remove_professional(self, calendar_id: str) -> bool:
        return self._calendar_client.unregister_calendar(calendar_id)

    def find_professional_with_appointment(
        self,
        professionals: list[Professional],
        patient_name: str,
        start: datetime,
    ) -> Professional | None:
        """Procura, em todas as agendas, quem tem consulta desse paciente.

        Usado quando o paciente pede para cancelar sem dizer (ou acertar)
        o nome do profissional -- em vez de falhar, procuramos em cada
        agenda por uma consulta marcada exatamente para esse patient_name
        e horario.
        """
        for professional in professionals:
            if self._calendar_client.has_appointment(
                professional.calendar_id, patient_name, start
            ):
                return professional
        return None

    def find_patient_appointment(
        self, professionals: list[Professional], patient_name: str
    ) -> tuple[Professional, datetime] | None:
        """Procura a proxima consulta futura desse paciente, em qualquer
        agenda.

        Essa e a mesma "tool" reaproveitada por qualquer fluxo que precise
        saber se/quando um paciente tem consulta marcada -- hoje pelo
        intent "check" (o paciente perguntando diretamente), e no futuro
        por um cancelamento sem data/hora informados, por exemplo.
        """
        for professional in professionals:
            start = self._calendar_client.find_upcoming_appointment(
                professional.calendar_id, patient_name
            )
            if start is not None:
                return professional, start
        return None

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
