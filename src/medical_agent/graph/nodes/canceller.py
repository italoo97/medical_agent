from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any

from medical_agent.graph.nodes._matching import find_professional
from medical_agent.graph.state import AppointmentState
from medical_agent.services.appointment_service import (
    AppointmentNotFoundError,
    AppointmentService,
)
from medical_agent.services.calendar_client import Professional


def make_canceller_node(
    appointment_service: AppointmentService,
) -> Callable[[AppointmentState], Coroutine[Any, Any, dict[str, str | None]]]:
    async def canceller(state: AppointmentState) -> dict[str, str | None]:
        intent = state['intent']
        if intent is None or not intent.patient_name:
            return {
                'error': (
                    'Não consegui cancelar a consulta porque faltou '
                    'informar: seu nome.'
                )
            }

        professional = find_professional(
            state['professionals'], intent.professional_name, intent.specialty
        )

        start: datetime
        if intent.date and intent.time:
            # Paciente deu um horario exato -- casa exatamente com ele.
            start = datetime.fromisoformat(f'{intent.date}T{intent.time}')
            if professional is None:
                professional = (
                    appointment_service.find_professional_with_appointment(
                        state['professionals'], intent.patient_name, start
                    )
                )
            if professional is None:
                return {'error': 'No matching professional found.'}
        else:
            # Sem data/hora -- procura a proxima consulta futura desse
            # paciente. Se um profissional foi identificado, a busca fica
            # restrita a agenda dele; senao, varre todas.
            scope: list[Professional] = (
                [professional]
                if professional is not None
                else state['professionals']
            )
            found = appointment_service.find_patient_appointment(
                scope, intent.patient_name
            )
            if found is None:
                return {'error': 'No matching professional found.'}
            professional, start = found

        try:
            appointment_service.cancel_appointment(
                professional.calendar_id,
                intent.patient_name,
                start,
            )
        except AppointmentNotFoundError as error:
            return {
                'error': str(error),
                'calendar_id': professional.calendar_id,
            }

        return {
            'error': None,
            'calendar_id': professional.calendar_id,
            'appointment_datetime': start.isoformat(),
        }

    return canceller
