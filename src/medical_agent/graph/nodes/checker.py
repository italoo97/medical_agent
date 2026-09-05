from collections.abc import Callable, Coroutine
from typing import Any

from medical_agent.graph.state import AppointmentState
from medical_agent.services.appointment_service import AppointmentService


def make_checker_node(
    appointment_service: AppointmentService,
) -> Callable[[AppointmentState], Coroutine[Any, Any, dict[str, str | None]]]:
    async def checker(state: AppointmentState) -> dict[str, str | None]:
        intent = state['intent']
        if intent is None or not intent.patient_name:
            return {
                'error': (
                    'Não consegui verificar sua consulta porque faltou '
                    'informar: seu nome.'
                )
            }

        found = appointment_service.find_patient_appointment(
            state['professionals'], intent.patient_name
        )
        if found is None:
            return {'error': 'No upcoming appointment found for this patient.'}

        professional, start = found
        return {
            'error': None,
            'calendar_id': professional.calendar_id,
            'appointment_datetime': start.isoformat(),
        }

    return checker
