from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any

from medical_agent.graph.nodes._matching import find_professional
from medical_agent.graph.state import AppointmentState
from medical_agent.services.appointment_service import (
    AppointmentNotFoundError,
    AppointmentService,
)


def make_canceller_node(
    appointment_service: AppointmentService,
) -> Callable[[AppointmentState], Coroutine[Any, Any, dict[str, str | None]]]:
    async def canceller(state: AppointmentState) -> dict[str, str | None]:
        intent = state['intent']
        if intent is None or not intent.date or not intent.time:
            return {'error': 'Missing date or time to cancel.'}

        professional = find_professional(
            state['professionals'], intent.professional_name, intent.specialty
        )
        if professional is None:
            return {'error': 'No matching professional found.'}

        start = datetime.fromisoformat(f'{intent.date}T{intent.time}')

        try:
            appointment_service.cancel_appointment(
                professional.calendar_id,
                intent.patient_name or 'Patient',
                start,
            )
        except AppointmentNotFoundError as error:
            return {
                'error': str(error),
                'calendar_id': professional.calendar_id,
            }

        return {'error': None, 'calendar_id': professional.calendar_id}

    return canceller
