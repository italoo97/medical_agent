from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any

from medical_agent.graph.nodes._matching import find_professional
from medical_agent.graph.state import AppointmentState
from medical_agent.services.appointment_service import (
    AppointmentService,
    SlotUnavailableError,
)


def make_scheduler_node(
    appointment_service: AppointmentService,
) -> Callable[[AppointmentState], Coroutine[Any, Any, dict[str, str | None]]]:
    async def scheduler(state: AppointmentState) -> dict[str, str | None]:
        intent = state['intent']
        if intent is None or not intent.date or not intent.time:
            return {'error': 'Missing date or time to schedule.'}

        professional = find_professional(
            state['professionals'], intent.professional_name, intent.specialty
        )
        if professional is None:
            return {'error': 'No matching professional found.'}

        start = datetime.fromisoformat(f'{intent.date}T{intent.time}')

        try:
            appointment_service.book_appointment(
                professional.calendar_id,
                start,
                intent.patient_name or 'Patient',
                intent.reason or 'General consultation',
            )
        except SlotUnavailableError as error:
            return {
                'error': str(error),
                'calendar_id': professional.calendar_id,
            }

        return {'error': None, 'calendar_id': professional.calendar_id}

    return scheduler
