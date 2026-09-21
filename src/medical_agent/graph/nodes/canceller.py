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
from medical_agent.services.conversation_state import ConversationStateStore


def _find_session_booking(
    conversation_state_store: ConversationStateStore,
    state: AppointmentState,
    named_professional: Professional | None,
) -> tuple[Professional, datetime, str] | None:
    """So usa a memoria da sessao quando ela nao contradiz o pedido atual.

    Se o paciente citou um profissional/especialidade NESTA mensagem e
    isso resolveu para alguem diferente do que essa sessao agendou,
    respeitamos o que foi dito agora em vez de presumir que e a mesma
    consulta.
    """
    booking = conversation_state_store.load_last_booking(state['session_id'])
    if booking is None:
        return None

    calendar_id, start, patient_name = booking
    if (
        named_professional is not None
        and named_professional.calendar_id != calendar_id
    ):
        return None

    professional = next(
        (
            candidate
            for candidate in state['professionals']
            if candidate.calendar_id == calendar_id
        ),
        None,
    )
    if professional is None:
        return None

    return professional, start, patient_name


def _cancel_and_report(
    appointment_service: AppointmentService,
    conversation_state_store: ConversationStateStore,
    session_id: str,
    professional: Professional,
    patient_name: str,
    start: datetime,
) -> dict[str, str | None]:
    try:
        appointment_service.cancel_appointment(
            professional.calendar_id, patient_name, start
        )
    except AppointmentNotFoundError as error:
        return {
            'error': str(error),
            'calendar_id': professional.calendar_id,
        }

    conversation_state_store.clear_last_booking(session_id)

    return {
        'error': None,
        'calendar_id': professional.calendar_id,
        'appointment_datetime': start.isoformat(),
    }


def make_canceller_node(
    appointment_service: AppointmentService,
    conversation_state_store: ConversationStateStore,
) -> Callable[[AppointmentState], Coroutine[Any, Any, dict[str, str | None]]]:
    async def canceller(state: AppointmentState) -> dict[str, str | None]:
        intent = state['intent']

        named_professional = find_professional(
            state['professionals'],
            intent.professional_name if intent is not None else None,
            intent.specialty if intent is not None else None,
        )

        session_booking = _find_session_booking(
            conversation_state_store, state, named_professional
        )
        if session_booking is not None:
            (
                session_professional,
                session_start,
                session_patient_name,
            ) = session_booking
            return _cancel_and_report(
                appointment_service,
                conversation_state_store,
                state['session_id'],
                session_professional,
                session_patient_name,
                session_start,
            )

        if intent is None or not intent.patient_name:
            return {
                'error': (
                    'Não consegui cancelar a consulta porque faltou '
                    'informar: seu nome.'
                )
            }

        professional: Professional | None = named_professional
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

        return _cancel_and_report(
            appointment_service,
            conversation_state_store,
            state['session_id'],
            professional,
            intent.patient_name,
            start,
        )

    return canceller
