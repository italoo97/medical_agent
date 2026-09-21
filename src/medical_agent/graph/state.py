from typing import TypedDict

from medical_agent.prompts.v1.identify_intent import Intent
from medical_agent.services.calendar_client import Professional


class AppointmentState(TypedDict):
    session_id: str
    user_message: str
    intent: Intent | None
    professionals: list[Professional]
    calendar_id: str | None
    appointment_datetime: str | None
    response: str
    error: str | None
