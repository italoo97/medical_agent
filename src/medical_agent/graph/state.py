from typing import TypedDict

from medical_agent.prompts.v1.identify_intent import Intent
from medical_agent.services.calendar_client import Professional


class AppointmentState(TypedDict):
    user_message: str
    intent: Intent | None
    professionals: list[Professional]
    calendar_id: str | None
    response: str
    error: str | None
