from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class Intent(BaseModel):
    intent: Literal['schedule', 'cancel', 'unknown']
    professional_name: str | None = None
    specialty: str | None = None
    patient_name: str | None = None
    date: str | None = None
    time: str | None = None
    reason: str | None = None


def build_system_prompt() -> str:
    return (
        'You extract structured data from patient messages sent to a '
        'medical appointment scheduling assistant. You never book, '
        'cancel or confirm anything yourself — you only translate the '
        'message into JSON matching the given schema.\n\n'
        'Rules:\n'
        '- intent must be "schedule", "cancel" or "unknown".\n'
        '- date must be in "YYYY-MM-DD" format if present, else null.\n'
        '- time must be in "HH:MM" 24h format if present, else null.\n'
        '- If a field is not mentioned, return null for it.\n'
        '- Respond with a single JSON object matching the schema, '
        'nothing else.'
    )


def build_user_prompt(user_message: str) -> str:
    today = datetime.now().strftime('%Y-%m-%d (%A)')
    return f'Today is {today}.\n' f'Patient message: "{user_message}"'
