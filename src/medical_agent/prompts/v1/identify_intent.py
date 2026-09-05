from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from medical_agent.prompts.v1._loader import load_sections
from medical_agent.services.calendar_client import Professional

_SECTIONS = load_sections('identify_intent.md')


class Intent(BaseModel):
    intent: Literal['schedule', 'cancel', 'check', 'unknown']
    professional_name: str | None = None
    specialty: str | None = None
    patient_name: str | None = None
    date: str | None = None
    time: str | None = None
    reason: str | None = None


def build_system_prompt() -> str:
    return _SECTIONS['system_prompt']


def _format_roster(professionals: list[Professional]) -> str:
    if not professionals:
        return '(nenhum profissional cadastrado no momento)'
    return '\n'.join(
        f'- {professional.name} ({professional.specialty})'
        for professional in professionals
    )


def build_user_prompt(
    user_message: str, professionals: list[Professional]
) -> str:
    today = datetime.now().strftime('%Y-%m-%d (%A)')
    return _SECTIONS['user_prompt'].format(
        today=today,
        professionals=_format_roster(professionals),
        user_message=user_message,
    )
