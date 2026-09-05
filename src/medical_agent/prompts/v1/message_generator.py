from medical_agent.prompts.v1._loader import load_sections
from medical_agent.prompts.v1.identify_intent import Intent

_SECTIONS = load_sections('message_generator.md')


def build_system_prompt() -> str:
    return _SECTIONS['system_prompt']


def build_user_prompt(
    original_message: str,
    intent: Intent | None,
    professional_name: str | None,
    error: str | None,
    appointment_datetime: str | None = None,
) -> str:
    lines = [
        _SECTIONS['header_original_message'].format(
            original_message=original_message
        ),
        '',
        _SECTIONS['header_outcome'],
    ]

    if intent is None:
        lines.append(_SECTIONS['line_unknown'])
        return '\n'.join(lines)

    lines.append(_SECTIONS['line_action'].format(action=intent.intent))
    lines.append(
        _SECTIONS['line_professional'].format(
            professional=professional_name or 'not resolved'
        )
    )
    lines.append(
        _SECTIONS['line_date'].format(date=intent.date or 'not provided')
    )
    lines.append(
        _SECTIONS['line_time'].format(time=intent.time or 'not provided')
    )

    if appointment_datetime is not None:
        found_date, found_time = appointment_datetime.split('T')
        lines.append(
            _SECTIONS['line_found_appointment'].format(
                date=found_date, time=found_time[:5]
            )
        )

    if error is not None:
        lines.append(_SECTIONS['line_result_failed'].format(reason=error))
    else:
        lines.append(_SECTIONS['line_result_success'])

    return '\n'.join(lines)
