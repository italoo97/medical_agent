from medical_agent.prompts.v1.identify_intent import Intent
from medical_agent.prompts.v1.message_generator import (
    build_system_prompt,
    build_user_prompt,
)


def test_system_prompt_is_not_empty() -> None:
    assert build_system_prompt()


def test_user_prompt_handles_unknown_intent() -> None:
    prompt = build_user_prompt(
        original_message='oi',
        intent=None,
        professional_name=None,
        error=None,
    )

    assert 'oi' in prompt


def test_user_prompt_reports_success_without_found_appointment_line() -> None:
    intent = Intent(intent='schedule', date='2026-09-10', time='16:00')

    prompt = build_user_prompt(
        original_message='quero marcar',
        intent=intent,
        professional_name='John Doe',
        error=None,
    )

    assert 'John Doe' in prompt
    assert '2026-09-10' in prompt
    assert '16:00' in prompt


def test_user_prompt_reports_failure_with_reason() -> None:
    intent = Intent(intent='schedule')

    prompt = build_user_prompt(
        original_message='quero marcar',
        intent=intent,
        professional_name=None,
        error='faltou o nome',
    )

    assert 'faltou o nome' in prompt
    assert 'not resolved' in prompt
    assert 'not provided' in prompt


def test_user_prompt_defaults_to_portuguese_when_no_language_given() -> None:
    prompt = build_user_prompt(
        original_message='oi',
        intent=None,
        professional_name=None,
        error=None,
    )

    assert 'Portuguese' in prompt


def test_user_prompt_uses_the_given_language_instead_of_the_default() -> None:
    intent = Intent(intent='schedule', date='2026-09-10', time='16:00')

    prompt = build_user_prompt(
        original_message='I want to book an appointment',
        intent=intent,
        professional_name='John Doe',
        error=None,
        language='English',
    )

    assert 'Reply language: English' in prompt
    assert 'Portuguese' not in prompt


def test_user_prompt_includes_found_appointment_line_when_present() -> None:
    intent = Intent(intent='check', patient_name='Maria')

    prompt = build_user_prompt(
        original_message='tenho consulta marcada?',
        intent=intent,
        professional_name='Jane Doe',
        error=None,
        appointment_datetime='2026-09-10T16:00:00-03:00',
    )

    assert '2026-09-10' in prompt
    assert '16:00' in prompt
