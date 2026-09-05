from medical_agent.prompts.v1.identify_intent import build_user_prompt
from medical_agent.services.calendar_client import Professional


def test_user_prompt_includes_the_professionals_roster() -> None:
    professionals = [
        Professional('cal-john', 'John Doe', 'Cardiology'),
        Professional('cal-jane', 'Jane Doe', 'Dermatology'),
    ]

    prompt = build_user_prompt('quero marcar uma consulta', professionals)

    assert '- John Doe (Cardiology)' in prompt
    assert '- Jane Doe (Dermatology)' in prompt


def test_user_prompt_handles_empty_roster() -> None:
    prompt = build_user_prompt('quero marcar uma consulta', [])

    assert '(nenhum profissional cadastrado no momento)' in prompt
