from pathlib import Path

from medical_agent.prompts.v1.identify_intent import Intent
from medical_agent.services.conversation_state import (
    ConversationStateStore,
)


def test_capitalizes_the_first_letter_of_each_name() -> None:
    intent = Intent(intent='schedule', patient_name='italo santos')

    assert intent.patient_name == 'Italo Santos'


def test_normalizes_a_fully_uppercase_name() -> None:
    intent = Intent(intent='schedule', patient_name='ITALO SANTOS')

    assert intent.patient_name == 'Italo Santos'


def test_collapses_extra_whitespace_between_names() -> None:
    intent = Intent(intent='schedule', patient_name='  italo   santos  ')

    assert intent.patient_name == 'Italo Santos'


def test_keeps_none_as_none() -> None:
    intent = Intent(intent='schedule', patient_name=None)

    assert intent.patient_name is None


def test_blank_name_becomes_none() -> None:
    intent = Intent(intent='schedule', patient_name='   ')

    assert intent.patient_name is None


def test_normalization_survives_the_conversation_state_roundtrip(
    tmp_path: Path,
) -> None:
    store = ConversationStateStore(str(tmp_path / 'state.db'))
    intent = Intent(intent='schedule', patient_name='ITALO santos')

    store.save('session-1', intent)
    loaded = store.load('session-1')

    assert loaded is not None
    assert loaded.patient_name == 'Italo Santos'
