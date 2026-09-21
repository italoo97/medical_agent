import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from medical_agent.prompts.v1.identify_intent import Intent
from medical_agent.services.conversation_state import (
    ConversationStateStore,
    merge_intent,
)


def _store(tmp_path: Path) -> ConversationStateStore:
    return ConversationStateStore(str(tmp_path / 'state.db'))


def _touch_updated_at(tmp_path: Path, session_id: str, when: datetime) -> None:
    """Reescreve updated_at direto no banco, simulando o passar do tempo."""
    with sqlite3.connect(str(tmp_path / 'state.db')) as conn:
        conn.execute(
            'UPDATE conversation_state SET updated_at = ? '
            'WHERE session_id = ?',
            (when.isoformat(), session_id),
        )


def _touch_booking_updated_at(
    tmp_path: Path, session_id: str, when: datetime
) -> None:
    with sqlite3.connect(str(tmp_path / 'state.db')) as conn:
        conn.execute(
            'UPDATE session_bookings SET updated_at = ? '
            'WHERE session_id = ?',
            (when.isoformat(), session_id),
        )


def test_load_returns_none_when_nothing_saved(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.load('missing-session') is None


def test_save_then_load_roundtrips_the_intent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    intent = Intent(intent='schedule', patient_name='Maria Santos')

    store.save('session-1', intent)
    loaded = store.load('session-1')

    assert loaded == intent


def test_save_upserts_existing_session(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save('session-1', Intent(intent='schedule', patient_name='Maria'))
    store.save('session-1', Intent(intent='cancel', patient_name='Maria'))

    loaded = store.load('session-1')

    assert loaded is not None
    assert loaded.intent == 'cancel'


def test_load_expires_after_ttl_and_clears_the_row(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save('session-1', Intent(intent='schedule', patient_name='Maria'))
    _touch_updated_at(
        tmp_path, 'session-1', datetime.now() - timedelta(hours=25)
    )

    assert store.load('session-1') is None
    # A linha expirada foi apagada -- uma nova sessao com o mesmo id
    # nao deveria "herdar" nada dela.
    with sqlite3.connect(str(tmp_path / 'state.db')) as conn:
        row = conn.execute(
            'SELECT 1 FROM conversation_state WHERE session_id = ?',
            ('session-1',),
        ).fetchone()
    assert row is None


def test_load_within_ttl_still_returns_the_intent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save('session-1', Intent(intent='schedule', patient_name='Maria'))
    _touch_updated_at(
        tmp_path, 'session-1', datetime.now() - timedelta(hours=23)
    )

    assert store.load('session-1') is not None


def test_load_by_patient_name_returns_none_when_no_match(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    assert store.load_by_patient_name('Ninguem Marcou') is None


def test_load_by_patient_name_finds_state_from_another_session(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    intent = Intent(intent='cancel', patient_name='Maria Santos')
    store.save('old-session', intent)

    found = store.load_by_patient_name('Maria Santos')

    assert found == intent


def test_load_by_patient_name_picks_the_most_recent_session(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.save(
        'older-session',
        Intent(
            intent='schedule', patient_name='Maria Santos', date='2026-09-01'
        ),
    )
    store.save(
        'newer-session',
        Intent(
            intent='cancel', patient_name='Maria Santos', date='2026-09-02'
        ),
    )

    found = store.load_by_patient_name('Maria Santos')

    assert found is not None
    assert found.date == '2026-09-02'


def test_load_by_patient_name_respects_ttl(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save('old-session', Intent(intent='cancel', patient_name='Maria'))
    _touch_updated_at(
        tmp_path, 'old-session', datetime.now() - timedelta(hours=25)
    )

    assert store.load_by_patient_name('Maria') is None


def test_clear_removes_the_row(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save('session-1', Intent(intent='schedule', patient_name='Maria'))

    store.clear('session-1')

    assert store.load('session-1') is None


def test_clear_is_a_no_op_for_unknown_session(tmp_path: Path) -> None:
    store = _store(tmp_path)

    store.clear('never-existed')  # nao deve levantar excecao


def test_merge_returns_new_when_nothing_was_saved() -> None:
    new = Intent(intent='schedule', patient_name='Maria')

    merged = merge_intent(None, new)

    assert merged == new


def test_merge_fills_missing_fields_from_saved_state() -> None:
    saved = Intent(
        intent='schedule',
        professional_name='John Doe',
        specialty='Cardiology',
        patient_name='Maria Santos',
        date='2026-09-10',
        time=None,
        reason='check-up',
    )
    new = Intent(intent='schedule', time='16:00')

    merged = merge_intent(saved, new)

    assert merged.professional_name == 'John Doe'
    assert merged.specialty == 'Cardiology'
    assert merged.patient_name == 'Maria Santos'
    assert merged.date == '2026-09-10'
    assert merged.time == '16:00'
    assert merged.reason == 'check-up'


def test_merge_lets_the_current_message_override_saved_fields() -> None:
    saved = Intent(intent='schedule', professional_name='John Doe')
    new = Intent(intent='schedule', professional_name='Jane Doe')

    merged = merge_intent(saved, new)

    assert merged.professional_name == 'Jane Doe'


def test_merge_keeps_the_ongoing_intent_when_new_message_is_unknown() -> None:
    saved = Intent(intent='cancel', patient_name='Maria')
    new = Intent(intent='unknown', patient_name='Maria')

    merged = merge_intent(saved, new)

    assert merged.intent == 'cancel'


def test_merge_lets_a_clear_new_intent_override_the_saved_one() -> None:
    saved = Intent(intent='schedule', patient_name='Maria')
    new = Intent(intent='cancel', patient_name='Maria')

    merged = merge_intent(saved, new)

    assert merged.intent == 'cancel'


def test_merge_keeps_the_saved_language_over_a_new_guess() -> None:
    saved = Intent(
        intent='schedule', patient_name='Italo', language='Portuguese'
    )
    new = Intent(intent='unknown', patient_name='Italo', language='Italian')

    merged = merge_intent(saved, new)

    assert merged.language == 'Portuguese'


def test_merge_uses_the_new_language_when_nothing_was_saved_yet() -> None:
    saved = Intent(intent='schedule', patient_name='Italo', language=None)
    new = Intent(intent='schedule', patient_name='Italo', language='English')

    merged = merge_intent(saved, new)

    assert merged.language == 'English'


def test_load_last_booking_returns_none_when_nothing_saved(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    assert store.load_last_booking('missing-session') is None


def test_save_and_load_last_booking_roundtrips(tmp_path: Path) -> None:
    store = _store(tmp_path)
    start = datetime(2026, 9, 10, 16, 0)

    store.save_last_booking('session-1', 'cal-john', start, 'Italo Santos')
    booking = store.load_last_booking('session-1')

    assert booking == ('cal-john', start, 'Italo Santos')


def test_save_last_booking_upserts_existing_session(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_last_booking(
        'session-1', 'cal-john', datetime(2026, 9, 10, 16, 0), 'Italo'
    )
    store.save_last_booking(
        'session-1', 'cal-jane', datetime(2026, 9, 11, 10, 0), 'Italo'
    )

    booking = store.load_last_booking('session-1')

    assert booking is not None
    assert booking[0] == 'cal-jane'


def test_load_last_booking_expires_after_ttl(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_last_booking(
        'session-1', 'cal-john', datetime(2026, 9, 10, 16, 0), 'Italo'
    )
    _touch_booking_updated_at(
        tmp_path, 'session-1', datetime.now() - timedelta(hours=25)
    )

    assert store.load_last_booking('session-1') is None
    with sqlite3.connect(str(tmp_path / 'state.db')) as conn:
        row = conn.execute(
            'SELECT 1 FROM session_bookings WHERE session_id = ?',
            ('session-1',),
        ).fetchone()
    assert row is None


def test_load_last_booking_within_ttl_still_returns_it(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.save_last_booking(
        'session-1', 'cal-john', datetime(2026, 9, 10, 16, 0), 'Italo'
    )
    _touch_booking_updated_at(
        tmp_path, 'session-1', datetime.now() - timedelta(hours=23)
    )

    assert store.load_last_booking('session-1') is not None


def test_clear_last_booking_removes_the_row(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_last_booking(
        'session-1', 'cal-john', datetime(2026, 9, 10, 16, 0), 'Italo'
    )

    store.clear_last_booking('session-1')

    assert store.load_last_booking('session-1') is None


def test_clear_last_booking_is_a_no_op_for_unknown_session(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    store.clear_last_booking('never-existed')
