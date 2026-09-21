import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from medical_agent.prompts.v1.identify_intent import Intent

_TTL = timedelta(hours=24)


class ConversationStateStore:
    """Persiste o Intent parcial de cada conversa entre mensagens.

    Guarda apenas o estado estruturado (não uma lista de mensagens), o
    suficiente para completar campos que faltaram em turnos anteriores.
    Cada linha expira sozinha depois de 24h sem atividade.
    """

    def __init__(self, db_path: str = 'conversation_state.db') -> None:
        self._db_path = Path(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_state (
                    session_id TEXT PRIMARY KEY,
                    patient_name TEXT,
                    intent_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_state_patient
                ON conversation_state (patient_name)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_bookings (
                    session_id TEXT PRIMARY KEY,
                    calendar_id TEXT NOT NULL,
                    start_iso TEXT NOT NULL,
                    patient_name TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _row_to_intent(
        self, intent_json: str, updated_at: str, session_id: str
    ) -> Intent | None:
        if datetime.fromisoformat(updated_at) < datetime.now() - _TTL:
            self.clear(session_id)
            return None
        return Intent.model_validate_json(intent_json)

    def load(self, session_id: str) -> Intent | None:
        with self._connect() as conn:
            row = conn.execute(
                'SELECT intent_json, updated_at FROM conversation_state '
                'WHERE session_id = ?',
                (session_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_intent(row[0], row[1], session_id)

    def load_by_patient_name(self, patient_name: str) -> Intent | None:
        """Fallback quando o session_id não tem nada salvo.

        Pega o estado em aberto mais recente com esse patient_name --
        cobre o caso de o paciente ter mudado de sessão no meio da
        conversa.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, intent_json, updated_at
                FROM conversation_state
                WHERE patient_name = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (patient_name,),
            ).fetchone()

        if row is None:
            return None

        session_id, intent_json, updated_at = row
        return self._row_to_intent(intent_json, updated_at, session_id)

    def save(self, session_id: str, intent: Intent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_state
                    (session_id, patient_name, intent_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    patient_name = excluded.patient_name,
                    intent_json = excluded.intent_json,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    intent.patient_name,
                    intent.model_dump_json(),
                    datetime.now().isoformat(),
                ),
            )

    def clear(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                'DELETE FROM conversation_state WHERE session_id = ?',
                (session_id,),
            )

    def save_last_booking(
        self,
        session_id: str,
        calendar_id: str,
        start: datetime,
        patient_name: str,
    ) -> None:
        """Lembra qual consulta essa sessao acabou de confirmar.

        Diferente de save()/load() (que guardam um pedido AINDA
        incompleto), isso guarda um agendamento JA CONFIRMADO -- para
        que um cancelamento nessa mesma sessao, mesmo sem repetir nome
        ou profissional, saiba exatamente qual consulta cancelar.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_bookings
                    (session_id, calendar_id, start_iso, patient_name,
                     updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    calendar_id = excluded.calendar_id,
                    start_iso = excluded.start_iso,
                    patient_name = excluded.patient_name,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    calendar_id,
                    start.isoformat(),
                    patient_name,
                    datetime.now().isoformat(),
                ),
            )

    def load_last_booking(
        self, session_id: str
    ) -> tuple[str, datetime, str] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT calendar_id, start_iso, patient_name, updated_at
                FROM session_bookings
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            return None

        calendar_id, start_iso, patient_name, updated_at = row
        if datetime.fromisoformat(updated_at) < datetime.now() - _TTL:
            self.clear_last_booking(session_id)
            return None

        return calendar_id, datetime.fromisoformat(start_iso), patient_name

    def clear_last_booking(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                'DELETE FROM session_bookings WHERE session_id = ?',
                (session_id,),
            )


def merge_intent(saved: Intent | None, new: Intent) -> Intent:
    """Completa campos que faltaram na mensagem atual com o que já sabíamos.

    A mensagem atual sempre tem prioridade quando ela realmente informa
    algo; só usamos o valor salvo quando o campo atual veio vazio. O
    campo "intent" é especial: se a mensagem atual não deixou claro o
    que o paciente quer ("unknown") mas havia um pedido em andamento,
    continuamos nesse pedido em vez de esquecê-lo. O campo "language" é
    o oposto dos demais: o valor salvo é que tem prioridade, veja o
    comentário abaixo.
    """
    if saved is None:
        return new

    return Intent(
        intent=new.intent if new.intent != 'unknown' else saved.intent,
        professional_name=new.professional_name or saved.professional_name,
        specialty=new.specialty or saved.specialty,
        patient_name=new.patient_name or saved.patient_name,
        date=new.date or saved.date,
        time=new.time or saved.time,
        reason=new.reason or saved.reason,
        # Ao contrario dos outros campos, aqui o valor SALVO tem
        # prioridade sobre o novo, quando ja existe um. Uma mensagem
        # curta ou ambigua ("italo", so um nome) pode fazer a LLM
        # "adivinhar" um idioma errado para ela sozinha; se ja
        # sabiamos o idioma correto de um turno anterior mais claro
        # na mesma conversa, nao deixamos esse chute isolado
        # substituir o que ja estava certo.
        language=saved.language or new.language,
    )
