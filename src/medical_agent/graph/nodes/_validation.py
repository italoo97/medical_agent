from medical_agent.prompts.v1.identify_intent import Intent


def build_missing_fields_error(
    intent: Intent | None, action: str
) -> str | None:
    """Descreve especificamente o que falta no pedido, ou None se completo.

    patient_name e obrigatorio aqui de proposito: sem ele, duas pessoas que
    nao dizem o nome cairiam no mesmo valor generico, e o cancelamento (que
    casa por calendar_id + patient_name + horario) poderia acabar
    encontrando ou cancelando a consulta de outra pessoa. Exigir o nome
    evita esse vazamento entre pacientes.
    """
    if intent is None:
        return f'Não consegui {action}: não entendi seu pedido.'

    missing = []
    if not intent.date:
        missing.append('a data')
    if not intent.time:
        missing.append('o horário')
    if not intent.patient_name:
        missing.append('seu nome')

    if not missing:
        return None

    joined = ', '.join(missing)
    return f'Não consegui {action} porque faltou informar: {joined}.'
