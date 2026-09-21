"""Ponto de entrada usado pelo `langgraph dev` (ver `langgraph.json`).

Diferente de `main.build_context`, que devolve um `AppContext` dentro de
um `async with` e fecha o `httpx.AsyncClient` ao sair dele, aqui a fábrica
não fecha nada sozinha -- o processo do `langgraph dev` é quem controla o
ciclo de vida do servidor, e ele fica de pé por toda a sessão de
desenvolvimento.
"""

import asyncio
from typing import Any

import httpx
from dotenv import load_dotenv

from medical_agent.core.config import get_settings
from medical_agent.graph.factory import build_graph
from medical_agent.services.appointment_service import AppointmentService
from medical_agent.services.conversation_state import (
    ConversationStateStore,
)
from medical_agent.services.google_calendar_client import (
    GoogleCalendarClient,
)
from medical_agent.services.llm import OpenRouterService

load_dotenv()


async def graph() -> Any:
    """Monta e devolve o grafo compilado -- é isso que `langgraph.json`
    aponta em `graphs.agent`.

    `langgraph dev` roda essa fábrica dentro do event loop principal do
    servidor e instrumenta chamadas sincronas conhecidas (sqlite3, I/O de
    arquivo, etc.) para acusar quem bloqueia o loop -- algo que passa
    despercebido no `main.build_context` porque lá não há esse watchdog.
    `ConversationStateStore` (cria/abre o sqlite) e `GoogleCalendarClient`
    (le a service account e monta o client da API) fazem só esse tipo de
    trabalho sincrono e rapido, entao rodar cada um em `asyncio.to_thread`
    resolve sem precisar reescrever nada como assincrono de verdade.
    """
    settings = get_settings()
    http_client = httpx.AsyncClient(timeout=30.0)
    conversation_state_store = await asyncio.to_thread(ConversationStateStore)
    llm_service = OpenRouterService(settings, http_client)
    calendar_client = await asyncio.to_thread(
        GoogleCalendarClient,
        settings.google_service_account_json.get_secret_value(),
        timezone=settings.timezone,
    )
    appointment_service = AppointmentService(calendar_client)

    return build_graph(
        llm_service, appointment_service, conversation_state_store
    )
