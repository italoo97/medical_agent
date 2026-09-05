"""Ponto de entrada da aplicacao.

Rodar como servidor HTTP (expoe POST /chat):
    poetry run uvicorn medical_agent.main:app --reload

Conversar direto pelo terminal, sem precisar de um cliente HTTP:
    poetry run python -m medical_agent.main
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request

from medical_agent.core.config import get_settings
from medical_agent.graph.factory import build_graph
from medical_agent.graph.state import AppointmentState
from medical_agent.schemas.chat import ChatRequest, ChatResponse
from medical_agent.services.appointment_service import AppointmentService
from medical_agent.services.google_calendar_client import (
    GoogleCalendarClient,
)
from medical_agent.services.llm import OpenRouterService

load_dotenv()


@dataclass
class AppContext:
    """Serviços e grafo montados uma unica vez, reaproveitados depois."""

    graph: Any
    appointment_service: AppointmentService


@asynccontextmanager
async def build_context() -> AsyncIterator[AppContext]:
    settings = get_settings()

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        llm_service = OpenRouterService(settings, http_client)
        calendar_client = GoogleCalendarClient(
            settings.google_service_account_json.get_secret_value(),
            timezone=settings.timezone,
        )
        appointment_service = AppointmentService(calendar_client)
        graph = build_graph(llm_service, appointment_service)

        yield AppContext(graph=graph, appointment_service=appointment_service)


async def ask(context: AppContext, user_message: str) -> AppointmentState:
    """Roda o grafo inteiro para uma mensagem e devolve o estado final.

    Devolver o estado inteiro (nao so o texto de resposta) deixa quem
    chamou livre para decidir o quanto de detalhe estruturado expor --
    o /chat usa isso para devolver intent/error/appointment_datetime
    junto da mensagem em linguagem natural; o modo CLI usa so o texto.
    """
    initial_state: AppointmentState = {
        'user_message': user_message,
        'intent': None,
        'professionals': context.appointment_service.list_professionals(),
        'calendar_id': None,
        'appointment_datetime': None,
        'response': '',
        'error': None,
    }
    final_state = await context.graph.ainvoke(initial_state)
    return final_state  # type: ignore[no-any-return]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with build_context() as context:
        app.state.context = context
        yield


app = FastAPI(title='Medical Appointment Agent', lifespan=lifespan)


def require_admin(x_admin_key: str = Header(...)) -> None:
    settings = get_settings()
    expected = settings.admin_api_key.get_secret_value()
    if x_admin_key != expected:
        raise HTTPException(status_code=401, detail='Invalid admin key')


@app.post('/chat', response_model=ChatResponse)
async def chat(chat_request: ChatRequest, request: Request) -> ChatResponse:
    context: AppContext = request.app.state.context
    final_state = await ask(context, chat_request.question)

    intent = final_state['intent']
    return ChatResponse(
        model='medical-agent',
        content=str(final_state['response']),
        intent=intent.intent if intent is not None else None,
        error=final_state['error'],
        calendar_id=final_state['calendar_id'],
        appointment_datetime=final_state['appointment_datetime'],
    )


@app.post(
    '/admin/professionals/{calendar_id}',
    dependencies=[Depends(require_admin)],
)
async def register_professional(
    calendar_id: str, request: Request
) -> dict[str, str]:
    """Registra uma agenda ja compartilhada com a service account.

    Compartilhar a agenda (na tela do Google Calendar) so cria a permissao;
    sem esse passo a service account nao "assina" a agenda e ela nunca
    aparece em list_professionals(). Chamar de novo para uma agenda ja
    registrada e seguro, nao da erro.
    """
    context: AppContext = request.app.state.context
    created = context.appointment_service.register_professional(calendar_id)
    return {'status': 'registered' if created else 'already_registered'}


@app.delete(
    '/admin/professionals/{calendar_id}',
    dependencies=[Depends(require_admin)],
)
async def remove_professional(
    calendar_id: str, request: Request
) -> dict[str, str]:
    """Desfaz o registro de uma agenda (profissional saiu, por exemplo)."""
    context: AppContext = request.app.state.context
    removed = context.appointment_service.remove_professional(calendar_id)
    return {'status': 'removed' if removed else 'not_registered'}


async def run_cli() -> None:
    print('Medical Appointment Agent -- modo terminal (ctrl+c sai)')
    async with build_context() as context:
        while True:
            user_message = input('\nVoce: ')
            final_state = await ask(context, user_message)
            print(f"Agente: {final_state['response']}")


if __name__ == '__main__':
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        print('\nAte mais!')
