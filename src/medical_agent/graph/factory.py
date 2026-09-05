from typing import Any

from langgraph.graph import END, StateGraph

from medical_agent.graph.nodes.canceller import make_canceller_node
from medical_agent.graph.nodes.checker import make_checker_node
from medical_agent.graph.nodes.identify_intent import (
    make_identify_intent_node,
)
from medical_agent.graph.nodes.message_generator import (
    make_message_generator_node,
)
from medical_agent.graph.nodes.scheduler import make_scheduler_node
from medical_agent.graph.routing import route_intent
from medical_agent.graph.state import AppointmentState
from medical_agent.services.appointment_service import AppointmentService
from medical_agent.services.base import BaseLLMService


def build_graph(
    llm_service: BaseLLMService,
    appointment_service: AppointmentService,
) -> Any:
    """Monta e compila o StateGraph do agente de agendamento."""
    graph = StateGraph(AppointmentState)

    graph.add_node(  # type: ignore[call-overload]
        'identify_intent',
        make_identify_intent_node(llm_service),
    )
    graph.add_node(  # type: ignore[call-overload]
        'scheduler',
        make_scheduler_node(appointment_service),
    )
    graph.add_node(  # type: ignore[call-overload]
        'canceller',
        make_canceller_node(appointment_service),
    )
    graph.add_node(  # type: ignore[call-overload]
        'checker',
        make_checker_node(appointment_service),
    )
    graph.add_node(  # type: ignore[call-overload]
        'message_generator',
        make_message_generator_node(llm_service),
    )

    graph.set_entry_point('identify_intent')

    graph.add_conditional_edges(
        'identify_intent',
        route_intent,
        {
            'schedule': 'scheduler',
            'cancel': 'canceller',
            'check': 'checker',
            'unknown': 'message_generator',
        },
    )

    graph.add_edge('scheduler', 'message_generator')
    graph.add_edge('canceller', 'message_generator')
    graph.add_edge('checker', 'message_generator')
    graph.add_edge('message_generator', END)

    return graph.compile()
