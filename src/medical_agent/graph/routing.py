from medical_agent.graph.state import AppointmentState


def route_intent(state: AppointmentState) -> str:
    intent = state['intent']
    if intent is None:
        return 'unknown'
    return intent.intent
