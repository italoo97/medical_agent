from medical_agent.graph.routing import route_intent
from medical_agent.prompts.v1.identify_intent import Intent


def test_routes_to_unknown_when_intent_is_missing() -> None:
    state = {'intent': None}

    assert route_intent(state) == 'unknown'  # type: ignore[arg-type]


def test_routes_to_the_extracted_intent_name() -> None:
    state = {'intent': Intent(intent='schedule')}

    assert route_intent(state) == 'schedule'  # type: ignore[arg-type]
