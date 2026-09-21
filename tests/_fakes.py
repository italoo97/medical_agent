from medical_agent.prompts.v1.identify_intent import Intent
from medical_agent.schemas.chat import ChatResponse
from medical_agent.services.base import BaseLLMService


class FakeLLMService(BaseLLMService):
    """Dublê de BaseLLMService -- nenhuma chamada de rede.

    `intents` e uma fila: cada chamada a generate_structured devolve o
    proximo item; quando a fila acaba, repete o ultimo. Isso permite
    simular varios turnos de uma conversa (cada um com um Intent
    diferente vindo "da LLM"), como acontece de verdade no /chat.
    """

    def __init__(
        self,
        intents: list[Intent] | None = None,
        chat_content: str = 'ok',
    ) -> None:
        self._intents = list(intents) if intents else []
        self.chat_content = chat_content
        self.received_system_prompts: list[str] = []
        self.received_user_prompts: list[str] = []

    async def generate(
        self, prompt: str, system_prompt: str | None = None
    ) -> ChatResponse:
        self.received_system_prompts.append(system_prompt or '')
        self.received_user_prompts.append(prompt)
        return ChatResponse(model='fake-model', content=self.chat_content)

    async def generate_structured(  # type: ignore[override]
        self, system_prompt: str, user_prompt: str, schema: type[Intent]
    ) -> Intent:
        self.received_system_prompts.append(system_prompt)
        self.received_user_prompts.append(user_prompt)
        assert self._intents, 'FakeLLMService: no intent configured'
        next_intent = self._intents[0]
        if len(self._intents) > 1:
            self._intents.pop(0)
        return schema.model_validate(next_intent.model_dump())
