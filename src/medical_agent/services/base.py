from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from medical_agent.schemas.chat import ChatResponse

SchemaT = TypeVar('SchemaT', bound=BaseModel)


class BaseLLMService(ABC):
    """Contrato para qualquer provedor de LLM usado pelo agente."""

    @abstractmethod
    async def generate(
        self, prompt: str, system_prompt: str | None = None
    ) -> ChatResponse:
        """Gera uma resposta livre (texto) para o prompt informado."""
        raise NotImplementedError

    @abstractmethod
    async def generate_structured(
        self, system_prompt: str, user_prompt: str, schema: type[SchemaT]
    ) -> SchemaT:
        """Gera uma resposta validada contra `schema`."""
        raise NotImplementedError
