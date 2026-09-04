from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=5)


class ChatResponse(BaseModel):
    model: str
    content: str
