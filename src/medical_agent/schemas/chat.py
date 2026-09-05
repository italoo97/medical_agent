from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=5)


class ChatResponse(BaseModel):
    model: str
    content: str
    intent: str | None = None
    error: str | None = None
    calendar_id: str | None = None
    appointment_datetime: str | None = None
