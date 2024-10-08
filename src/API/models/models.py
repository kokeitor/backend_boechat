from pydantic import BaseModel
from typing import Optional


class ChatResponse(BaseModel):
    userMessage: str
    iaResponse: Optional[str] = None


class OpenAIChatGraph(BaseModel):
    userMessage: str
    generationGraph: str
    context: str
    iaResponse: Optional[str] = None
