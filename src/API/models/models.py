from pydantic import BaseModel
from typing import Optional


class ChatResponse(BaseModel):
    userMessage: str
    iaResponse: Optional[str] = None
    files: Optional[list[str]] = None
