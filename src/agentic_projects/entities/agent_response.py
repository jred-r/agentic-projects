from typing import List

from pydantic import BaseModel, Field


class Source(BaseModel):
    Source: str = Field(description="the url of the source")


class AgentResponse(BaseModel):
    answer: str = Field(description="agents response")
    sources: List[Source] = Field(default_factory=list, description="list of sources")
