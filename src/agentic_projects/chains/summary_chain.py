from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from ..prompts.summary_prompt import create_summary_prompt


class SummaryChian:
    def __init__(self, model: Runnable[LanguageModelInput, AIMessage]) -> None:
        self.chain = create_summary_prompt() | model

    def invoke(self, information: str):
        return self.chain.invoke(input={"information": information})
