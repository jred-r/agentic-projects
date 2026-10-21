from langchain_core.language_models.chat_models import BaseChatModel

from ..prompts.summary_prompt import create_summary_prompt


class SummaryChian:
    def __init__(self, model: BaseChatModel) -> None:
        self.chain = create_summary_prompt() | model

    def invoke(self, information: str):
        return self.chain.invoke(input={"information": information})
