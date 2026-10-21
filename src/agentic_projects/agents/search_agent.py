from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

from agentic_projects.middleware.fallback_model import FallbackModelMiddleware
from agentic_projects.model.llm_model import LLMModel


class SearchAgent:
    def __init__(self) -> None:
        self.model, self.fallback_model = LLMModel().get_models()

    def invoke(self, prompt: str):
        tools = [TavilySearch()]
        middleware = [FallbackModelMiddleware(self.fallback_model)] if self.fallback_model is not None else []
        agent = create_agent(model=self.model, tools=tools, middleware=middleware)
        return agent.invoke({"messages": [HumanMessage(content=prompt)]})


sa = SearchAgent()
print(sa.invoke("What is the weather in Tokyo!"))
