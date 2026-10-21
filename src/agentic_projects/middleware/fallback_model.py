from collections.abc import Callable
from typing import Any

import langchain.agents.middleware as agent_middleware
from langchain_core.language_models import BaseChatModel


class FallbackModelMiddleware(agent_middleware.AgentMiddleware):
    def __init__(self, fallback_model: BaseChatModel) -> None:
        self.fallback_model = fallback_model

    def wrap_model_call(
        self,
        request: agent_middleware.ModelRequest[Any],
        handler: Callable[[agent_middleware.ModelRequest[Any]], agent_middleware.ModelResponse[Any]],
    ) -> agent_middleware.ModelResponse[Any]:
        try:
            return handler(request)
        except Exception as error:
            return handler(request.override(model=self.fallback_model))
