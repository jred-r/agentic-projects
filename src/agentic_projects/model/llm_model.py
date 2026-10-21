import os

from dotenv import load_dotenv
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


class LLMModel:
    def __init__(self) -> None:
        load_dotenv()

    def get_models(self) -> tuple[ChatOpenAI, ChatGoogleGenerativeAI]:
        gemini_model = ChatGoogleGenerativeAI(model=os.getenv("MODEL_NAME", "gemini-2.0-flash"))
        open_router_api_key = os.getenv("OPEN_ROUTER_API_KEY", "Not Set")
        open_router_model = ChatOpenAI(
            model=os.getenv("OPEN_ROUTER_MODEL", "openrouter/free"),
            api_key=SecretStr(open_router_api_key),
            base_url=os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
        return open_router_model, gemini_model

    def get_model(self) -> Runnable[LanguageModelInput, AIMessage]:
        open_router_model, gemini_model = self.get_models()
        return gemini_model.with_fallbacks([open_router_model])
