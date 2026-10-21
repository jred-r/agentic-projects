import logging
import os

from dotenv import load_dotenv
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class LLMModel:
    def __init__(self) -> None:
        load_dotenv()

    def get_model(self) -> Runnable[LanguageModelInput, AIMessage]:
        gemini_model = ChatGoogleGenerativeAI(
            model=os.getenv("MODEL_NAME", "gemini-2.0-flash")
        )
        open_router_api_key = os.getenv("OPEN_ROUTER_API_KEY")
        if not open_router_api_key:
            logger.warning("OPEN_ROUTER_API_KEY is not set; Gemini has no fallback")
            return gemini_model

        open_router_model = ChatOpenAI(
            model=os.getenv("OPEN_ROUTER_MODEL", "openrouter/free"),
            api_key=SecretStr(open_router_api_key),
            base_url=os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
        return gemini_model.with_fallbacks([open_router_model])
