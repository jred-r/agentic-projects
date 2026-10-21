from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langsmith import traceable
import os

load_dotenv()
MAX_ITERATIONS = 10
model = os.getenv("MODEL_NAME")


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product"""
    prices = {"laptop": 1299.99, "headphones": 149.5, "keyboard": 89.50}
    print(f"executing get_product_price(product:{product})")
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Applies a discount based on the tier and returns final price. Available tiers: bronze, silver, gold."""
    print(f"executing apply_discount(price={price}, discount_tier={discount_tier})")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 20}
    discounted = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discounted / 100), 2)


@traceable(name="agent loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(model=model, model_provider="google_genai")
    llm_with_tools = llm.bind_tools(tools)
    messages = [SystemMessage("You are a helpful shopping assistant, you have access to the catalog and discount tools."), HumanMessage(question)]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {iteration}")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        # run as long as tool calls are required.
        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content
        else:
            init_tool_call = tool_calls[0]
            tool_name = init_tool_call.get("name")
            tool_args = init_tool_call.get("args", {})
            tool_call_id = init_tool_call.get("id")

            print(f" Tool Selected {tool_name} with args {tool_args}")
            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"tool not found {tool_name}")

            observation = tool_to_use.invoke(tool_args)

            print(f"tool result - {observation}")
            messages.append(ai_message)
            messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("iterations maxed")
    return None


if __name__ == "__main__":
    run_agent("What is the price of a laptop after applying a gold tier discount.")
