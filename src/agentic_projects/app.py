import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from .chains.summary_chain import SummaryChian


class App:
    def __init__(self) -> None:
        load_dotenv()

        self.model_name = os.getenv("MODEL_NAME")
        print(f"You are interacting with {self.model_name}")

        self.model = ChatGoogleGenerativeAI(model=self.model_name)

    def run(self) -> None:
        chain = SummaryChian(self.model)
        print("Say something about yourself!")
        information = input()
        response = chain.invoke(information)
        print(response.text)


if __name__ == "__main__":
    app = App()
    app.run()
