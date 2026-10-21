from agentic_projects.chains.summary_chain import SummaryChian
from agentic_projects.model.llm_model import LLMModel


class App:
    def __init__(self) -> None:
        self.model = LLMModel().get_model()

    def run(self) -> None:
        chain = SummaryChian(self.model)
        print("Say something about yourself!")
        information = input()
        response = chain.invoke(information)
        print(response.text)


if __name__ == "__main__":
    app = App()
    app.run()
