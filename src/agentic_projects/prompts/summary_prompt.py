from langchain_core.prompts import PromptTemplate


def create_summary_prompt() -> PromptTemplate:
    template = """ 
            Given the {information} about a person, create:
                
                1. A short summary
                2. Two interesting facts
            """
    return PromptTemplate(input_variables=["information"], template=template)
