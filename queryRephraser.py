import yaml
from langchain_groq import ChatGroq
from pydantic import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


class QueryRephraseOutput(BaseModel):
    rephrasedOutput: str | None = Field(
        description="A clear and concise rephrased version of the user's query. If the query is unclear, invalid, or requires clarification, this will be `None`."
    )
    doubt: str | None = Field(
        description="A message indicating any doubt, required clarification, or reason why the input query is invalid. If the query is successfully rephrased, this will be `None`."
    )

queryRephraseParser = JsonOutputParser(pydantic_object = QueryRephraseOutput)

with open("prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)

queryRephrasePrompt = PromptTemplate(
    template = prompts["queryRephrasePrompt"],
    input_variables = ["metadata", "query"],
    partial_variables = {"format_instructions": queryRephraseParser.get_format_instructions()}
)

queryRephraseModel = ChatGroq(
    model = "qwen-2.5-32b",
    temperature = 1,
    max_tokens = 512
)

queryRephraseChain = RunnablePassthrough() | queryRephrasePrompt | queryRephraseModel | queryRephraseParser
