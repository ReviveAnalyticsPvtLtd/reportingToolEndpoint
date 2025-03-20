import yaml
from langchain_cerebras import ChatCerebras
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

with open("prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)

codeGeneratorPrompt = PromptTemplate.from_template(prompts["codeGeneratorPrompt"])

codeGeneratorModel = ChatCerebras(
    model = "llama-3.3-70b",
    temperature = 1
)

codeGeneratorParser = StrOutputParser()

codeGeneratorChain = RunnablePassthrough() | codeGeneratorPrompt | codeGeneratorModel | codeGeneratorParser