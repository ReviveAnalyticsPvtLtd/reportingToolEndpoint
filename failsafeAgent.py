import yaml
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

with open("prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)

codeGeneratorPrompt = PromptTemplate.from_template(prompts["codeGeneratorPrompt"])

failsafeModel = ChatGroq(
    model = "qwen-2.5-coder-32b",
    temperature = 1
)

codeGeneratorParser = StrOutputParser()

failsafeModelChain = RunnablePassthrough() | codeGeneratorPrompt | failsafeModel | codeGeneratorParser