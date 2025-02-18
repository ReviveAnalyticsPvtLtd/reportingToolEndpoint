from queryRephraser import queryRephraseChain
from codeGenerator import codeGeneratorChain
from langgraph.graph import StateGraph, START, END
from langchain_experimental.utilities import PythonREPL
from flask import Flask, request, jsonify
from typing_extensions import TypedDict
from flask_cors import CORS
from waitress import serve
import json
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

pythonRepl = PythonREPL()

string = "import pandas as pd\n\n"
for i in os.listdir("."):
    if (os.path.isfile(i)) & (i.split(".")[-1].lower() == "csv"):
      string += i.split(".")[0] + f" = pd.read_csv('{i}')\n"
pythonRepl.run(string)

with open("metadata.json", "rb") as f:
    metadata = json.load(f)

class State(TypedDict):
    inputQuery: str
    metadata: str
    rephrasedQuery: str
    generatedCode: str
    codeOutput: str
    finalOutput: dict

def rephraseQuery(state: State):
    response = queryRephraseChain.invoke({
        "query": state["inputQuery"],
        "metadata": state["metadata"]
    })

    return {
        "rephrasedQuery": response
    }

def generateCode(state: State):
    response = codeGeneratorChain.invoke({
        "query": state["rephrasedQuery"],
        "metadata": state["metadata"]
    })
    return {
        "generatedCode": response
    }

def runInPythonSandbox(state: State):
    code = pythonRepl.sanitize_input(state["generatedCode"])
    response = pythonRepl.run(code)
    return {
        "codeOutput": response
    }

def formatJsonResponse(state: State):
    response = json.loads(state["codeOutput"])
    return {
        "finalOutput": response
    }

def router(state: State):
    if state["rephrasedQuery"]["doubt"] == None:
        return "continue"
    else:
        return "interrupt"
    
workflow = StateGraph(State)

workflow.add_node("rephraseQuery", rephraseQuery)
workflow.add_node("generateCode", generateCode)
workflow.add_node("runInPythonSandbox", runInPythonSandbox)
workflow.add_node("formatJsonResponse", formatJsonResponse)

workflow.add_edge(START, "rephraseQuery")
workflow.add_conditional_edges("rephraseQuery", router, {"continue": "generateCode", "interrupt": "formatJsonResponse"})
workflow.add_edge("generateCode", "runInPythonSandbox")
workflow.add_edge("runInPythonSandbox", "formatJsonResponse")
workflow.add_edge("formatJsonResponse", END)

workflow = workflow.compile()

def generate_chart_data(query: str):
    inputData = {"metadata": metadata, "query": query}
    try:
        responseJson = workflow.invoke(inputData)
        return responseJson
    except Exception as e:
        return {"error": f"Endpoint says: {e}"}
        

@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.get_json()
        query = data.get("query", "")
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        chart_data = generate_chart_data(query)
        return jsonify(chart_data)
    except Exception as e:
        return jsonify({"error": "An error occurred while processing the request."}), 500

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)