from queryRephraser import queryRephraseChain
from codeGenerator import codeGeneratorChain
from failsafeAgent import failsafeModelChain
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

string = "import pandas as pd\nimport json\n\n"
for i in os.listdir("."):
    if (os.path.isfile(i)) & (i.split(".")[-1].lower() == "csv"):
      string += i.split(".")[0] + f" = pd.read_csv('{i}')\n"
string += "metadata = json.load(open('metadata.json', 'rb'))"
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
    code = "\n".join(state["generatedCode"].split("```")[-2].split("\n")[1:])
    response = pythonRepl.run(code)
    return {
        "codeOutput": response
    }

def outputEvaluationRouter(state: State):
    try:
        _ = json.loads(state["codeOutput"])
        return "pass"
    except json.JSONDecodeError:
        return "fail"

def failsafe(state: State):
    response = failsafeModelChain.invoke({
        "query": state["rephrasedQuery"],
        "metadata": state["metadata"]
    })
    return {
        "generatedCode": response
    }

def formatJsonResponse(state: State):
    if "codeOutput" in state.keys():
        try:
            response = json.loads(state["codeOutput"])
        except Exception as e:
            response = {"error": f"Endpoint says: {e}"}
        return {
            "finalOutput": response
        }
    else:
        return {
            "finalOutput": {"response": state["rephrasedQuery"]["doubt"]}
        }

def router(state: State):
    if state["rephrasedQuery"]["doubt"] == None:
        return "continue"
    else:
        return "interrupt"
    
workflow = StateGraph(State)

workflow.add_node("rephraseQuery", rephraseQuery)
workflow.add_node("generateCode", generateCode)
workflow.add_node("failsafe", failsafe)
workflow.add_node("runInPythonSandbox", runInPythonSandbox)
workflow.add_node("formatJsonResponse", formatJsonResponse)

workflow.add_edge(START, "rephraseQuery")
workflow.add_conditional_edges("rephraseQuery", router, {"continue": "generateCode", "interrupt": "formatJsonResponse"})
workflow.add_edge("generateCode", "runInPythonSandbox")
workflow.add_conditional_edges("runInPythonSandbox", outputEvaluationRouter, {"pass": "formatJsonResponse", "fail": "failsafe"})
workflow.add_edge("failsafe", "formatJsonResponse")
workflow.add_edge("formatJsonResponse", END)

workflow = workflow.compile()

def generate_chart_data(query: str):
    inputData = {"metadata": metadata, "inputQuery": query}
    try:
        responseJson = workflow.invoke(inputData)
        return responseJson["finalOutput"]
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