from queryRephraser import queryRephraseChain
from codeGenerator import codeGeneratorChain
from failsafeAgent import failsafeModelChain
from langgraph.graph import StateGraph, START, END
from langchain_experimental.utilities import PythonREPL
from flask import Flask, request, jsonify
from typing_extensions import TypedDict
from flask_caching import Cache
from flask_cors import CORS
from waitress import serve
import hashlib
import json
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.config["CACHE_TYPE"] = "simple"
app.config["CACHE_DEFAULT_TIMEOUT"] = 180

cache = Cache(app)

def generate_cache_key():
    """Generate a unique cache key based on the request body."""
    data = request.get_json()
    query = data.get("query", "")
    dataset = data.get("dataset", "")
    if ((not query) or (not dataset)):
        return None
    return hashlib.md5((query + dataset).encode()).hexdigest()

replManager = {
    "manufacturing": PythonREPL(),
    "banking": PythonREPL(),
    "supplyChain": PythonREPL(),
    "telecommunications": PythonREPL()
}

datasetsDir = os.path.join(os.getcwd(), "datasets")
for environment in replManager.keys():
    string = "import pandas as pd\nimport json\n\n"
    for table in os.listdir(os.path.join(datasetsDir, environment)):
        if (os.path.isfile(os.path.join(datasetsDir, environment, table))) & (table.split(".")[-1].lower() == "csv"):
            string += table.split(".")[0] + f" = pd.read_csv('{os.path.join(datasetsDir, environment, table)}')\n"
    string += f"metadata = json.load(open('{os.path.join(datasetsDir, environment, "metadata.json")}', 'rb'))"
    replManager[environment].run(string)

print("DATA LOAD SUCCESSFUL")

class State(TypedDict):
    dataset: str
    inputQuery: str
    rephrasedQuery: str
    metadata: str
    generatedCode: str
    codeOutput: str
    finalOutput: dict

def rephraseQuery(state: State):
    with open(f"{os.path.join(datasetsDir, state['dataset'], 'metadata.json')}", "rb") as f:
        metadata = json.load(f)
    response = queryRephraseChain.invoke({
        "query": state["inputQuery"],
        "metadata": metadata
    })
    return {
        "rephrasedQuery": response,
        "metadata": metadata
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
    response = replManager[state["dataset"]].run(code)
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
workflow.add_node("failsafePythonSandbox", runInPythonSandbox)
workflow.add_node("formatJsonResponse", formatJsonResponse)

workflow.add_edge(START, "rephraseQuery")
workflow.add_conditional_edges("rephraseQuery", router, {"continue": "generateCode", "interrupt": "formatJsonResponse"})
workflow.add_edge("generateCode", "runInPythonSandbox")
workflow.add_conditional_edges("runInPythonSandbox", outputEvaluationRouter, {"pass": "formatJsonResponse", "fail": "failsafe"})
workflow.add_edge("failsafe", "failsafePythonSandbox")
workflow.add_edge("failsafePythonSandbox", "formatJsonResponse")
workflow.add_edge("formatJsonResponse", END)

workflow = workflow.compile()

print("GRAPH COMPILATION SUCCESSFUL")

def generate_chart_data(query: str, dataset: str):
    inputData = {"dataset": dataset, "inputQuery": query}
    try:
        responseJson = workflow.invoke(inputData)
        return responseJson["finalOutput"]
    except Exception as e:
        return {"error": f"Endpoint says: {e}"}
        

@app.route("/generate_chart", methods=["POST"])
@cache.cached(timeout=180, key_prefix=generate_cache_key)
def generate_chart():
    try:
        data = request.get_json()
        query = data.get("query", "")
        dataset = data.get("dataset", "")
        if ((not query) or (not dataset)):
            return jsonify({"error": "Query/Dataset is required"}), 400
        
        chart_data = generate_chart_data(query = query, dataset = dataset)
        return jsonify(chart_data)
    except Exception as e:
        return jsonify({"error": f"Endpoint says: {e}"}), 500

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)