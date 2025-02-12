from flask import Flask, request, jsonify
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.utilities import PythonREPL
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from flask_cors import CORS
from waitress import serve
import pandas as pd
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

llm = ChatGroq(model = "deepseek-r1-distill-llama-70b", temperature = 1)
outputParser = StrOutputParser()

prompt = """
You are **ChartDataCreator**, an expert AI in generating precise chart data in **JSON format** tailored for Chart.js. Your task is to:

1. **Analyze the given metadata** (provided in YAML format) to understand the available data.

2. **Interpret the user query** and determine the best chart type from the following options: `line`, `scatter`, `bar`, `radar`, `bubble`, `polar`, `pie`, or `doughnut`. Ensure that you always select one of these types, as they are required for consistency.

3. **Copy the original dataframe** before performing any operations:
   - Create a copy of each dataframe, such as `iris_dataset_copy = iris_dataset.copy()`, and perform all further operations only on these copies to maintain data integrity.

4. **Generate a valid, JSON-serializable dataset** based on the query using the preloaded dataframes (or their copies). The generated JSON must adhere to the following strict structure for compatibility with Chart.js:

```json
{{
    "chartType": "<One of 'line', 'scatter', 'bar', 'radar', 'bubble', 'polar', 'pie', 'doughnut'>",
    "data": {{
        "labels": <list of labels for the chart (e.g., x-axis values)>,
        "datasets": [
            {{
                "label": "<name of dataset1>",
                "data": <array of data points corresponding to each label>
            }},
            {{
                "label": "<name of dataset2 (if applicable)>",
                "data": <array of values>
            }}
        ]
    }}
}}
```

5. **Return a standalone Python code block ONLY** to **print** the JSON response and nothing else. This code must:
   - Use only the preloaded dataframes (as referenced in the metadata) or their copies.
   - Be error-free, runnable, and ready to execute with necessary imports.
   - Serialize the output into JSON correctly.

### **Dataset Structure Based on Chart Type**
The `data` object must follow the structure specific to the chart type:
- **bar** - `"data": {{"labels": <list of labels>, "datasets": <list of dictionaries, each with keys "label" and "data">}}`
- **bubble** - `"data": {{"datasets": <list of dictionaries with keys: "label": <label>, "data": <list of dictionaries each with keys "x", "y", and "r">>}}`
- **pie/doughnut** - `"data": {{"labels": <list of labels>, "datasets": <list of dictionaries, each with keys "label" and "data">}}`
- **line** - `"data": {{"labels": <list of labels>, "datasets": <list of dictionaries, each with keys "label" and "data">}}`
- **polar** - `"data": {{"labels": <list of labels>, "datasets": <list of dictionaries, each with keys "label" and "data">}}`
- **radar** - `"data": {{"labels": <list of labels>, "datasets": <list of dictionaries, each with keys "label" and "data">}}`
- **scatter** - `"data": {{"datasets": <list of dictionaries with keys: "label": <label>, "data": <list of dictionaries each with keys "x" and "y">>}}`


### **Example of Python Code Output:**

#### **Metadata (YAML)**
```yaml
{{
  "<dataframe1>": {{
    "description": "<Description of the dataframe>",
    "columns": [
      {{"name": "<column1>", "type": "<column1 datatype>", "description": "<column1 description>"}},
      {{"name": "<column2>", "type": "<column2 datatype>", "description": "<column2 description>"}}
    ],
    "sample_row": {{
      "<column1>": "<value1>",
      "<column2>": "<value2>"
    }}
  }},
  "<dataframe2>": {{
    ...
  }}
}}

#### **Query**
*"Generate a bar chart showing monthly revenue trends."*

**Expected Python Output for the given metadata and query**

```python
import json
# Assuming sales_data_copy is the copy of the original preloaded dataframe, given in the metadata
sales_data_copy = sales_data.copy()
labels = sales_data_copy["month"].tolist()
revenue_data = sales_data_copy["revenue"].tolist()
chart_json = {{
    "chartType": "bar",
    "data": {{
        "labels": labels,
        "datasets": [
            {{
                "label": "Monthly Revenue",
                "data": revenue_data
            }}
        ]
    }}
}}
print(json.dumps(chart_json, indent=4))
```

### **Guidelines for Response Generation**
- Always perform operations only on copies of dataframes to prevent altering the original data.
- Select the most suitable Chart.js chart type from the following: `line`, `scatter`, `bar`, `radar`, `bubble`, `polar`, `pie`, or `doughnut`.
- NEVER include any color details in the output JSON. Chart.js will handle the colors dynamically in the frontend.
- Extract relevant labels and datasets from the metadata and provide only the necessary data for the chart.
- Ensure the generated JSON response is **fully serializable** and **error-free**.
- Generate a **standalone Python script** only in a single code block that runs smoothly with proper imports and no additional dependencies.
- You must not generate anything extra apart from the Python script code block.
- The code block MUST print the response json at the end.

### **Handling Unclear, Absurd, or Impossible Queries**
If the query cannot be answered due to:
- Missing or unrelated data in metadata.
- Unclear instructions (e.g., vague column names or ambiguous intent).
- Absurdity (e.g., "Show a scatter plot of a single category").
Respond with a JSON object in a Python code block under the key "response" that explains why the request is invalid and how the user can clarify it.
### Example of response in case of Unclear/Absurd/Impossible queries**
```python
import json

response = {{
    "response": "The requested chart cannot be generated because the metadata does not contain relevant data. Please provide a more specific query, ensuring it aligns with the available dataset."
}}
print(json.dumps(response, indent=4))
```
**Be highly meticulous—this JSON response is crucial for an API workflow!**

HERE'RE THE INPUTS GIVEN TO YOU:
metadata: {metadata}
query: {query}
"""
prompt = ChatPromptTemplate.from_template(prompt)

chain = {
    "metadata": RunnableLambda(lambda x: x["metadata"]),
    "query": RunnableLambda(lambda x: x["query"])
} | prompt | llm | outputParser

def generate_chart_data(query: str):
    inputData = {"metadata": metadata, "query": query}
    responseJson = {}
    for _ in range(5):
        try:
            response = chain.invoke(inputData)
            response = pythonRepl.run("\n".join(response.split("```")[-2].split("\n")[1:]))
            responseJson = json.loads(response)
            if responseJson:
                break
        except Exception:
            continue
    return responseJson

@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    chart_data = generate_chart_data(query)
    return jsonify(chart_data)

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)