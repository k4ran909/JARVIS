from typing import Union
import json
import re
from langchain_ollama import ChatOllama
from src.FUNCTION.Tools.get_env import EnvManager
from DATA.tools import ALL_FUNCTIONS, UI_ALL_FUNCTIONS, ALL_FUNCTIONS_EXAMPLE, UI_ALL_FUNCTIONS_EXAMPLE

# UI_ON = load_variable("UI")
# AVAILABLE_FUNCTION_NAMES_STRING = [func.get("name") for func in ALL_FUNCTIONS.get("tools")]
# SYSTEM_MESSAGE = f"""You are an AI that determines the best function to call based on user input.\n\n### Available Functions:\n{ALL_FUNCTIONS["tools"] if UI_ON == "NO" else UI_ALL_FUNCTIONS["tools"]}\n\n### Instructions:\n- Choose the function name.\n- Extract necessary arguments.\n- **Respond ONLY in valid JSON format** as follows:\n\n```json\n[\n   {{\n     "name": "function_name_here",\n     "parameters": {{\n         "arg1": "value1",\n         "arg2": "value2"\n     }}\n   }}\n]\n```\n\n### Examples:\n{ALL_FUNCTIONS_EXAMPLE if UI_ON == "NO" else UI_ALL_FUNCTIONS_EXAMPLE}\n```\n"""


UI_ON = EnvManager.load_variable("UI")
AVAILABLE_FUNCTION_NAMES_STRING = [func.get("name") for func in ALL_FUNCTIONS.get("tools")]
# - If no function is relevant, return an empty list: []
# - If the user query involves multiple actions, respond with a list of function calls in the correct order.

SYSTEM_MESSAGE = f"""You are an AI that determines the best function to call based on user input.

### Available Functions:
{ALL_FUNCTIONS["tools"] if UI_ON == "NO" else UI_ALL_FUNCTIONS["tools"]}

### Instructions:
- Choose the function name.
- Extract necessary arguments.
- **Respond ONLY in valid JSON format** as follows:

```json
[
    {{
    "name": "<function_name>",
        "arguments": {{
            "arg1": "<value1>",
            "arg2": "<value2>"
        }}
    }}
]
```

### Examples:
{ALL_FUNCTIONS_EXAMPLE if UI_ON == "NO" else UI_ALL_FUNCTIONS_EXAMPLE}
```
"""

class LocalFunctionCall:
    def __init__(self):
        self.model = EnvManager.load_variable("Function_call_model")

    def _load_tools(self, file_path: str) -> Union[dict, list]:
        try:
            with open(file_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            print("Error: Tools configuration file not found.")
            return []
        except json.JSONDecodeError:
            print("Error: Invalid JSON format.")
            return []

    def load_tools_message(self, file_path: str) -> str:
        tools = self._load_tools(file_path)
        return json.dumps(tools, indent=2)

    def _parse_tool_calls(self, response: str) -> Union[list, None]:
        try:
            # Remove markdown fences
            response = response.replace("```json", "").replace("```", "").strip()
            
            # Extract JSON-like part
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if not match:
                return None
            
            json_str = match.group(0)
            
            # Replace double braces with single braces
            json_str = json_str.replace("{{", "{").replace("}}", "}")
            
            # Parse JSON
            return json.loads(json_str)
        
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}\nOriginal string:\n{json_str}")
            return None


    def create_function_call(self, user_query: str) -> Union[list, None]:
        try:
            llm = ChatOllama(model=self.model, temprature=0)
            response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user_query}])
            functional_response = self._parse_tool_calls(response.content)
            return [
                func for func in functional_response if func.get("name", "").lower() in AVAILABLE_FUNCTION_NAMES_STRING
            ] if functional_response else None
        except Exception as e:
            print(f"Error creating function call: {e}")
            return None

if __name__ == "__main__":
    query = "please send email send email "
    function_caller = LocalFunctionCall()
    response = function_caller.create_function_call(query)
    print(response)
