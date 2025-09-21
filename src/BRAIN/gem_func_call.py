from google import genai
from google.genai import types
from DATA.tools import ALL_FUNCTIONS, UI_ALL_FUNCTIONS
from src.FUNCTION.Tools.get_env import EnvManager


class GeminiFunctionCaller:
    def __init__(self):
        self.genai_key = EnvManager.load_variable("genai_key")
        self.UI_ON = EnvManager.load_variable("UI")
        self.client = genai.Client(api_key=self.genai_key)
        self.tools_config = self._get_tools_config()

    def _get_tools_config(self) -> types.GenerateContentConfig:
        tools = UI_ALL_FUNCTIONS["tools"] if self.UI_ON == "YES" else ALL_FUNCTIONS["tools"]
        return types.GenerateContentConfig(
            temperature=0,
            tools=[types.Tool(function_declarations=tools)],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            )
        )

    def _call_gemini(self, query: str):
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=query,
                config=self.tools_config
            )
            return response.function_calls
        except Exception as e:
            print(f"[Gemini Error] {e}")
            return []

    def generate_function_calls(self, user_query: str) -> list[dict]:
        results = []
        function_calls = self._call_gemini(user_query)

        for fn in function_calls:
            results.append({
                "name": fn.name,
                "parameters": fn.args
            })

        return results
