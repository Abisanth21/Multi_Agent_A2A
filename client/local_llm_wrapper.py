# # local_llm_wrapper.py
# import requests
# from langchain_core.messages import AIMessage, HumanMessage

# class LocalLLMChat:
#     def __init__(self, model: str = "llama3.3", api_key: str = "", endpoint: str = ""):
#         self.model = model
#         self.api_key = api_key
#         self.endpoint = endpoint or "http://dvt-aiml.wv.mentorg.com:4000/v1/chat/completions"
#         self.headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }

#     def invoke(self, messages):
#         chat_history = []
#         for msg in messages:
#             if isinstance(msg, HumanMessage):
#                 chat_history.append({"role": "user", "content": msg.content})
#             elif isinstance(msg, AIMessage):
#                 chat_history.append({"role": "assistant", "content": msg.content})
#             else:
#                 chat_history.append({"role": "system", "content": msg.content})

#         payload = {
#             "model": self.model,
#             "messages": chat_history,
#             "max_tokens": 500,
#             "temperature": 0.5,
#         }

#         response = requests.post(self.endpoint, headers=self.headers, json=payload)
#         response.raise_for_status()
#         result = response.json()
#         content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
#         return AIMessage(content=content)
# local_llm_wrapper.py
import requests
import json
from typing import List, Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

class LocalLLMChat:
    def __init__(self, model: str = "llama3.3", api_key: str = "", endpoint: str = ""):
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint or "http://dvt-aiml.wv.mentorg.com:4000/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def bind_tools(self, tools: List[BaseTool]):
        """Bind tools to the LLM for function calling support."""
        self.tools = tools
        self.tool_schemas = []
        
        for tool in tools:
            # Convert LangChain tool to OpenAI function format
            tool_schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            
            # Add parameters from tool args schema if available
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema_dict = tool.args_schema.model_json_schema()
                if 'properties' in schema_dict:
                    tool_schema["function"]["parameters"]["properties"] = schema_dict['properties']
                if 'required' in schema_dict:
                    tool_schema["function"]["parameters"]["required"] = schema_dict['required']
            
            self.tool_schemas.append(tool_schema)
        
        return self

    def invoke(self, messages):
        chat_history = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                chat_history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                # Handle AI messages with potential tool calls
                message_dict = {"role": "assistant", "content": msg.content}
                
                # Check if AI message has tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    message_dict["tool_calls"] = []
                    for tool_call in msg.tool_calls:
                        message_dict["tool_calls"].append({
                            "id": tool_call.get("id", f"call_{len(message_dict['tool_calls'])}"),
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["args"])
                            }
                        })
                
                chat_history.append(message_dict)
            elif isinstance(msg, ToolMessage):
                # Handle tool responses
                chat_history.append({
                    "role": "tool",
                    "tool_call_id": getattr(msg, 'tool_call_id', 'unknown'),
                    "content": msg.content
                })
            elif isinstance(msg, SystemMessage):
                chat_history.append({"role": "system", "content": msg.content})
            else:
                # Fallback for other message types
                chat_history.append({"role": "system", "content": str(msg.content)})

        payload = {
            "model": self.model,
            "messages": chat_history,
            "max_tokens": 1000,
            "temperature": 0.1,
        }
        
        # Add tools if available
        if hasattr(self, 'tool_schemas') and self.tool_schemas:
            payload["tools"] = self.tool_schemas
            payload["tool_choice"] = "auto"

        try:
            response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            
            # Create AI message
            ai_message = AIMessage(content=content or "")
            
            # Handle tool calls if present
            if "tool_calls" in message and message["tool_calls"]:
                tool_calls = []
                for tool_call in message["tool_calls"]:
                    function = tool_call.get("function", {})
                    tool_calls.append({
                        "name": function.get("name"),
                        "args": json.loads(function.get("arguments", "{}")),
                        "id": tool_call.get("id")
                    })
                ai_message.tool_calls = tool_calls
            
            return ai_message
            
        except requests.exceptions.Timeout:
            return AIMessage(content="Request timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            return AIMessage(content=f"Error communicating with LLM: {str(e)}")
        except Exception as e:
            return AIMessage(content=f"Unexpected error: {str(e)}")
        
    def with_structured_output(self, schema):
        """For compatibility with LangChain structured output."""
        return self