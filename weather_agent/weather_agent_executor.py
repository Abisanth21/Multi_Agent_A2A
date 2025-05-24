# weather_agent/weather_agent_executor.py
import asyncio
import logging
from typing import Dict, Any, List
import json
import os

import httpx
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
# from langchain_openai import ChatOpenAI
from local_llm_wrapper import LocalLLMChat

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from typing_extensions import Annotated, TypedDict

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Artifact,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import get_text_parts
from a2a.utils.errors import ServerError

logger = logging.getLogger(__name__)

# Define the state for our graph
class WeatherState(TypedDict):
    messages: Annotated[list, add_messages]
    location: str
    weather_data: Dict[str, Any]

# @tool
@tool
def get_current_weather(location: str) -> str:
    # print(f"[Tool] Called get_current_weather for: {location}")
    """Get current weather information for a specific location."""
    try:
        api_key = os.getenv('WEATHER_API_KEY')
        print(f"[Tool] Called get_current_weather for: {location}")
        if not api_key:
            return "Weather API key not configured"

        base_url = "http://api.weatherapi.com/v1/current.json"
        params = {
            'key': api_key,
            'q': location,
            'aqi': 'no'
        }
        
        import requests
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            weather_info = {
                'location': data['location']['name'],
                'country': data['location']['country'],
                'temperature (C)': data['current']['temp_c'],
                'condition': data['current']['condition']['text'],
                'humidity': data['current']['humidity'],
                'wind_kph': data['current']['wind_kph']
            }
            return json.dumps(weather_info)
        else:
            return f"Could not fetch weather data. Error: {response.status_code}"

    except Exception as e:
        return f"Error fetching weather data: {str(e)}"

@tool
def get_weather_forecast(location: str, days: int = 5) -> str:
    """Get weather forecast for a specific location."""
    try:
        api_key = os.getenv('WEATHER_API_KEY')
        if not api_key:
            return "Weather API key not configured"
        
        # Using OpenWeatherMap forecast API
        base_url = "http://api.weatherapi.com/v1/forecast.json"
        params = {
        'key': api_key,
        'q': location,
        'days': days,
        'aqi': 'no',
        'alerts': 'no'
        }
        
        import requests
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            forecasts = []
            
            for item in data['list'][:days*2]:  # Limit to requested days
                forecast = {
                    'datetime': item['dt_txt'],
                    'temperature': item['main']['temp'],
                    'description': item['weather'][0]['description'],
                    'humidity': item['main']['humidity']
                }
                forecasts.append(forecast)
            
            return json.dumps({'location': data['city']['name'], 'forecasts': forecasts})
        else:
            return f"Could not fetch forecast data for {location}. Error: {response.status_code}"
            
    except Exception as e:
        return f"Error fetching forecast data: {str(e)}"

class WeatherAgentExecutor(AgentExecutor):
    """Weather Agent using LangGraph for ReAct pattern."""

    def __init__(self):
        self.llm = LocalLLMChat(
        model=os.getenv("LOCAL_LLM_MODEL", "llama3.3"),
        api_key=os.getenv("LOCAL_LLM_API_KEY")
        )
        # self.llm = ChatOpenAI(
        #     model="gpt-4o-mini", 
        #     temperature=0,
        #     api_key=os.getenv('OPENAI_API_KEY')
        # )
        
        # Available tools
        self.tools = [get_current_weather, get_weather_forecast]
        
        # Create the graph
        self.graph = self._create_weather_graph()

    def _create_weather_graph(self) -> StateGraph:
        """Create the LangGraph for weather processing."""
        
        def should_continue(state: WeatherState) -> str:
            """Decide whether to continue or end."""
            messages = state["messages"]
            last_message = messages[-1]
            
            # If the LLM makes a tool call, then we route to the "tools" node
            if last_message.tool_calls:
                return "tools"
            # Otherwise, we stop (reply to the user)
            return END

        def call_model(state: WeatherState):
            """Call the LLM with the current state."""
            messages = state["messages"]
            
            # Add system message for weather agent behavior
            system_message = SystemMessage(content="""
            You are a helpful weather agent. Your job is to provide accurate weather information.

            When users ask about weather:
            1. Extract the location from their query
            2. Use appropriate tools to get weather data
            3. Present the information in a clear, user-friendly format
            4. Include relevant details like temperature, conditions, humidity, etc.

            If location is not clear, ask for clarification.
            Always use the tools available to get real-time data.
            """)
            
            full_messages = [system_message] + messages
            response = self.llm.invoke(full_messages)
            return {"messages": [response]}

        # Create the graph
        workflow = StateGraph(WeatherState)
        
        # Add nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
        )
        
        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Execute the weather agent."""
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        if not context.current_task:
            updater.submit()
        updater.start_work()
        
        try:
            # Extract text from the message
            user_message = ""
            for part in context.message.parts:
                if isinstance(part.root, TextPart):
                    user_message += part.root.text
            
            # Update status
            updater.update_status(
                TaskState.working,
                message=updater.new_agent_message(
                    [Part(root=TextPart(text="Processing weather request..."))]
                ),
            )
            
            # Initialize state
            initial_state = WeatherState(
                messages=[HumanMessage(content=user_message)],
                location="",
                weather_data={}
            )
            
            # Run the graph
            final_state = await asyncio.to_thread(
                self.graph.invoke, 
                initial_state
            )
            
            # Get the final response
            final_message = final_state["messages"][-1]
            response_text = final_message.content
            
            # Create artifact with the response
            response_parts = [Part(root=TextPart(text=response_text))]
            updater.add_artifact(response_parts)
            updater.complete()
            
        except Exception as e:
            logger.error(f"Error in weather agent execution: {e}")
            error_message = f"Sorry, I encountered an error while processing your weather request: {str(e)}"
            error_parts = [Part(root=TextPart(text=error_message))]
            updater.add_artifact(error_parts)
            updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=UnsupportedOperationError())

# # weather_agent/weather_agent_executor.py
# import asyncio
# import logging
# from typing import Dict, Any, List
# import json
# import os

# import httpx
# from langgraph.graph import StateGraph, END
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode
# from local_llm_wrapper import LocalLLMChat

# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_core.tools import tool
# from typing_extensions import Annotated, TypedDict

# from a2a.server.agent_execution import AgentExecutor, RequestContext
# from a2a.server.events.event_queue import EventQueue
# from a2a.server.tasks import TaskUpdater
# from a2a.types import (
#     Artifact,
#     Part,
#     TaskState,
#     TextPart,
#     UnsupportedOperationError,
# )
# from a2a.utils import get_text_parts
# from a2a.utils.errors import ServerError

# logger = logging.getLogger(__name__)

# # Define the state for our graph
# class WeatherState(TypedDict):
#     messages: Annotated[list, add_messages]
#     location: str
#     weather_data: Dict[str, Any]

# @tool
# def get_current_weather(location: str) -> str:
#     """Get current weather information for a specific location."""
#     try:
#         api_key = os.getenv('WEATHER_API_KEY')
#         if not api_key:
#             # Mock weather data for demonstration
#             mock_weather = {
#                 'kanyakumari': {'temp': 28, 'condition': 'Sunny', 'humidity': 75},
#                 'new york': {'temp': 15, 'condition': 'Cloudy', 'humidity': 60},
#                 'london': {'temp': 12, 'condition': 'Rainy', 'humidity': 80},
#                 'tokyo': {'temp': 20, 'condition': 'Partly Cloudy', 'humidity': 65},
#                 'default': {'temp': 22, 'condition': 'Clear', 'humidity': 55}
#             }
            
#             location_key = location.lower()
#             weather = mock_weather.get(location_key, mock_weather['default'])
            
#             weather_info = {
#                 'location': location.title(),
#                 'country': 'Mock Country',
#                 'temperature (C)': weather['temp'],
#                 'condition': weather['condition'],
#                 'humidity': weather['humidity'],
#                 'wind_kph': 15,
#                 'note': 'Mock data - Weather API key not configured'
#             }
#             return json.dumps(weather_info)

#         base_url = "http://api.weatherapi.com/v1/current.json"
#         params = {
#             'key': api_key,
#             'q': location,
#             'aqi': 'no'
#         }
        
#         import requests
#         response = requests.get(base_url, params=params, timeout=10)
#         if response.status_code == 200:
#             data = response.json()
#             weather_info = {
#                 'location': data['location']['name'],
#                 'country': data['location']['country'],
#                 'temperature (C)': data['current']['temp_c'],
#                 'condition': data['current']['condition']['text'],
#                 'humidity': data['current']['humidity'],
#                 'wind_kph': data['current']['wind_kph']
#             }
#             return json.dumps(weather_info)
#         else:
#             return f"Could not fetch weather data. Error: {response.status_code}"

#     except Exception as e:
#         logger.error(f"Error in get_current_weather: {e}")
#         return f"Error fetching weather data: {str(e)}"

# @tool
# def get_weather_forecast(location: str, days: int = 3) -> str:
#     """Get weather forecast for a specific location."""
#     try:
#         api_key = os.getenv('WEATHER_API_KEY')
#         if not api_key:
#             # Mock forecast data
#             mock_forecasts = []
#             base_temp = 25
#             conditions = ['Sunny', 'Partly Cloudy', 'Cloudy', 'Light Rain']
            
#             for i in range(days):
#                 temp_variation = (-2, 0, 2)[i % 3]
#                 mock_forecasts.append({
#                     'date': f'2024-05-{22 + i}',
#                     'temperature_high': base_temp + temp_variation + 3,
#                     'temperature_low': base_temp + temp_variation - 3,
#                     'condition': conditions[i % len(conditions)],
#                     'humidity': 60 + (i * 5)
#                 })
            
#             return json.dumps({
#                 'location': location.title(),
#                 'forecasts': mock_forecasts,
#                 'note': 'Mock data - Weather API key not configured'
#             })
        
#         base_url = "http://api.weatherapi.com/v1/forecast.json"
#         params = {
#             'key': api_key,
#             'q': location,
#             'days': days,
#             'aqi': 'no',
#             'alerts': 'no'
#         }
        
#         import requests
#         response = requests.get(base_url, params=params, timeout=10)
        
#         if response.status_code == 200:
#             data = response.json()
#             forecasts = []
            
#             for day in data['forecast']['forecastday']:
#                 forecast = {
#                     'date': day['date'],
#                     'temperature_high': day['day']['maxtemp_c'],
#                     'temperature_low': day['day']['mintemp_c'],
#                     'condition': day['day']['condition']['text'],
#                     'humidity': day['day']['avghumidity']
#                 }
#                 forecasts.append(forecast)
            
#             return json.dumps({
#                 'location': data['location']['name'], 
#                 'forecasts': forecasts
#             })
#         else:
#             return f"Could not fetch forecast data for {location}. Error: {response.status_code}"
            
#     except Exception as e:
#         logger.error(f"Error in get_weather_forecast: {e}")
#         return f"Error fetching forecast data: {str(e)}"


# class WeatherAgentExecutor(AgentExecutor):
#     """Weather Agent using LangGraph for ReAct pattern."""

#     def __init__(self):
#         self.llm = LocalLLMChat(
#             model=os.getenv("LOCAL_LLM_MODEL", "llama3.3"),
#             api_key=os.getenv("LOCAL_LLM_API_KEY")
#         )
        
#         # Available tools
#         self.tools = [get_current_weather, get_weather_forecast]
        
#         # Bind tools to LLM for function calling
#         self.llm_with_tools = self.llm.bind_tools(self.tools)
        
#         # Create the graph
#         self.graph = self._create_weather_graph()

#     def _create_weather_graph(self) -> StateGraph:
#         """Create the LangGraph for weather processing."""
        
#         def should_continue(state: WeatherState) -> str:
#             """Decide whether to continue or end."""
#             messages = state["messages"]
#             last_message = messages[-1]
            
#             # If the LLM makes a tool call, then we route to the "tools" node
#             if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
#                 return "tools"
#             # Otherwise, we stop (reply to the user)
#             return END

#         def call_model(state: WeatherState):
#             """Call the LLM with the current state."""
#             messages = state["messages"]
            
#             # Add system message for weather agent behavior
#             system_message = SystemMessage(content="""
#             You are a helpful weather agent. Your job is to provide accurate weather information.

#             When users ask about weather:
#             1. Extract the location from their query
#             2. Determine if they want current weather or forecast
#             3. Use appropriate tools to get weather data
#             4. Present the information in a clear, user-friendly format
#             5. Include relevant details like temperature, conditions, humidity, etc.

#             Available tools:
#             - get_current_weather: Get current weather for a location
#             - get_weather_forecast: Get weather forecast for multiple days

#             If location is not clear, ask for clarification.
#             Always use the tools available to get real-time data.
#             Present temperatures in Celsius and include condition descriptions.
#             """)
            
#             full_messages = [system_message] + messages
#             response = self.llm_with_tools.invoke(full_messages)
#             return {"messages": [response]}

#         # Create the graph
#         workflow = StateGraph(WeatherState)
        
#         # Add nodes
#         workflow.add_node("agent", call_model)
#         workflow.add_node("tools", ToolNode(self.tools))
        
#         # Set entry point
#         workflow.set_entry_point("agent")
        
#         # Add conditional edges
#         workflow.add_conditional_edges(
#             "agent",
#             should_continue,
#         )
        
#         # Add edge from tools back to agent
#         workflow.add_edge("tools", "agent")
        
#         return workflow.compile()

#     async def execute(self, context: RequestContext, event_queue: EventQueue):
#         """Execute the weather agent."""
#         updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
#         if not context.current_task:
#             updater.submit()
#         updater.start_work()
        
#         try:
#             # Extract text from the message
#             user_message = ""
#             for part in context.message.parts:
#                 if isinstance(part.root, TextPart):
#                     user_message += part.root.text
            
#             logger.info(f"Processing weather request: {user_message}")
            
#             # Update status
#             updater.update_status(
#                 TaskState.working,
#                 message=updater.new_agent_message(
#                     [Part(root=TextPart(text="Processing weather request..."))]
#                 ),
#             )
            
#             # Initialize state
#             initial_state = WeatherState(
#                 messages=[HumanMessage(content=user_message)],
#                 location="",
#                 weather_data={}
#             )
            
#             # Run the graph
#             final_state = await asyncio.to_thread(
#                 self.graph.invoke, 
#                 initial_state
#             )
            
#             # Get the final response
#             final_message = final_state["messages"][-1]
#             response_text = final_message.content
            
#             logger.info(f"Weather agent response: {response_text}")
            
#             # Create artifact with the response
#             response_parts = [Part(root=TextPart(text=response_text))]
#             updater.add_artifact(response_parts)
#             updater.complete()
            
#         except Exception as e:
#             logger.error(f"Error in weather agent execution: {e}", exc_info=True)
#             error_message = f"Sorry, I encountered an error while processing your weather request: {str(e)}"
#             error_parts = [Part(root=TextPart(text=error_message))]
#             updater.add_artifact(error_parts)
#             updater.complete()

#     async def cancel(self, context: RequestContext, event_queue: EventQueue):
#         raise ServerError(error=UnsupportedOperationError())