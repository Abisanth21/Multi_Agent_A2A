# currency_agent/currency_agent_executor.py
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
class CurrencyState(TypedDict):
    messages: Annotated[list, add_messages]
    from_currency: str
    to_currency: str
    amount: float
    exchange_data: Dict[str, Any]

# @tool
@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Get current exchange rate between two currencies using exchangerate-api.com"""
    try:
        api_key = os.getenv('EXCHANGE_API_KEY')
        base_url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_currency.upper()}/{to_currency.upper()}"

        import requests
        response = requests.get(base_url)

        if response.status_code == 200:
            data = response.json()
            rate = data.get("conversion_rate")
            result = {
                'from_currency': from_currency.upper(),
                'to_currency': to_currency.upper(),
                'exchange_rate': rate,
                'timestamp': data.get("time_last_update_utc")
            }
            return json.dumps(result)
        else:
            return f"Could not fetch exchange rate. Error: {response.status_code}"

    except Exception as e:
        return f"Error fetching exchange rate: {str(e)}"

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another using exchangerate-api.com"""
    try:
        rate_response = get_exchange_rate(from_currency, to_currency)
        rate_data = json.loads(rate_response)

        if 'exchange_rate' not in rate_data:
            return rate_response

        exchange_rate = rate_data['exchange_rate']
        converted = round(amount * exchange_rate, 2)

        return json.dumps({
            'original_amount': amount,
            'from_currency': from_currency.upper(),
            'to_currency': to_currency.upper(),
            'exchange_rate': exchange_rate,
            'converted_amount': converted,
            'timestamp': rate_data.get("timestamp")
        })

    except Exception as e:
        return f"Error converting currency: {str(e)}"

@tool
def get_multiple_rates(base_currency: str, target_currencies: List[str] = None) -> str:
    """Get multiple exchange rates from a base currency using exchangerate-api.com"""
    try:
        if not target_currencies:
            target_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF']

        api_key = os.getenv('EXCHANGE_API_KEY')
        base_url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{base_currency.upper()}"

        import requests
        response = requests.get(base_url)

        if response.status_code == 200:
            data = response.json()
            all_rates = data.get("conversion_rates", {})
            filtered = {cur: all_rates[cur] for cur in target_currencies if cur in all_rates}

            return json.dumps({
                'base_currency': base_currency.upper(),
                'rates': filtered,
                'timestamp': data.get("time_last_update_utc")
            })

        else:
            return f"Could not fetch multiple exchange rates. Error: {response.status_code}"

    except Exception as e:
        return f"Error fetching rates: {str(e)}"


class CurrencyAgentExecutor(AgentExecutor):
    """Currency Agent using LangGraph for ReAct pattern."""

    def __init__(self):
        # self.llm = ChatOpenAI(
        #     model="gpt-4o-mini", 
        #     temperature=0,
        #     api_key=os.getenv('OPENAI_API_KEY')
        # )
        self.llm = LocalLLMChat(
        model=os.getenv("LOCAL_LLM_MODEL", "llama3.3"),
        api_key=os.getenv("LOCAL_LLM_API_KEY")
        )
        # Available tools
        self.tools = [get_exchange_rate, convert_currency, get_multiple_rates]
        
        # Create the graph
        self.graph = self._create_currency_graph()

    def _create_currency_graph(self) -> StateGraph:
        """Create the LangGraph for currency processing."""
        
        def should_continue(state: CurrencyState) -> str:
            """Decide whether to continue or end."""
            messages = state["messages"]
            last_message = messages[-1]
            
            # If the LLM makes a tool call, then we route to the "tools" node
            if last_message.tool_calls:
                return "tools"
            # Otherwise, we stop (reply to the user)
            return END

        def call_model(state: CurrencyState):
            """Call the LLM with the current state."""
            messages = state["messages"]
            
            # Add system message for currency agent behavior
            system_message = SystemMessage(content="""
            You are a helpful currency conversion agent. Your job is to provide accurate exchange rates and currency conversions.

            When users ask about currencies:
            1. Extract the currencies and amounts from their query
            2. Use appropriate tools to get real-time exchange rate data
            3. Present the information in a clear, user-friendly format
            4. Include the timestamp of the rates when available
            5. For conversions, show both the rate and the converted amount

            Available tools:
            - get_exchange_rate: Get rate between two specific currencies
            - convert_currency: Convert a specific amount from one currency to another
            - get_multiple_rates: Get rates from one base currency to multiple others

            Common currency codes: USD, EUR, GBP, JPY, AUD, CAD, CHF, CNY, INR, etc.

            If currency codes are not clear, ask for clarification or suggest common ones.
            Always use the tools to get real-time data rather than providing outdated information.
            """)
            
            full_messages = [system_message] + messages
            response = self.llm.invoke(full_messages)
            return {"messages": [response]}

        # Create the graph
        workflow = StateGraph(CurrencyState)
        
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
        """Execute the currency agent."""
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
                    [Part(root=TextPart(text="Processing currency request..."))]
                ),
            )
            
            # Initialize state
            initial_state = CurrencyState(
                messages=[HumanMessage(content=user_message)],
                from_currency="",
                to_currency="",
                amount=0.0,
                exchange_data={}
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
            logger.error(f"Error in currency agent execution: {e}")
            error_message = f"Sorry, I encountered an error while processing your currency request: {str(e)}"
            error_parts = [Part(root=TextPart(text=error_message))]
            updater.add_artifact(error_parts)
            updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=UnsupportedOperationError())

# currency_agent/currency_agent_executor.py
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
# class CurrencyState(TypedDict):
#     messages: Annotated[list, add_messages]
#     from_currency: str
#     to_currency: str
#     amount: float
#     exchange_data: Dict[str, Any]

# @tool
# def get_exchange_rate(from_currency: str, to_currency: str) -> str:
#     """Get current exchange rate between two currencies using exchangerate-api.com"""
#     try:
#         api_key = os.getenv('EXCHANGE_API_KEY')
#         if not api_key:
#             # Fallback to mock data for demonstration
#             mock_rates = {
#                 ('USD', 'INR'): 82.75,
#                 ('USD', 'EUR'): 0.85,
#                 ('USD', 'GBP'): 0.73,
#                 ('EUR', 'USD'): 1.18,
#                 ('GBP', 'USD'): 1.37,
#                 ('INR', 'USD'): 0.012
#             }
            
#             key = (from_currency.upper(), to_currency.upper())
#             rate = mock_rates.get(key, 1.0)
            
#             result = {
#                 'from_currency': from_currency.upper(),
#                 'to_currency': to_currency.upper(),
#                 'exchange_rate': rate,
#                 'timestamp': '2024-05-22T10:00:00Z',
#                 'note': 'Mock data - API key not configured'
#             }
#             return json.dumps(result)

#         base_url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_currency.upper()}/{to_currency.upper()}"

#         import requests
#         response = requests.get(base_url, timeout=10)

#         if response.status_code == 200:
#             data = response.json()
#             rate = data.get("conversion_rate")
#             result = {
#                 'from_currency': from_currency.upper(),
#                 'to_currency': to_currency.upper(),
#                 'exchange_rate': rate,
#                 'timestamp': data.get("time_last_update_utc")
#             }
#             return json.dumps(result)
#         else:
#             return f"Could not fetch exchange rate. Error: {response.status_code}"

#     except Exception as e:
#         logger.error(f"Error in get_exchange_rate: {e}")
#         return f"Error fetching exchange rate: {str(e)}"

# @tool
# def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
#     """Convert an amount from one currency to another using exchangerate-api.com"""
#     try:
#         rate_response = get_exchange_rate(from_currency, to_currency)
#         rate_data = json.loads(rate_response)

#         if 'exchange_rate' not in rate_data:
#             return rate_response

#         exchange_rate = rate_data['exchange_rate']
#         converted = round(amount * exchange_rate, 2)

#         return json.dumps({
#             'original_amount': amount,
#             'from_currency': from_currency.upper(),
#             'to_currency': to_currency.upper(),
#             'exchange_rate': exchange_rate,
#             'converted_amount': converted,
#             'timestamp': rate_data.get("timestamp")
#         })

#     except Exception as e:
#         logger.error(f"Error in convert_currency: {e}")
#         return f"Error converting currency: {str(e)}"

# @tool
# def get_multiple_rates(base_currency: str, target_currencies: List[str] = None) -> str:
#     """Get multiple exchange rates from a base currency"""
#     try:
#         if not target_currencies:
#             target_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'INR']

#         api_key = os.getenv('EXCHANGE_API_KEY')
        
#         if not api_key:
#             # Mock data for demonstration
#             mock_rates = {
#                 'USD': {'EUR': 0.85, 'GBP': 0.73, 'JPY': 110.0, 'INR': 82.75},
#                 'EUR': {'USD': 1.18, 'GBP': 0.86, 'JPY': 129.5, 'INR': 97.5},
#                 'INR': {'USD': 0.012, 'EUR': 0.010, 'GBP': 0.009, 'JPY': 1.33}
#             }
            
#             base_rates = mock_rates.get(base_currency.upper(), {})
#             filtered = {cur: base_rates.get(cur, 1.0) for cur in target_currencies if cur != base_currency.upper()}
            
#             return json.dumps({
#                 'base_currency': base_currency.upper(),
#                 'rates': filtered,
#                 'timestamp': '2024-05-22T10:00:00Z',
#                 'note': 'Mock data - API key not configured'
#             })

#         base_url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{base_currency.upper()}"

#         import requests
#         response = requests.get(base_url, timeout=10)

#         if response.status_code == 200:
#             data = response.json()
#             all_rates = data.get("conversion_rates", {})
#             filtered = {cur: all_rates[cur] for cur in target_currencies if cur in all_rates}

#             return json.dumps({
#                 'base_currency': base_currency.upper(),
#                 'rates': filtered,
#                 'timestamp': data.get("time_last_update_utc")
#             })
#         else:
#             return f"Could not fetch multiple exchange rates. Error: {response.status_code}"

#     except Exception as e:
#         logger.error(f"Error in get_multiple_rates: {e}")
#         return f"Error fetching rates: {str(e)}"


# class CurrencyAgentExecutor(AgentExecutor):
#     """Currency Agent using LangGraph for ReAct pattern."""

#     def __init__(self):
#         self.llm = LocalLLMChat(
#             model=os.getenv("LOCAL_LLM_MODEL", "llama3.3"),
#             api_key=os.getenv("LOCAL_LLM_API_KEY")
#         )
        
#         # Available tools
#         self.tools = [get_exchange_rate, convert_currency, get_multiple_rates]
        
#         # Bind tools to LLM for function calling
#         self.llm_with_tools = self.llm.bind_tools(self.tools)
        
#         # Create the graph
#         self.graph = self._create_currency_graph()

#     def _create_currency_graph(self) -> StateGraph:
#         """Create the LangGraph for currency processing."""
        
#         def should_continue(state: CurrencyState) -> str:
#             """Decide whether to continue or end."""
#             messages = state["messages"]
#             last_message = messages[-1]
            
#             # If the LLM makes a tool call, then we route to the "tools" node
#             if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
#                 return "tools"
#             # Otherwise, we stop (reply to the user)
#             return END

#         def call_model(state: CurrencyState):
#             """Call the LLM with the current state."""
#             messages = state["messages"]
            
#             # Add system message for currency agent behavior
#             system_message = SystemMessage(content="""
#             You are a helpful currency conversion agent. Your job is to provide accurate exchange rates and currency conversions.

#             When users ask about currencies:
#             1. Extract the currencies and amounts from their query
#             2. Use appropriate tools to get real-time exchange rate data
#             3. Present the information in a clear, user-friendly format
#             4. Include the timestamp of the rates when available
#             5. For conversions, show both the rate and the converted amount

#             Available tools:
#             - get_exchange_rate: Get rate between two specific currencies
#             - convert_currency: Convert a specific amount from one currency to another
#             - get_multiple_rates: Get rates from one base currency to multiple others

#             Common currency codes: USD, EUR, GBP, JPY, AUD, CAD, CHF, CNY, INR, etc.

#             If currency codes are not clear, ask for clarification or suggest common ones.
#             Always use the tools to get real-time data rather than providing outdated information.
#             """)
            
#             full_messages = [system_message] + messages
#             response = self.llm_with_tools.invoke(full_messages)
#             return {"messages": [response]}

#         # Create the graph
#         workflow = StateGraph(CurrencyState)
        
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
#         """Execute the currency agent."""
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
            
#             logger.info(f"Processing currency request: {user_message}")
            
#             # Update status
#             updater.update_status(
#                 TaskState.working,
#                 message=updater.new_agent_message(
#                     [Part(root=TextPart(text="Processing currency request..."))]
#                 ),
#             )
            
#             # Initialize state
#             initial_state = CurrencyState(
#                 messages=[HumanMessage(content=user_message)],
#                 from_currency="",
#                 to_currency="",
#                 amount=0.0,
#                 exchange_data={}
#             )
            
#             # Run the graph
#             final_state = await asyncio.to_thread(
#                 self.graph.invoke, 
#                 initial_state
#             )
            
#             # Get the final response
#             final_message = final_state["messages"][-1]
#             response_text = final_message.content
            
#             logger.info(f"Currency agent response: {response_text}")
            
#             # Create artifact with the response
#             response_parts = [Part(root=TextPart(text=response_text))]
#             updater.add_artifact(response_parts)
#             updater.complete()
            
#         except Exception as e:
#             logger.error(f"Error in currency agent execution: {e}", exc_info=True)
#             error_message = f"Sorry, I encountered an error while processing your currency request: {str(e)}"
#             error_parts = [Part(root=TextPart(text=error_message))]
#             updater.add_artifact(error_parts)
#             updater.complete()

#     async def cancel(self, context: RequestContext, event_queue: EventQueue):
#         raise ServerError(error=UnsupportedOperationError())