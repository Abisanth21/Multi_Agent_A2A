import asyncio
import logging
import os
from collections.abc import AsyncIterable
from typing import Any, Dict
from uuid import uuid4

import httpx
from pydantic import ConfigDict
from langchain_core.messages import HumanMessage, SystemMessage

from local_llm_wrapper import LocalLLMChat

from a2a.client import A2AClient
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Artifact,
    FilePart,
    FileWithBytes,
    FileWithUri,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    SendMessageSuccessResponse,
    Task,
    # taskId,
    TaskState,
    TaskStatus,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import get_text_parts
from a2a.utils.errors import ServerError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TASK_POLLING_DELAY_SECONDS = 0.2

class ClientAgentExecutor(AgentExecutor):
    """Client Agent that routes queries to specialized agents using local LLM."""

    def __init__(self, available_agents: Dict[str, Dict[str, Any]]):
        self.available_agents = available_agents

        # Build instruction with agent information
        agent_descriptions = []
        for agent_name, agent_info in available_agents.items():
            keywords = ", ".join(agent_info['keywords'])
            agent_descriptions.append(
                f"- {agent_name.title()} Agent: {agent_info['description']} (Keywords: {keywords})"
            )

        self.instruction = f"""
        You are a Client Agent that routes user queries to specialized agents.

        Available Agents:
        {chr(10).join(agent_descriptions)}
        
        Your job is to:
        1. Analyze the user's query to understand what they need
        2. Determine which specialized agent can best handle the request
        3. Route the query to the appropriate agent using the available tools
        4. Provide the response back to the user

        If a query involves multiple domains (e.g., "What's the weather in Japan and convert 100 USD to JPY"), 
        you can call multiple agents sequentially.

        If the query is not related to any available agents, politely explain what you can help with.
        """

        self.llm = LocalLLMChat(
            model=os.getenv("LOCAL_LLM_MODEL", "llama3.3"),
            api_key=os.getenv("LOCAL_LLM_API_KEY")
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        if not context.current_task:
            updater.submit()
        updater.start_work()

        # Extract the message text
        user_message = ""
        for part in context.message.parts:
            if isinstance(part.root, TextPart):
                user_message += part.root.text

        updater.update_status(
            TaskState.working,
            message=updater.new_agent_message(
                [Part(root=TextPart(text="Analyzing query with LLM..."))]
            ),
        )

        # Route the query
        result = await self._route_query(user_message, context, updater)

        updater.add_artifact([Part(root=TextPart(text=result['response']))])
        updater.complete()

    async def _route_query(self, user_query: str, context: RequestContext, updater: TaskUpdater) -> Dict:
        messages = [
            SystemMessage(content=self.instruction),
            HumanMessage(content=user_query)
        ]
        response_msg = self.llm.invoke(messages)
        response_text = response_msg.content.lower()

        logger.debug(f"LLM response for routing: {response_text}")
        
        if any(k in response_text for k in ["weather", "temperature", "rain", "forecast", "climate"]):
            logger.info("Routing to weather agent")
            return await self.call_weather_agent(user_query, context)
        elif any(k in response_text for k in ["currency", "convert", "exchange", "dollar", "euro", "yen", "rate"]):
            logger.info("Routing to currency agent")
            return await self.call_currency_agent(user_query, context)
        else:
            return {"response": "I'm not sure how to help with that. I can assist with weather or currency conversions."}

    async def call_weather_agent(self, query: str, context: RequestContext):
        return await self._call_specialized_agent('weather', query, context)

    async def call_currency_agent(self, query: str, context: RequestContext):
        return await self._call_specialized_agent('currency', query, context)

    # async def _call_specialized_agent(self, agent_name: str, query: str, context: RequestContext):
    #     agent_info = self.available_agents[agent_name]

    #     # Ensure URL has correct format with v1/messages endpoint
    #     agent_url = agent_info['url']
    #     if not agent_url.endswith('/'):
    #         agent_url += '/'
    #     if not agent_url.endswith('v1/messages'):
    #         agent_url += 'v1/messages'
            
    #     print(f"Calling {agent_name} agent at {agent_url}")

    #     request = SendMessageRequest(
    #         params=MessageSendParams(
    #             message=Message(
    #                 contextId=context.context_id,
    #                 taskId=None,
    #                 messageId=str(uuid4()),
    #                 role=Role.user,
    #                 parts=[Part(root=TextPart(text=query))],
    #             )
    #         )
    #     )

    #     try:
    #         async with httpx.AsyncClient(timeout=30.0) as client:
    #             agent_client = A2AClient(httpx_client=client, url=agent_url)
    #             response = await agent_client.send_message(request)
    #             print(f"Response received from {agent_name} agent")
    #     except Exception as e:
    #         print(f"Error calling {agent_name} agent: {e}")
    #         return {"response": f"Error communicating with {agent_name} agent: {str(e)}"}

    #     content = []
    #     if isinstance(response.root, SendMessageSuccessResponse):
    #         result = response.root.result
    #         if isinstance(result, Task):
    #             if result.artifacts:
    #                 for artifact in result.artifacts:
    #                     content.extend(get_text_parts(artifact.parts))
    #             if not content:
    #                 content.extend(get_text_parts(result.status.message.parts))
    #         else:
    #             content.extend(get_text_parts(result.parts))

    #     return {"response": "\n".join(content) if content else f"No response from {agent_name} agent"}
    # async def _call_specialized_agent(self, agent_name: str, query: str, context: RequestContext):

    #     agent_info = self.available_agents[agent_name]

    # # For A2A, check your package's documentation for the correct endpoint path
    # # You might need to inspect the actual endpoint patterns using a tool like Swagger
    #     agent_url = agent_info['url']
    #     if not agent_url.endswith('/'):
    #         agent_url += '/'
    
    # # Print URLs for debugging
    #     print(f"Base URL for {agent_name} agent: {agent_url}")
    
    # # Try with different endpoint patterns (uncomment one at a time for testing)
    # # full_url = f"{agent_url}v1/messages"
    #     # full_url = f"{agent_url}messages"
    #     # full_url = f"{agent_url}messages"
    #     full_url = agent_url
 
    # # full_url = agent_url  # Try the base URL directly
    
    #     print(f"Full request URL: {full_url}")
    
    #     request = SendMessageRequest(
    #         params=MessageSendParams(
    #             message=Message(
    #                 contextId=context.context_id,
    #                 taskId=None,
    #                 messageId=str(uuid4()),
    #                 role=Role.user,
    #                 parts=[Part(root=TextPart(text=query))],
    #             )
    #         )
    #     )

    #     try:
    #         async with httpx.AsyncClient(timeout=60.0) as client:
    #             agent_client = A2AClient(httpx_client=client, url=full_url)
    #             response = await agent_client.send_message(request)
    #     except Exception as e:
    #         print(f"Error calling {agent_name} agent: {e}")
    #         return {"response": f"Error communicating with {agent_name} agent: {str(e)}"}

    # Rest of the code remains the same...
    async def _call_specialized_agent(self, agent_name: str, query: str, context: RequestContext):
        agent_info = self.available_agents[agent_name]
        agent_url = agent_info['url'].rstrip('/') + '/'  # normalize
        full_url = agent_url  # we know the JSON-RPC endpoint is at the base URL

        print(f"Calling {agent_name} agent at {full_url}")

        request = SendMessageRequest(
            params=MessageSendParams(
                message=Message(
                    contextId=context.context_id,
                    taskId=None,
                    messageId=str(uuid4()),
                    role=Role.user,
                    parts=[Part(root=TextPart(text=query))],
                )
            )
        )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                agent_client = A2AClient(httpx_client=client, url=full_url)
                response = await agent_client.send_message(request)
        except Exception as e:
            return {"response": f"Error communicating with {agent_name} agent: {e}"}

        # —— now parse out the text parts and return! ——
        content_lines = []
        if isinstance(response.root, SendMessageSuccessResponse):
            res = response.root.result
            # handle both Task and direct message cases
            artifacts = getattr(res, "artifacts", None)
            if artifacts:
                for art in artifacts:
                    for part in art.parts:
                        if hasattr(part.root, "text"):
                            content_lines.append(part.root.text)
            else:
                msg = getattr(res, "status", None)
                if msg:
                    for part in msg.message.parts:
                        content_lines.append(part.root.text)
                else:
                    # fallback on direct parts
                    for part in getattr(res, "parts", []):
                        content_lines.append(part.root.text)
        else:
            # non-success response
            content_lines.append(str(response.root))

        return {"response": "\n".join(content_lines) or f"No response from {agent_name} agent"}

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=UnsupportedOperationError())

