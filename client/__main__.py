# client_agent/__main__.py
import asyncio
import functools
import logging
import os
from typing import Dict, List

import click
import uvicorn
from dotenv import load_dotenv

from client_agent_executor import ClientAgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=4004)
@click.option('--weather-agent', 'weather_agent', default='http://localhost:3000')
@click.option('--currency-agent', 'currency_agent', default='http://localhost:2000')
def main(host: str, port: int, weather_agent: str, currency_agent: str):
    # Verify API key for LLM
    if not os.getenv('LOCAL_LLM_API_KEY'):
        raise ValueError('LOCAL_LLM_API_KEY must be set')

    # Define available agents
    available_agents = {
        'weather': {
            'url': weather_agent,  # Base URL without trailing slash
            'description': 'Handles weather-related queries, forecasts, and conditions',
            'keywords': ['weather', 'temperature', 'rain', 'forecast', 'climate', 'sunny', 'cloudy']
        },
        'currency': {
            'url': currency_agent,  # Base URL without trailing slash
            'description': 'Handles currency conversion and exchange rates',
            'keywords': ['currency', 'exchange', 'convert', 'rate', 'dollar', 'euro', 'exchange rate']
        }
    }

    # Define skills
    skills = [
        AgentSkill(
            id='weather_queries',
            name='Weather Information',
            description='Get weather information, forecasts, and climate data',
            tags=['weather', 'forecast'],
            examples=[
                'What is the weather like in New York?',
                'Will it rain tomorrow in London?',
                'What is the temperature in Tokyo?'
            ],
        ),
        AgentSkill(
            id='currency_conversion',
            name='Currency Conversion',
            description='Convert currencies and get exchange rates',
            tags=['currency', 'finance'],
            examples=[
                'Convert 100 USD to EUR',
                'What is the exchange rate for GBP to JPY?',
                'How much is 50 euros in dollars?'
            ],
        ),
    ]

    # Create the agent executor
    agent_executor = ClientAgentExecutor(available_agents)
    
    # Create agent card
    agent_card = AgentCard(
        name='Multi-Agent Client',
        description='I can help you with weather information and currency conversion by routing your queries to specialized agents.',
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=skills,
        authentication={"schemes": ['public']},
    )
    
    # Set up the server
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, 
        task_store=InMemoryTaskStore()
    )
    # app = A2AStarletteApplication(agent_card, request_handler, base_path="")
    app = A2AStarletteApplication(agent_card, request_handler)
    print(f"Starting Client Agent on {host}:{port}")
    print(f"Weather Agent: {weather_agent}")
    print(f"Currency Agent: {currency_agent}")
    
    uvicorn.run(app.build(), host="0.0.0.0", port=port)

if __name__ == '__main__':
    main()