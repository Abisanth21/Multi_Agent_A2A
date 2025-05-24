# weather_agent/__main__.py
import asyncio
import logging
import os

import click
import uvicorn
from dotenv import load_dotenv

from weather_agent_executor import WeatherAgentExecutor
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
@click.option('--host', 'host', default='0.0.0.0')
@click.option('--port', 'port', default=3000)
def main(host: str, port: int):
    # Verify API keys
    if not os.getenv('LOCAL_LLM_API_KEY'):
        raise ValueError('LOCAL_LLM_API_KEY environment variable must be set')
    
    if not os.getenv('WEATHER_API_KEY'):
        print("Warning: WEATHER_API_KEY not set. Using free tier with limitations.")

    skill = AgentSkill(
        id='weather_info',
        name='Weather Information',
        description='Provide weather information, forecasts, and climate data for any location',
        tags=['weather', 'forecast', 'climate'],
        examples=[
            'What is the weather like in New York?',
            'Will it rain tomorrow in London?',
            'What is the temperature in Tokyo?',
            'Give me the 5-day forecast for San Francisco'
        ],
    )

    agent_executor = WeatherAgentExecutor()
    
    agent_card = AgentCard(
        name='Weather Agent',
        description='I provide accurate weather information and forecasts for any location worldwide.',
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        authentication={"schemes": ['public']},
    )
    
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, 
        task_store=InMemoryTaskStore()
    )
    # app = A2AStarletteApplication(agent_card, request_handler, base_path="")
    app = A2AStarletteApplication(agent_card, request_handler)
    print(f"Starting Weather Agent on {host}:{port}")
    uvicorn.run(app.build(), host="0.0.0.0", port=port)

if __name__ == '__main__':
    main()