# currency_agent/__main__.py
import asyncio
import logging
import os

import click
import uvicorn
from dotenv import load_dotenv

from currency_agent_executor import CurrencyAgentExecutor
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
@click.option('--port', 'port', default=2000)
def main(host: str, port: int):
    # Verify API keys
    if not os.getenv('LOCAL_LLM_API_KEY'):
        raise ValueError('LOCAL_LLM_API_KEY environment variable must be set')
    
    if not os.getenv('EXCHANGE_API_KEY'):
        print("Warning: EXCHANGE_API_KEY not set. Using free tier with limitations.")

    skill = AgentSkill(
        id='currency_conversion',
        name='Currency Conversion',
        description='Convert currencies and provide exchange rate information',
        tags=['currency', 'finance', 'exchange'],
        examples=[
            'Convert 100 USD to EUR',
            'What is the exchange rate for GBP to JPY?',
            'How much is 50 euros in dollars?',
            'Show me rates for USD to multiple currencies'
        ],
    )

    agent_executor = CurrencyAgentExecutor()
    
    agent_card = AgentCard(
        name='Currency Agent',
        description='I provide real-time currency conversion and exchange rate information.',
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
    print(f"Starting Currency Agent on {host}:{port}")
    uvicorn.run(app.build(), host="0.0.0.0", port=port)

if __name__ == '__main__':
    main()