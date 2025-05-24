# # ## taskid not found

# # # cli_interface.py
# # import asyncio
# # import uuid
# # from typing import Dict, Any
# # import httpx
# # from a2a.client import A2AClient
# # from a2a.types import (
# #     Message,
# #     MessageSendParams,
# #     Part,
# #     Role,
# #     SendMessageRequest,
# #     SendMessageSuccessResponse,
# #     Task,
# #     TextPart
# # )

# # async def send_query_to_client_agent(query: str, client_agent_url: str = "http://localhost:4004"):
# #     """Send a query to the client agent and await the response."""
# #     print(f"\nProcessing: '{query}'...")
# #     print(f"Connecting to: {client_agent_url}")
    
# #     # Generate unique IDs
# #     context_id = f"cli-{uuid.uuid4()}"
# #     message_id = f"msg-{uuid.uuid4()}"
    
# #     # Create message request
# #     request = SendMessageRequest(
# #         params=MessageSendParams(
# #             message=Message(
# #                 contextId=context_id,
# #                 taskId=None,
# #                 messageId=message_id,
# #                 role=Role.user,
# #                 parts=[Part(TextPart(text=query))],
# #             )
# #         )
# #     )
    
# #     # Send request to client agent with timeout and better error handling
# #     try:
# #         async with httpx.AsyncClient(timeout=30.0) as client:
# #             agent_client = A2AClient(httpx_client=client, url=client_agent_url)
# #             response = await agent_client.send_message(request)
# #     except httpx.ConnectError as e:
# #         raise Exception(f"Connection failed to {client_agent_url}. Is the client agent running? Error: {str(e)}")
# #     except httpx.TimeoutException as e:
# #         raise Exception(f"Request timed out to {client_agent_url}. Error: {str(e)}")
# #     except Exception as e:
# #         raise Exception(f"Unexpected error connecting to {client_agent_url}: {str(e)}")
    
# #     # Process response
# #     result_text = ""
# #     if hasattr(response.root, "result"):
# #         result = response.root.result
# #         if isinstance(result, Task):
# #             # Wait for task to complete and get response
# #             task_id = result.taskId
# #             task_status = result.status.state
            
# #             # If task has artifacts already
# #             if result.artifacts:
# #                 for artifact in result.artifacts:
# #                     for part in artifact.parts:
# #                         if hasattr(part.root, "text"):
# #                             result_text += part.root.text + "\n"
# #             else:
# #                 # If task has a message in status
# #                 for part in result.status.message.parts:
# #                     if hasattr(part.root, "text"):
# #                         result_text += part.root.text + "\n"
# #         else:
# #             # Direct message response
# #             for part in result.parts:
# #                 if hasattr(part.root, "text"):
# #                     result_text += part.root.text + "\n"
    
# #     return result_text.strip()

# # async def test_connection(url: str = "http://localhost:4004"):
# #     """Test if the client agent is reachable."""
# #     try:
# #         async with httpx.AsyncClient(timeout=5.0) as client:
# #             response = await client.get(f"{url}/health")  # or whatever health endpoint exists
# #             return True
# #     except:
# #         return False

# # async def main():
# #     print("\n===== MULTI-AGENT LANGGRAPH CLI =====")
# #     print("Type 'exit' or 'quit' to end the session.")
    
# #     # Test connection first
# #     client_url = "http://localhost:4004"
# #     print(f"Testing connection to client agent at {client_url}...")
    
# #     if not await test_connection(client_url):
# #         print(f"⚠️  Warning: Cannot reach client agent at {client_url}")
# #         print("Make sure your client agent is running on port 4004")
        
# #         # Allow user to specify different URL
# #         custom_url = input("Enter client agent URL (or press Enter to continue anyway): ").strip()
# #         if custom_url:
# #             client_url = custom_url
# #     else:
# #         print("✅ Client agent is reachable")
    
# #     while True:
# #         query = input("\nEnter your query: ")
        
# #         if query.lower() in ["exit", "quit"]:
# #             print("Goodbye!")
# #             break
        
# #         if not query.strip():
# #             continue
            
# #         try:
# #             result = await send_query_to_client_agent(query, client_url)
# #             print("\n" + "=" * 40)
# #             print("RESPONSE:")
# #             print(result)
# #             print("=" * 40)
# #         except Exception as e:
# #             print(f"Error: {str(e)}")
# #             print("\n💡 Troubleshooting tips:")
# #             print("1. Check if client agent is running: netstat -an | findstr :4004")
# #             print("2. Try different URL: http://127.0.0.1:4004")
# #             print("3. Check client agent logs for errors")

# # if __name__ == "__main__":
# #     asyncio.run(main())



# ## working wit some issues.....

# # cli_interface.py
# import asyncio
# import uuid
# from typing import Dict, Any
# import httpx
# from a2a.client import A2AClient
# from a2a.types import (
#     Message,
#     MessageSendParams,
#     Part,
#     Role,
#     SendMessageRequest,
#     SendMessageSuccessResponse,
#     Task,
#     TextPart
# )

# async def send_query_to_client_agent(query: str, client_agent_url: str = "http://localhost:4004"):
#     """Send a query to the client agent and await the response."""
#     print(f"\nProcessing: '{query}'...")
#     print(f"Connecting to: {client_agent_url}")
    
#     # Generate unique IDs
#     context_id = f"cli-{uuid.uuid4()}"
#     message_id = f"msg-{uuid.uuid4()}"
    
#     # Create message request
#     request = SendMessageRequest(
#         params=MessageSendParams(
#             message=Message(
#                 contextId=context_id,
#                 taskId=None,
#                 messageId=message_id,
#                 role=Role.user,
#                 parts=[Part(TextPart(text=query))],
#             )
#         )
#     )
    
#     # Send request to client agent with timeout and better error handling
#     try:
#         async with httpx.AsyncClient(timeout=30.0) as client:
#             agent_client = A2AClient(httpx_client=client, url=client_agent_url)
#             response = await agent_client.send_message(request)
#     except httpx.ConnectError as e:
#         raise Exception(f"Connection failed to {client_agent_url}. Is the client agent running? Error: {str(e)}")
#     except httpx.TimeoutException as e:
#         raise Exception(f"Request timed out to {client_agent_url}. Error: {str(e)}")
#     except Exception as e:
#         raise Exception(f"Unexpected error connecting to {client_agent_url}: {str(e)}")
    
#     # Process response
#     result_text = ""
#     if hasattr(response.root, "result"):
#         result = response.root.result
#         if isinstance(result, Task):
#             print(f"Debug: Task object attributes: {dir(result)}")
            
#             # Try different possible attribute names for task ID
#             task_id = None
#             for attr_name in ['taskId', 'task_id', 'id', 'task_identifier']:
#                 if hasattr(result, attr_name):
#                     task_id = getattr(result, attr_name)
#                     print(f"Debug: Found task ID as '{attr_name}': {task_id}")
#                     break
            
#             # Try to get task status
#             task_status = None
#             if hasattr(result, 'status'):
#                 if hasattr(result.status, 'state'):
#                     task_status = result.status.state
#                     print(f"Debug: Task status: {task_status}")
            
#             # Try to extract text from various possible locations
#             extracted = False
            
#             # Try artifacts first
#             if hasattr(result, 'artifacts') and result.artifacts:
#                 print("Debug: Processing artifacts")
#                 for artifact in result.artifacts:
#                     if hasattr(artifact, 'parts'):
#                         for part in artifact.parts:
#                             if hasattr(part, 'root') and hasattr(part.root, "text"):
#                                 result_text += part.root.text + "\n"
#                                 extracted = True
#                             elif hasattr(part, "text"):
#                                 result_text += part.text + "\n"
#                                 extracted = True
            
#             # Try status message
#             if not extracted and hasattr(result, 'status') and hasattr(result.status, 'message'):
#                 print("Debug: Processing status message")
#                 if hasattr(result.status.message, 'parts'):
#                     for part in result.status.message.parts:
#                         if hasattr(part, 'root') and hasattr(part.root, "text"):
#                             result_text += part.root.text + "\n"
#                             extracted = True
#                         elif hasattr(part, "text"):
#                             result_text += part.text + "\n"
#                             extracted = True
            
#             # If still no text, try to get any string representation
#             if not extracted:
#                 result_text = f"Task created with ID: {task_id}, Status: {task_status}"
#                 print(f"Debug: Full result object: {result}")
                
#         else:
#             print("Debug: Processing direct message response")
#             # Direct message response
#             if hasattr(result, 'parts'):
#                 for part in result.parts:
#                     if hasattr(part, 'root') and hasattr(part.root, "text"):
#                         result_text += part.root.text + "\n"
#                     elif hasattr(part, "text"):
#                         result_text += part.text + "\n"
#             else:
#                 result_text = str(result)
    
#     return result_text.strip()

# async def test_connection(url: str = "http://localhost:4004"):
#     """Test if the client agent is reachable."""
#     try:
#         async with httpx.AsyncClient(timeout=5.0) as client:
#             response = await client.get(f"{url}/health")  # or whatever health endpoint exists
#             return True
#     except:
#         return False

# async def main():
#     print("\n===== MULTI-AGENT LANGGRAPH CLI =====")
#     print("Type 'exit' or 'quit' to end the session.")
    
#     # Test connection first
#     client_url = "http://localhost:4004"
#     print(f"Testing connection to client agent at {client_url}...")
    
#     if not await test_connection(client_url):
#         print(f"⚠️  Warning: Cannot reach client agent at {client_url}")
#         print("Make sure your client agent is running on port 4004")
        
#         # Allow user to specify different URL
#         custom_url = input("Enter client agent URL (or press Enter to continue anyway): ").strip()
#         if custom_url:
#             client_url = custom_url
#     else:
#         print("✅ Client agent is reachable")
    
#     while True:
#         query = input("\nEnter your query: ")
        
#         if query.lower() in ["exit", "quit"]:
#             print("Goodbye!")
#             break
        
#         if not query.strip():
#             continue
            
#         try:
#             result = await send_query_to_client_agent(query, client_url)
#             print("\n" + "=" * 40)
#             print("RESPONSE:")
#             print(result)
#             print("=" * 40)
#         except Exception as e:
#             print(f"Error: {str(e)}")
#             print("\n💡 Troubleshooting tips:")
#             print("1. Check if client agent is running: netstat -an | findstr :4004")
#             print("2. Try different URL: http://127.0.0.1:4004")
#             print("3. Check client agent logs for errors")

# if __name__ == "__main__":
#     asyncio.run(main())

# cli_interface.py
import asyncio
import uuid
from typing import Dict, Any
import httpx
from a2a.client import A2AClient
from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    SendMessageSuccessResponse,
    Task,
    TextPart
)

async def send_query_to_client_agent(query: str, client_agent_url: str = "http://localhost:4004"):
    """Send a query to the client agent and await the response."""
    print(f"\nProcessing: '{query}'...")
    print(f"Connecting to: {client_agent_url}")
    
    # Generate unique IDs
    context_id = f"cli-{uuid.uuid4()}"
    message_id = f"msg-{uuid.uuid4()}"
    
    # Create message request
    request = SendMessageRequest(
        params=MessageSendParams(
            message=Message(
                contextId=context_id,
                taskId=None,
                messageId=message_id,
                role=Role.user,
                parts=[Part(TextPart(text=query))],
            )
        )
    )
    
    # Send request to client agent with timeout and better error handling
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            agent_client = A2AClient(httpx_client=client, url=client_agent_url)
            response = await agent_client.send_message(request)
    except httpx.ConnectError as e:
        raise Exception(f"Connection failed to {client_agent_url}. Is the client agent running? Error: {str(e)}")
    except httpx.TimeoutException as e:
        raise Exception(f"Request timed out to {client_agent_url}. Error: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error connecting to {client_agent_url}: {str(e)}")
    
    # Process response
    result_text = ""
    if hasattr(response.root, "result"):
        result = response.root.result
        if isinstance(result, Task):
            # print(f"Debug: Task object attributes: {dir(result)}")
            
            # Try different possible attribute names for task ID
            task_id = None
            for attr_name in ['taskId', 'task_id', 'id', 'task_identifier']:
                if hasattr(result, attr_name):
                    task_id = getattr(result, attr_name)
                    print(f"Debug: Found task ID as '{attr_name}': {task_id}")
                    break
            
            # Try to get task status
            task_status = None
            if hasattr(result, 'status'):
                if hasattr(result.status, 'state'):
                    task_status = result.status.state
                    print(f"Debug: Task status: {task_status}")
            
            # Try to extract text from various possible locations
            extracted = False
            
            # Try artifacts first
            if hasattr(result, 'artifacts') and result.artifacts:
                # print("Debug: Processing artifacts")
                for artifact in result.artifacts:
                    # print(f"Debug: Artifact type: {type(artifact)}")
                    # print(f"Debug: Artifact attributes: {dir(artifact)}")
                    
                    if hasattr(artifact, 'parts'):
                        # print(f"Debug: Found {len(artifact.parts)} parts in artifact")
                        for i, part in enumerate(artifact.parts):
                            # print(f"Debug: Part {i} type: {type(part)}")
                            # print(f"Debug: Part {i} attributes: {dir(part)}")
                            
                            # Try different ways to access text
                            text_found = False
                            if hasattr(part, 'root') and hasattr(part.root, "text"):
                                result_text += part.root.text + "\n"
                                extracted = True
                                text_found = True
                                # print(f"Debug: Found text in part.root.text")
                            elif hasattr(part, "text"):
                                result_text += part.text + "\n"
                                extracted = True
                                text_found = True
                                # print(f"Debug: Found text in part.text")
                            elif hasattr(part, 'content'):
                                result_text += str(part.content) + "\n"
                                extracted = True
                                text_found = True
                                # print(f"Debug: Found text in part.content")
                            
                            if not text_found:
                                # print(f"Debug: No text found in part {i}, trying string conversion")
                                result_text += str(part) + "\n"
                                extracted = True
                    else:
                        # print(f"Debug: Artifact has no parts, trying direct conversion")
                        result_text += str(artifact) + "\n"
                        extracted = True
            
            # Try status message
            if not extracted and hasattr(result, 'status') and hasattr(result.status, 'message'):
                print("Debug: Processing status message")
                if hasattr(result.status.message, 'parts'):
                    for part in result.status.message.parts:
                        if hasattr(part, 'root') and hasattr(part.root, "text"):
                            result_text += part.root.text + "\n"
                            extracted = True
                        elif hasattr(part, "text"):
                            result_text += part.text + "\n"
                            extracted = True
            
            # If still no text, try to get any string representation
            if not extracted:
                result_text = f"Task created with ID: {task_id}, Status: {task_status}"
                # print(f"Debug: Full result object: {result}")
                
        else:
            # print("Debug: Processing direct message response")
            # Direct message response
            if hasattr(result, 'parts'):
                for part in result.parts:
                    if hasattr(part, 'root') and hasattr(part.root, "text"):
                        result_text += part.root.text + "\n"
                    elif hasattr(part, "text"):
                        result_text += part.text + "\n"
            else:
                result_text = str(result)
    
    return result_text.strip()

async def test_connection(url: str = "http://localhost:4004"):
    """Test if the client agent is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}/health")  # or whatever health endpoint exists
            return True
    except:
        return False

async def main():
    print("\n===== MULTI-AGENT LANGGRAPH CLI =====")
    print("Type 'exit' or 'quit' to end the session.")
    
    # Test connection first
    client_url = "http://localhost:4004"
    print(f"Testing connection to client agent at {client_url}...")
    
    if not await test_connection(client_url):
        print(f"⚠️  Warning: Cannot reach client agent at {client_url}")
        print("Make sure your client agent is running on port 4004")
        
        # Allow user to specify different URL
        custom_url = input("Enter client agent URL (or press Enter to continue anyway): ").strip()
        if custom_url:
            client_url = custom_url
    else:
        print("✅ Client agent is reachable")
    
    while True:
        query = input("\nEnter your query: ")
        
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        if not query.strip():
            continue
            
        try:
            result = await send_query_to_client_agent(query, client_url)
            print("\n" + "=" * 40)
            print("RESPONSE:")
            print(result)
            print("=" * 40)
        except Exception as e:
            print(f"Error: {str(e)}")
            print("\n💡 Troubleshooting tips:")
            print("1. Check if client agent is running: netstat -an | findstr :4004")
            print("2. Try different URL: http://127.0.0.1:4004")
            print("3. Check client agent logs for errors")

if __name__ == "__main__":
    asyncio.run(main())