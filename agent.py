import logging
import asyncio
import os
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Load environment variables (expecting GOOGLE_API_KEY)
load_dotenv()

APP_NAME = "reminder_app"
USER_ID = "user_default"
SESSION_ID = "session_default"
MODEL_NAME = "gemini-2.0-flash"

from google.adk.tools import ToolContext

def add_reminder(tool_context: ToolContext, reminder_text: str) -> str:
    """
    Adds a reminder to the user's list.
    
    Args:
        tool_context: The ADK tool context, providing access to session state.
        reminder_text: The content of the reminder to add.
    """
    # Access session state from tool_context
    # Note: tool_context.state might be a dict-like interface to the session state
    state = tool_context.state
    reminders = state.get("reminders", [])
    if not isinstance(reminders, list):
        reminders = []
    
    reminders.append(reminder_text)
    
    state["reminders"] = reminders
    
    return f"Added reminder: '{reminder_text}'"

def get_reminders(tool_context: ToolContext) -> str:
    """
    Retrieves the list of current reminders.
    
    Args:
        tool_context: The ADK tool context, providing access to session state.
    """
    state = tool_context.state
    reminders = state.get("reminders", [])
    
    if not reminders:
        return "You have no reminders set."
    
    response = "Here are your reminders:\n"
    for i, r in enumerate(reminders, 1):
        response += f"{i}. {r}\n"
    return response

async def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return

    # 1. Setup Session Service (In-Memory for this session only)
    session_service = InMemorySessionService()
    
    # 2. Define the Agent
    agent = LlmAgent(
        model=MODEL_NAME,
        name="reminder_agent_with_AG",
        instruction="""You are a helpful reminder assistant. 
        You have tools to add reminders and list existing reminders.
        Always use the 'add_reminder' tool when the user asks to remember something.
        Always use the 'get_reminders' tool when the user asks what they have to do or what's on the list.
        If the user asks to delete, apologize and say that feature is not implemented yet.
        """,
        tools=[add_reminder, get_reminders]
    )

    # 3. Create Runner
    # We initialize the runner with our agent and session service.
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    # Explicitly create the session
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    print(f"Agent '{agent.name}' started. Type 'quit' to exit.")
    print("--------------------------------------------------")

    # 4. Interactive Loop
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            # Create a user content message structure for the ADK
            user_msg = types.Content(
                role="user",
                parts=[types.Part(text=user_input)]
            )

            # Run the agent asynchronously for this turn
            # returns an AsyncGenerator of events
            events = runner.run_async(
                user_id=USER_ID,
                session_id=SESSION_ID,
                new_message=user_msg
            )

            # Iterate through events to find the final response
            async for event in events:
                if event.is_final_response():
                    # The content is in event.content which is a google.genai.types.Content object
                    agent_text = event.content.parts[0].text
                    print(f"Agent: {agent_text}")
                    
    finally:
        # Cleanup
        await runner.close()

if __name__ == "__main__":
    asyncio.run(main())
