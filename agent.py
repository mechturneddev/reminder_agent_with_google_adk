import logging
import asyncio
import os
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Load environment variables (expecting GOOGLE_API_KEY)
load_dotenv()

APP_NAME = "reminder_app"
USER_ID = "user_default"
SESSION_ID = "session_default_v2" # Updated session ID for v2
MODEL_NAME = "gemini-2.0-flash"
DB_URL = "sqlite+aiosqlite:///reminders.db"

from google.adk.tools import ToolContext

def add_reminder(tool_context: ToolContext, reminder_text: str) -> str:
    """
    Adds a reminder to the user's list.
    
    Args:
        tool_context: The ADK tool context, providing access to session state.
        reminder_text: The content of the reminder to add.
    """
    # Access session state from tool_context
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

    # 1. Setup Session Service (Persistent via SQLite)
    session_service = DatabaseSessionService(db_url=DB_URL)
    
    # 2. Define the Agent
    agent = LlmAgent(
        model=MODEL_NAME,
        name="reminder_agent_with",
        instruction="""You are a helpful reminder assistant. 
        You have tools to add reminders and list existing reminders.
        Always use the 'add_reminder' tool when the user asks to remember something.
        Always use the 'get_reminders' tool when the user asks what they have to do or what's on the list.
        If the user asks to delete, apologize and say that feature is not implemented yet.
        """,
        tools=[add_reminder, get_reminders]
    )

    # 3. Create Runner
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    # Explicitly create or get the existing session (creates if not exists)
    # Explicitly create or get the existing session (creates if not exists)
    try:
        # Note: get_session returns None if not found, instead of raising ValueError
        session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
        
        if session is None:
            print(f"Session not found (returned None), creating new session: {SESSION_ID}")
            session = await session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=SESSION_ID
            )
        else:
            print(f"Loaded existing session: {SESSION_ID}")
            
        print(f"Session state: {session.state}")
    except ValueError:
        # Fallback in case it raises ValueError in some versions
        print(f"Session lookup failed (ValueError), creating new session: {SESSION_ID}")
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
        print(f"Created session state: {session.state}")

    # Debug: List sessions to ensure it's there
    all_sessions = await session_service.list_sessions(app_name=APP_NAME)
    #print(f"All sessions in DB: {[s.id for s in all_sessions]}")

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
