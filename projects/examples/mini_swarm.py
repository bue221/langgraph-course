from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langgraph_swarm import create_handoff_tool, create_swarm

model = ChatOpenAI(model="gpt-4o")

# STATE --------------------------------------------------------------------------

class GlobalState(TypedDict):
    # Swarm agent - active agent - who is the current agent?
    active_agent: str

    # Messages
    messages: Annotated[List[BaseMessage], add_messages]

    # Remaining steps - how many steps are left to complete?
    remaining_steps: int

    # Sum - a simple state variable
    sum: int
    
# TOOLS --------------------------------------------------------------------------

@tool(return_direct=True)
def add(a: int, b: int, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Add two numbers"""
    return Command(
        graph=Command.PARENT,
        update={
            "sum": a + b,
            "messages": [ToolMessage(content=f"SUM_CALCULATED", tool_call_id=tool_call_id)]
        },
    )
    
# PROMPTS ------------------------------------------------------------------------

def bob_prompt(state: GlobalState) -> str:
    print('state', state)
    return f"""
    You are Bob, you speak like a pirate.
    You are currently talking to {state["active_agent"]}.
    """

#  AGENTS ------------------------------------------------------------------------

alice = create_react_agent(
    model,
    [add, create_handoff_tool(agent_name="Bob")],
    prompt="You are Alice, an addition expert.",
    name="Alice",
    state_schema=GlobalState,
)


bob = create_react_agent(
    model,
    [create_handoff_tool(agent_name="Alice", description="Transfer to Alice, she can help with math")],
    prompt=bob_prompt,
    name="Bob",
    state_schema=GlobalState,
)

#  WORKFLOW AND MEMORY ------------------------------------------------------------
checkpointer = InMemorySaver()

workflow = create_swarm(
    [alice, bob],
    default_active_agent="Alice",
    state_schema=GlobalState,
)

app = workflow.compile(checkpointer=checkpointer)