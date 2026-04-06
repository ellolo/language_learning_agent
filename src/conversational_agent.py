from typing import Any

from typing_extensions import Literal, NotRequired

from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langgraph.types import Command
from langchain.agents import create_agent
from langchain.agents import AgentState
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from language_agent import build_graph


##########################
# STATE AND INITIALIZATION
##########################

# build the language learning workflow graph
language_workflow = build_graph()

#set state
USER_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
LANGUAGES = ["English", "Spanish", "French", "German", "Italian"]

UserLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2"]
TargetLanguage = Literal["English", "Spanish", "French", "German", "Italian"]

class ConversationalAgentState(AgentState):
    user_name: NotRequired[str]
    user_level: NotRequired[UserLevel]
    target_language: NotRequired[TargetLanguage]


#######
# TOOLS
#######

@tool
def get_user_name(runtime: ToolRuntime):
    """
    Get the user name.
    """
    if "user_name" not in runtime.state:
        return "User name not available. Ask the user to provide their name first."
    return f"User name: {runtime.state["user_name"]}."


@tool
def get_user_level(runtime: ToolRuntime):
    """
    Get the user level.
    """
    if "user_level" not in runtime.state:
        return "User level not available. Ask the user to provide their level first."
    return f"User level: {runtime.state["user_level"]}."


@tool
def get_target_language(runtime: ToolRuntime):
    """
    Get the target language.
    """
    if "target_language" not in runtime.state:
        return "Target language not available. Ask the user to provide their target language first."
    return f"Target language: {runtime.state["target_language"]}."


@tool
def update_user_name(user_name: str, runtime: ToolRuntime):
    """
    Update the user name.
    """
    return Command(
        update={
            "user_name": user_name,
            "messages": [ToolMessage(content=f"User name updated to {user_name}.", tool_call_id=runtime.tool_call_id)],
        }
    )


@tool
def update_user_level(user_level: UserLevel, runtime: ToolRuntime):
    """
    Update the user level in the state.
    """
    if user_level not in USER_LEVELS:
        return f"Invalid user level: {user_level}. Valid user levels are: {USER_LEVELS}."
    return Command(
        update={
            "user_level": user_level,
            "messages": [ToolMessage(content=f"User level updated to {user_level}.", tool_call_id=runtime.tool_call_id)],
        }
    )


@tool
def update_target_language(target_language: TargetLanguage, runtime: ToolRuntime):
    """
    Update the target language.
    """
    if target_language not in LANGUAGES:
        return f"Invalid target language: {target_language}. Valid target languages are: {LANGUAGES}."
    return Command(
        update={
            "target_language": target_language,
            "messages": [ToolMessage(content=f"Target language updated to {target_language}.", tool_call_id=runtime.tool_call_id)],
        }
    )


@tool
def language_learning_workflow(
    user_message: str,
    user_level: str,
    target_language: str,
):
    """
    This tool will start the language learning workflow.
    The tool generates learning material that matches the user message, the user level and target language.
    Two types of learning material can be generated:
    - grammar: a grammar exercise in the target language at the user level. The user can decide the topic of the grammar exercise via the user message.
    - reading: a reading text in the target language at the user level. The user can decide the topic of the reading material via the user message.
    The tool returns the learning material in plain text.
    """
    result = language_workflow.invoke(
        {
            "user_message": user_message,
            "user_level": user_level,
            "target_language": target_language,
        }
    )
    response = result["response"]
    return response


############
# MIDDLEWARE
############

# runs the language learning workflow tool only if the user level and target language are available
@wrap_tool_call
async def check_state_available(request: ToolCallRequest, handler):

    tool_name = request.tool_call["name"]
    if tool_name != "language_learning_workflow":
        return await handler(request)

    tool_call_id = request.tool_call["id"]
    args = request.tool_call["args"]

    # check if the user message is non-empty
    user_message = args["user_message"]
    if not isinstance(user_message, str) or not user_message.strip():
        return ToolMessage(
            content=("Error calling the tool: you must provide a non-empty `user_message` argument describing what the learner wants."),
            tool_call_id=tool_call_id,
            name=tool_name,
        )

    # check if the user level and target language are available
    user_level = request.state.get("user_level", None)
    target_language = request.state.get("target_language", None)

    if user_level is None or target_language is None:
        missing_fields_message = "Error calling the tool. Ask the user to provide the following information before calling the tool: "
        if user_level is None:
            missing_fields_message += "the user level (one of these values: A1, A2, B1, B2, C1, C2), "
        if target_language is None:
            missing_fields_message +=  "the target language (one of these values: English, Spanish, French, German, Italian), "
        missing_fields_message = missing_fields_message.rstrip(", ") + "."
        return ToolMessage(
            content=missing_fields_message,
            tool_call_id=tool_call_id,
            name=tool_name,
        )

    # inject state values into the tool call arguments
    new_args = {
        "user_message": user_message.strip(),
        "user_level": user_level,
        "target_language": target_language,
    }
    new_tool_call = {**request.tool_call, "args": new_args}
    new_request = request.override(tool_call=new_tool_call)
    return await handler(new_request)


#######
# AGENT
#######

model = init_chat_model(
    model="gpt-4o-mini",
    temperature=0.7,
)

all_tools = [
    get_user_name,
    get_user_level,
    get_target_language,
    update_user_name,
    update_user_level,
    update_target_language,
    language_learning_workflow,
]


def build_conversational_agent(
    *,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Create the conversational agent graph.

    LangGraph API / ``langgraph dev`` injects persistence and **rejects** compiled graphs
    that already bundle a custom checkpointer. Use ``checkpointer=None`` (the default)
    for Studio; use ``InMemorySaver()`` in notebooks when you need thread memory locally.
    """
    return create_agent(
        model=model,
        tools=all_tools,
        middleware=[check_state_available],
        state_schema=ConversationalAgentState,
        checkpointer=checkpointer,
        system_prompt="""
    You are a conversational agent that helps the user learn a new target language.
    You must start the conversation in English.
    You must as soon as possible ask the user if they want to switch to the target language.
    If they do, make sure you know the proficiency level of the user and adapt your conversation accordingly.
    You can use the language_learning_workflow tool to generate learning material (reading or grammar exercises) if the user asks for it.
    """,
    )


# For ``langgraph dev`` / LangGraph Cloud: no custom checkpointer (platform handles persistence).
conversational_agent = build_conversational_agent()

# For notebooks and local scripts that need ``thread_id`` / checkpointing without the API.
conversational_agent_with_checkpointer = build_conversational_agent(
    checkpointer=InMemorySaver()
)
