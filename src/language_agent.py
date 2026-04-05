import os
from langchain.agents.middleware.types import wrap_model_call
from typing_extensions import Literal, TypedDict
from tavily import TavilyClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from typing import Any
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware



#########
# GLOBALS
#########

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


########################
# STATE AND OUTPUT TYPES
########################

# output of the router node
class Classification(BaseModel):
    classification: Literal["grammar", "reading", "unknown"] = Field(description="The classification of the user's message")

# output of the agent nodes
class AgentOutput(BaseModel):
    agent_name: str = Field(description="The name of the agent that generated the result")
    result: str = Field(description="The result returned by the agent")

# input state of the graph
class InputState(BaseModel):
    # required
    user_message: str = Field(description="The message from the user")
    # required
    user_level: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = Field(description="The level of the user")
    # optional
    target_language: Literal["English", "Spanish", "French", "German", "Italian"] = Field(default="English", description="The target language for the user to learn")


# full accumulated graph state (input fields plus router / agent / answer updates)
class GraphState(InputState):
    classification: Classification | None = Field(default=None, description="The classification of the user's message")
    agent_output: AgentOutput | None = Field(default=None, description="The output of the agent that generated the result")
    response: str | None = Field(default=None, description="The response to the user's message")


########
# TOOLS
########

@tool
def web_search(query: str) -> dict[str, Any]:
    """
    Search the web for the given query.
    """
    return tavily.search(query)


########
# AGENTS
########

# context to pass to the agents in order for them to generate sensical results
class AgentContext(TypedDict):
    target_language: Literal["English", "Spanish", "French", "German", "Italian"]
    user_level: Literal["A1", "A2", "B1", "B2", "C1", "C2"]


# READING AGENT

# set system prompt right before calling the LLM model,
# dynamically adding the user level and target language
@wrap_model_call
def set_reading_agentsystem_prompt(request, handler):
    user_level = request.runtime.context["user_level"]
    target_language = request.runtime.context["target_language"]
    system_prompt = f"""
    You are an agent specialized in generating reading material for {target_language} language learning.
    You generate material for CEFR level {user_level}.
    You take in input a user message that defines what type of reading material the user wants to read.
    You must generate in output a maximum 3-paragraph text in {target_language} at CEFR level {user_level} matching the user wishes.
    You can use the web_search tool to help you generate text that fits the user message. 
    """
    request = request.override(system_message=system_prompt)
    return handler(request)

reading_agent = create_agent(
    model = llm,
    tools = [web_search],
    context_schema = AgentContext,
    middleware = [
        ToolCallLimitMiddleware(thread_limit=20, run_limit=10),
        set_reading_agentsystem_prompt,
    ],
)


# GRAMMAR AGENT

# structured output of the grammar agent
class GrammarAgentStructuredOutput(BaseModel):
    exercise_type: Literal["fill_in_the_blank", "multiple_choice", "question"] = Field(description="The type of generated exercise")
    exercise_content: str = Field(description="The content of the generated exercise")
    exercise_answer: str = Field(description="The correct answer to the generated exercise")


@wrap_model_call
def set_grammar_agentsystem_prompt(request, handler):
    user_level = request.runtime.context["user_level"]
    target_language = request.runtime.context["target_language"]
    system_prompt = f"""
    You are an agent specialized in generating grammar exercises for {target_language} language learning.
    You generate exercises appropriate for CEFR level {user_level}.
    You take in input a user message that defines what type of grammar exercise the user wants to solve.
    You must generate only one exercise, choose from the following types:
    - fill_in_the_blank: a sentence with a blank, the user must fill in the blank.
    - multiple_choice: a sentence with multiple choices, the user must choose the correct answer.
    - question: a question with a answer, the user must answer the question.
    The exercise must be written in {target_language} and appropriate for CEFR level {user_level}.
    """
    request = request.override(system_message=system_prompt)
    return handler(request)

grammar_agent = create_agent(
    model = llm,
    tools = [web_search],
    response_format = GrammarAgentStructuredOutput,
    context_schema = AgentContext,
    middleware = [
        ToolCallLimitMiddleware(thread_limit=20, run_limit=10),
        set_grammar_agentsystem_prompt,
    ],
)


#######
# NODES
#######

# a node of a langgraph graph takes in input the state and returns an update to the state
def reading_node(state: GraphState):
    response = reading_agent.invoke(
        {"messages": [HumanMessage(content=state.user_message)]},
        context={
            "target_language": state.target_language,
            "user_level": state.user_level
            }
    )
    # the response of a langchain agent is its state, which includes the messages
    output = response["messages"][-1].content
    return {"agent_output": AgentOutput(agent_name="reading_agent", result=output)}


def grammar_node(state: GraphState):
    response = grammar_agent.invoke(
        {"messages": [HumanMessage(content=state.user_message)]},
        context={
            "target_language": state.target_language,
            "user_level": state.user_level
            }
    )
    structured_output = response.get("structured_response")
    if not isinstance(structured_output, GrammarAgentStructuredOutput):
        raise ValueError(
            "Grammar agent did not return structured output; "
            f"got {type(structured_output).__name__!r}"
        )
    output = f"""
    Exercise type:{structured_output.exercise_type}
    Exercise content: {structured_output.exercise_content}
    Exercise answer: {structured_output.exercise_answer}
    """
    return {"agent_output": AgentOutput(agent_name="grammar_agent", result=output)}

# since the router node does not need tools and keep message history, we can 
# implement it with a simple llm call, instead of using a langchain agent.
def router_node(state: GraphState):

    llm_with_structured_output = llm.with_structured_output(Classification)
    
    system_prompt = """
    Analyze the user's message and classify it into one of the following categories:
    - grammar: The user want to obtain a grammar exercise.
    - reading: The user want to obtain a text to read.
    You can only select one of the categories.
    If multiple categories are present, select the most relevant one.
    If the user's message is not related to any of the categories, return "unknown".
    """

    user_prompt = f"""
    User message: {state.user_message}
    """

    prompt = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
        ]
    
    response = llm_with_structured_output.invoke(prompt)
    return {"classification": response}


# node that produces the final response to the user of the workflow
def answer_node(state: GraphState):
    assert state.classification is not None
    if state.classification.classification == "unknown":
        return {"response": "I'm sorry, I don't know how to answer that question. I can only help you with grammar and reading exercises."}
    assert state.agent_output is not None
    return {"response": state.agent_output.result}


def route_to_agents(state: GraphState) -> Literal["grammar_node", "reading_node", "unknown"]:
    # decide to which node to route based on the classification provided by the router node
    assert state.classification is not None
    if state.classification.classification == "grammar":
        return "grammar_node"
    elif state.classification.classification == "reading":
        return "reading_node"
    return "unknown"


#######
# GRAPH
#######

def build_graph():
    builder = StateGraph(GraphState, input_schema=InputState, output_schema=GraphState)

    builder.add_node("router_node", router_node)
    builder.add_node("grammar_node", grammar_node)
    builder.add_node("reading_node", reading_node)
    builder.add_node("answer_node", answer_node)

    builder.add_edge(START, "router_node")
    builder.add_conditional_edges("router_node", route_to_agents, {
        "grammar_node": "grammar_node",
        "reading_node": "reading_node",
        "unknown": "answer_node"
    })
    builder.add_edge("grammar_node", "answer_node")
    builder.add_edge("reading_node", "answer_node")
    builder.add_edge("answer_node", END)

    return builder.compile()


language_workflow = build_graph()