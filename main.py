from dotenv import load_dotenv
import os
import google.generativeai as genai
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()

# Configure with your Google API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model
llm = genai.GenerativeModel('gemini-2.0-flash')

class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical", "creative", "technical"] = Field(
        ...,
        description="""Classify the message as:
        - 'emotional': for feelings/personal issues
        - 'logical': for facts/information
        - 'creative': for artistic/innovative requests
        - 'technical': for coding/technical help"""
    )

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None

def classify_message(state: State):
    last_message = state["messages"][-1]
    
    prompt = """Classify this message as: emotional, logical, creative, or technical:
    - emotional: feelings, personal issues
    - logical: facts, information
    - creative: art, writing, innovative ideas
    - technical: coding, technical help
    
    Message: {message}
    
    Respond ONLY with one word: emotional, logical, creative, or technical""".format(
        message=last_message.content
    )
    
    response = llm.generate_content(prompt)
    classification = response.text.strip().lower()
    
    if classification not in ["emotional", "logical", "creative", "technical"]:
        classification = "logical"  # default fallback
    
    return {"message_type": classification}


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None

def classify_message(state: State):
    last_message = state["messages"][-1]
    
    prompt = """Classify the following user message as either 'emotional' or 'logical':
    - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
    - 'logical': if it asks for facts, information, logical analysis, or practical solutions
    
    User message: {message}
    
    Respond ONLY with either 'emotional' or 'logical', nothing else.""".format(message=last_message.content)
    
    response = llm.generate_content(prompt)
    classification = response.text.strip().lower()
    
    # Simple validation
    if classification not in ["emotional", "logical"]:
        classification = "logical"  # default fallback
    
    return {"message_type": classification}


def router(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}
    elif message_type == "creative":
        return {"next": "creative_agent"}
    elif message_type == "technical":
        return {"next": "technical_agent"}
    
    return {"next": "logical"}


def therapist_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "user", "parts": ["""You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked."""]},
        {"role": "user", "parts": [last_message.content]}
    ]
    
    response = llm.generate_content(messages)
    return {"messages": [{"role": "assistant", "content": response.text}]}


def logical_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "user", "parts": ["""You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses."""]},
        {"role": "user", "parts": [last_message.content]}
    ]
    
    response = llm.generate_content(messages)
    return {"messages": [{"role": "assistant", "content": response.text}]}


def creative_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "user", "parts": ["""You are a creative assistant. Help with:
                        - Story writing
                        - Art ideas
                        - Innovative concepts
                        - Creative problem solving
                        Be imaginative and original!"""]},
        {"role": "user", "parts": [last_message.content]}
    ]
    
    response = llm.generate_content(messages)
    return {"messages": [{"role": "assistant", "content": response.text}]}


def technical_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "user", "parts": ["""You are a technical expert. Provide:
                        - Code solutions
                        - Technical explanations
                        - System design help
                        - Debugging assistance
                        Be precise and accurate!"""]},
        {"role": "user", "parts": [last_message.content]}
    ]
    
    response = llm.generate_content(messages)
    return {"messages": [{"role": "assistant", "content": response.text}]}


graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_node("creative", creative_agent)
graph_builder.add_node("technical", technical_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {
        "therapist": "therapist",
        "logical": "logical",
        "creative": "creative",
        "technical": "technical"
    }
)

graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)
graph_builder.add_edge("creative", END)
graph_builder.add_edge("technical", END)

graph = graph_builder.compile()


# Run the chatbot
def run_chatbot():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run_chatbot()