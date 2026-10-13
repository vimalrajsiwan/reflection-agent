from typing import List, Dict, Any, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from reflection.chain import reflection_chain, generation_chain

load_dotenv()

class MessageGraph(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def generation_node(state: MessageGraph):
    return {"messages": [generation_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessageGraph):
   res = reflection_chain.invoke({"messages": state["messages"]})
   return {"messages": [HumanMessage(content=res.content)]} 


GENERATOR = "generator"
REFLECTOR = "reflector"

builder = StateGraph(state_schema=MessageGraph)

builder.add_node(GENERATOR, generation_node)
builder.add_node(REFLECTOR, reflection_node)    
builder.set_entry_point(GENERATOR)

def should_continue(state: MessageGraph):
    if len(state["messages"]) >= 6:
        return END
    return REFLECTOR

builder.add_conditional_edges(GENERATOR, should_continue, {END: END, REFLECTOR: REFLECTOR})

builder.add_edge(REFLECTOR, GENERATOR)

graph = builder.compile()

print(graph.get_graph().draw_mermaid())

##graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Starting graph execution...")

    input = {
        "messages": [
            HumanMessage(content=
                         """Make this tweet better:" @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

            """)
        ]
    }

    response = graph.invoke(input)

    print("Final response:------------------------\n\n")
    print(response["messages"][-1].content)
