import datetime
from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser, PydanticToolsParser
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from schema import AnswerQuestion, ReviseAnswer

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
            You are exprt researcher.
            Current time is {time}.

            1. {first_instruction}
            2. Reflect and Critique your own answer. Be severe to maximize improvement
            3. Recommend search queries to research information and improve your answer.
        """),
        MessagesPlaceholder(variable_name="messages"),
        ("system","Answer the user's question above using the required format.")
    ]
).partial(time = lambda: datetime.datetime.now().isoformat())


llm = ChatOpenAI(model="gpt-4o-mini")
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools= [AnswerQuestion])

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction = "Provide a detailed ~250 word answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)


revise_instructions = """
    Revise your previous answer using the new information.
    You should use the previous critique to add important information to your answer.
    You must include numerical citations in your revise answer to insure it can be verified.
    Add a reference section to the buttom of your answer (which does not count towards the word limit)
    In form of:
     - [1] https://www.example.com
     - [2] https://www.example.com
    You should use the previous critique to remove superfluous information from your answer and make sure 
    It is not more that ~250 words.
"""


revisor = actor_prompt_template.partial(
        first_instruction = revise_instructions
    ) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

if __name__ == "__main__":
    print("Starting first responder execution...")

    human_message = HumanMessage(content="""
        Write about AI powered SOC / autonomous SOC problem domain.
        list startups that do that and raised capital.
    """)


    chain = (
        first_responder_prompt_template |
        llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion") |
        parser_pydantic

    )

    res = chain.invoke(input= {"messages": [human_message]})

    print(res)