from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            "You are a viral twitter influencer grading a tweet. "
            "Generate critieuqe and recomendation for users, "
            "Alwasy provide detailed recomendations, including request for lenght, virality, style, etc."
        )
        ,
        MessagesPlaceholder(variable_name="messages")
    ]
)     

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts. "
            "Generate the best twitter post possible for the user's request."
            "If the user provide the critique, respond with a revised version of your previous attempts."
        )
        ,
        MessagesPlaceholder(variable_name="messages")
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

reflection_chain = reflection_prompt | llm
generation_chain = generation_prompt | llm

