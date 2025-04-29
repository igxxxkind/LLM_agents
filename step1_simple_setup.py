import textwrap

# from os.path import join, dirname
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv(r"D:\projects\Python\LLM_agents\.env")


llm_gpt4 = ChatOpenAI(model="gpt-4o")

system_prompt = """
You explain things to people in a way
that is easy to understand to a child.
You are a teacher.
"""
user_prompt = """
What is LangChain?
"""


messages = [SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)]

response = llm_gpt4.invoke(messages)
answer = textwrap.fill(response.content, 100)

print(answer)
