from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI

# import textwrap
# from os.path import join, dirname
# from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage, SystemMessage
# load_dotenv(r"D:\projects\Python\LLM_agents\.env")

llm_gpt4 = ChatOpenAI(model="gpt-4o")

"""
LCEL expression language simplifies building complex chains from basic blocks.
Use pipe operator to combain one compionent to the next. 
chain_1 = prompt | llm_gpt4 | output_parser
It is a declarative way to compose RUNNABLES into a chain.
query_message -> prompt -> chatmodel -> output_parser -> string => output
chain = chain_1 | chain_2
"""
   
"""
    RUNNABLES are the basic building blocks of the LLM system that can be invoked,
    batched, streamed, transformed and composed.
    chain = prompt| (lambda input: {"x": input}) | llm_gpt4 | output_parser
    Such chains can be invoked, batched and streamed.
    
    There are 4 types of RUNNABLE objects:
    1. RunnableSequence: class that chains together multiple runnable components. 
    It is ensured that each components receives the input and sequentially passes the output to the next component.
    All chains are RunnableSequence objects.
    2. RunnableLambda: a class that turns Python callable (function or lambda) into a Runnable object, 
    enabling it to be integrated into a runnable sequence.
    3. RunnablePassThrough: a class that passes the input through unchanged or
    adds additional keys to the output, allowing for flexible integration into runnable sequences in
    where input modification is needed.
    4. RunnableParallel: a class that runs multiple runnables in parallel and combines the results.
    Like running the same input but returning different outputs.
"""
summarize_prompt_template = """
You are a helpful assistant that explains AI concepts:
{context}
Summarize the context
"""

summarize_prompt = PromptTemplate.from_template(summarize_prompt_template)

output_parser = StrOutputParser()

chain = summarize_prompt | llm_gpt4 | JsonOutputParser()

chain.invoke(input={"context": "Provide an example of a JSON file?"})

print(type(chain))

####  Runnable Lambda

from langchain_core.runnables import RunnableLambda

summarize_chain = summarize_prompt | llm_gpt4 | output_parser

length_lambda = RunnableLambda(lambda summary: f"Summary length: {len(summary)} characters")

lambda_chain = summarize_chain | length_lambda

lambda_chain.invoke({"context": "what is langchain"})

print(type(lambda_chain.steps[-1]))

chain_with_function = summarize_chain | (lambda summary: f"Summary length: {len(summary)} characters")

chain_with_function.invoke({"context": "what is langchain"})

summarize_chain.invoke({"context": "what is langchain's runnableLambda"})

#### Runnable PassThrough ####

from langchain_core.runnables import RunnablePassthrough

summarize_chain = summarize_prompt | llm_gpt4 | output_parser

passthrough = RunnablePassthrough()

placeholder_chain = summarize_chain | passthrough | length_lambda

placeholder_chain.invoke({"context": "what is langchain"})

print(type(placeholder_chain.steps[-1]))
print(type(placeholder_chain.steps[-2]))

### PASS THROUGH AN ASSIGNMENT ###

wrap_summary_lambda = RunnableLambda(lambda summary: {"summary": summary})

assign_passthrough = RunnablePassthrough.assign(length = lambda x: len(x["summary"]))

summarize_chain = summarize_prompt | llm_gpt4 | output_parser| wrap_summary_lambda 

assign_chain = summarize_chain | assign_passthrough

summarize_chain.invoke({"context": "what is langchain"})
assign_chain.invoke({"context": "what is langchain"})

