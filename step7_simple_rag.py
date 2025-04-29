from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI

import time
import os
from os.path import join, dirname
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
load_dotenv(r"D:\projects\Python\LLM_agents\.env")

"""
RAG - retrieval augmented generation: augment the LLM knowledge with external data. 
1. Index external data in the vector store.=>  load data, chunk them, load in the vector database. 
2. Retrieve and generate the output with LLM. => fetch the context  and send it to the LLM model with the query.

Retrieving data is necessary for providing the context to the LLM model. 
Without the context it is impossible to minimize hallucinations of the LLM.
Moreover, the result is likely to be a hallucination only.
"""

from langchain.prompts import ChatPromptTemplate

llm_gpt4 = ChatOpenAI(model="gpt-4o")

template = """
Answer the question based only on the following context:
{context}
Question: {question}  
"""

prompt = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

output_parser = StrOutputParser()

# this is a simple RAG chain
chain = (
    {"context": (lambda x: x['question']) | retriever, # to extract question from the input using lambda function and pass the question to the retriever to get the context
     "question": (lambda x: x['question'])} # pass the question to LLM
    | prompt # to format the context and question into a prompt
    | llm_gpt4 # to generate the answer
    | output_parser) # to parse the output

answer = chain.invoke(input={"question": "What can you do with LLama 3?"})






