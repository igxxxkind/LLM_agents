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
    A Retriever is an interface that returns A DOCUMENT given A QUERY. It is not a vectorstore.
    A retriever DOES NOT store any document. 
    
    Splitters are used to split data before loading it into a vector store.
    There ae several textsplitters available in the langchain_community package.
    RecursiveText, code, characters, html.
    
    NExt step is to load data into a vector store. We need a Loader for this task and a database.
    Here we use redia as a vector_store and a youtube loader.
"""

from langchain_community.document_loaders import YoutubeLoader

loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=AOEGOhkGtjI", add_video_info=False)

docs=loader.load()

llm_gpt4 = ChatOpenAI(model="gpt-4o")

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20,
    length_function = len,
    is_separator_regex=False
)

docs_split = text_splitter.split_documents(docs)

len(docs_split)

from pinecone import Pinecone, ServerlessSpec
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "langchain-test-index"  # change if desired

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = PineconeVectorStore(index=index_name, embedding=embeddings)

pine_base = PineconeVectorStore.from_documents(docs_split, index_name = index_name, embedding=embeddings) 

retriever = pine_base.as_retriever(search_kwargs={"k": 10})

retriever.invoke("data_analysis")


