"""
    Chains are a sequence of interconnected components that process a use4r's query,
    utilizing one or more LLMs, to generate and deliver valuable information to the user.
    
    These are pipelines in the world of LLMs. Query --->LLM
    
    Prompt is a set of instructions providedd by a user to guide the model response.
    It helps LLM understand the context and generate relevant and coherent output.
    
    Prompt templates are predefined recipes for generating prompts for language models.
    Document is a piece of text and metadata. Loaders have a load method
    for loading data as documents from a source.
    """
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI

# import textwrap
# from os.path import join, dirname
# from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage, SystemMessage
# load_dotenv(r"D:\projects\Python\LLM_agents\.env")

llm_gpt4 = ChatOpenAI(model="gpt-4o")

prompt_template = """
You are a helpful assistant that explains AI topics.
    Given the following input: {topic}.
    Provide Explanation of a given topic
"""

prompt = PromptTemplate(template=prompt_template, 
                        input_variables=["topic"])

chain = prompt | llm_gpt4

chain.invoke(input={"topic": "What is LangChain?"})

loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=8BV9TW490nQ&t=935s", 
    add_video_info=False
)

docs = loader.load()
transcript = docs[0].page_content

transcript
# summarize the YT video given transcript

prompt_template = """
You are a helpful assistant that explains YT videos.
    Given the following video transcript: {video_transcript}.
    Give a summary of the video.
"""
prompt_yt = PromptTemplate(
    template=prompt_template, input_variables=["video_transcript"]
)

chain = prompt_yt | llm_gpt4  # setting up the chain

chain.invoke(input={"video_transcript": docs}).content

# this chain takes a list of docs, 
# combines the documents and formats them all into a prompt

prompt_template = """
You are a helpful assistant that explains AI topics.
    Given the following context: {context}.
    Summarize what langchain can do
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context"])

chain = create_stuff_documents_chain(llm_gpt4, prompt)

output = chain.invoke(input={"context": docs})

print(output)
