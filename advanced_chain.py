from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(r"D:\projects\Python\LLM_agents\.env")

prompt = ChatPromptTemplate.from_template("tell me a joke about  {topic}")
model = ChatOpenAI(model="gpt-4o")
output_parser = StrOutputParser()

chain = prompt | model | output_parser # to pipe things together

chain.invoke({"topic": "Trump"})

from langchain_core.messages.human import HumanMessage

messages = [HumanMessage(content="tell me a short joke about UBS")]
model.invoke(messages) 

############################

from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

# RunnablePassThrrough does not change a thing

chain = RunnablePassthrough() | RunnablePassthrough() | RunnablePassthrough()

chain.invoke('hello world')

# RunnableLambda is to let us run some custom functions

def input_to_upper(input: str):
    output = input.upper()
    return output

chain = RunnablePassthrough() | RunnableLambda(lambda x: input_to_upper(x)) | RunnablePassthrough()

chain.invoke('Hello World')

# RunnableParallel is to run 2 tasks with the same input - it is not about multithreading

chain = RunnableParallel({"x": RunnablePassthrough(), "y": RunnablePassthrough()})

chain.invoke('hello')

chain.invoke({"input": "hello", "input2": "goodbye"})

chain = RunnableParallel({"x": RunnablePassthrough(), "y": lambda z: z["input2"]}) 

chain.invoke({"input": "hello", "input3": "goodbye"}) # -> Error

chain.invoke({"input": "hello", "input2": "goodbye"}) # addresses the input2 key in the dictionary
# Note; this cannot be chained to a next runnable. RunnableLambda must be used in thi example

### NESTED CHAINS

def find_keys_to_uppercase(input: dict):
    output = input.get("input", "not found").upper()
    return output

chain = RunnableParallel({"x": RunnablePassthrough() | RunnableLambda(find_keys_to_uppercase), 'y': lambda z: z["input2"]})

chain.invoke({"input": "hello", "input2": "goodbye"}) 
chain.invoke({"input": "hello", "input2": "goodbye"})# to get NOT FOUND for the input key

# add keys on the fly

chain = RunnableParallel({'x': RunnablePassthrough()})

def assign_func(input):
    return 100

def multiply_func(input):
    return input*11.33

chain.invoke({"input": "hello", "input2": "goodbye"}) 

chain = RunnableParallel({"x": RunnablePassthrough()}).assign(extra=RunnableLambda(assign_func)) 
# to assign a new key-value _extra_ when we invoke a new chain

result = chain.invoke({"input": "hello", "input2": "goodbye"}) 

print(result)

# compose multiple chains together for a more complex chain

def extractor(input: dict): # returns a string out of a dictionary
    return input.get("extra", 'Key not found')

def cupper(upper: str):
    return str(upper).upper()

new_chain = RunnableLambda(extractor) | RunnableLambda(cupper) # first Lambda pass the output to a second Lambda

new_chain.invoke({'extra': 'test'})

final_chain = chain | new_chain

final_chain.invoke({"inp455646541ut": "helldwedro", "inputfsdf2": "gfsdfoodbye"})

# in the first chain (79) we add another key_value pair to the chain output leaving the rest untouched
# in the second chain (94) we look for the added key-value pair, extract the value and act on it
# the input dict has been lost since the second chain works with one field only

#### real stuff

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

vectorstore = FAISS.from_texts(['Cats love tuna'], embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template=template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI()
    | StrOutputParser()
    )

rag_chain_1 = ( # plain analogue
    RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
    | prompt
    | ChatOpenAI()
    | StrOutputParser()
    )


rag_chain.invoke('What do cats like to eat?')









