from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchTool
from langchain.agents import Tool
from langchain.tools import BaseTool

llm_gpt4 = ChatOpenAI(model="gpt-4o", temperature=0.0)
search = DuckDuckGoSearchTool()

tools = [Tool(name = "search", 
              func=search.run,
              description = "useful for when you need to answer questions about current events. You should ask targeted questions.")]

# Custom tools are just functions that will get called

# Tool 1
def meaning_of_life(input=""):
    return "The meaning of life is 42 if rounded but actually is 42.17658"

life_tool = Tool(
    name="Meaning of Life",
    func=meaning_of_life,
    description="Useful for when you need to answer questions about the meaning of life. Input should be MOL."
)

# Tool 2
import random

def random_num(input = ""):
    return random.randint(0,5)

random_tool = Tool(
    name="Random number generator",
    func=random_num,
    description="Useful for when you need to get a random number. Input should be 'random'"
)

# Creating an agent
from langchain.agents import initialize_agent

tools = [search, random_tool, life_tool]

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=3,
    return_messages=True
)

# create an agent

conversational_agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm_gpt4,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=memory)

conversational_agent('What time is in London?')

conversational_agent('Can you give me a random number?')

conversational_agent('What is the meaning of life?')


built_in_prompt = conversational_agent.agent.llm_chain.prompt.messages[0].prompt.template

fixed_prompt = '''
    Assistant is a large language model trained by OpenAI.
    
    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Assistant doesn't know anything related to random numbers or to the meaning of life and should use a tool for questions about these topics.
    
    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
    '''

conversational_agent.agent.llm_chain.prompt.messages[0].prompt.template = fixed_prompt

conversational_agent('What time is in London?')

conversational_agent('Can you give me a random number?')

conversational_agent('What is the meaning of life?') #8:43

