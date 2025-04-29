"""
Agents are programs capable of performing tasks by integrating LLMs with tools.
They do so by leveraging the LLM's capability to generate JSON or other structured formats as output.

Agent is a class that uses an LLM to choose a sequence of actions to take.
The main difference to a chain: in a chain all actions are hardcoded.
In agents, LLM decides what to do next and what tools to use based on the output of the previous action.
Agent may loop back and repeat a previous step.
"""
from langchain_community.document_loaders import YoutubeLoader
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import YouTubeSearchTool
youtube_tool = YouTubeSearchTool()

llm_gpt4 = ChatOpenAI(model="gpt-4o")


prompt = hub.pull("hwchase17/openai-tools-agent")

prompt.messages

# contains a variable called "agent_scratchpad" where agents can store information between invocations
tools = [youtube_tool]

agent = create_tool_calling_agent(llm_gpt4, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# these are the only three things we need to create an agent: prompt, tool and LLM
# youtube_tool.run({'query': 'LangChain', 'num_results': 5}) -> Error

agent_executor.invoke({"input": "WHat are the most popular langchain videos on YT?", 'num_results': 5})


from langchain_core.tools import tool

@tool
def transcibe_videos(video_url: str)->str:
    "Extract transcripts from YT video"
    loader = YoutubeLoader.from_youtube_url(
        video_url, add_video_info=False
    )
    docs=loader.load()
    return docs

tools = [youtube_tool, transcibe_videos]

agent = create_tool_calling_agent(llm_gpt4, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke(
    {
        "input": "What topics does the rabbitmetric YT channel cover?"
    }
)

