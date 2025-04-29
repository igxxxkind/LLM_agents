"""
    Tools are interfaces that an agent, chain or LLM can use to interact with the world.
    Toolkits - collection of tools designed to be used together for specific tasks.
    There are two wyas to use a tool:
    1. Run a tool: my_tool.run("query")
    2. Bind a tool to a LLM: llm.bind_tools([tool1, tool2, tool3])
    Then we will have an LLm that can use the tools in chains and agents
    
"""

from langchain_community.tools import YouTubeSearchTool
youtube_tool = YouTubeSearchTool()
youtube_tool.run("Rabbitmetrics")

llm_with_tools = llm_gpt4.bind_tools([youtube_tool])

msg = llm_with_tools.invoke("Rabbitmetrics YT videos")

msg.tool_calls[0]['args']["query"]

# use lambda to extract arguments from the LLM_with_tools
# and feed them tothe youtube tool

chain=llm_with_tools | (lambda x: x.tool_calls[0]['args']["query"]) | youtube_tool
# llm | name of a channel | youtube tool # to invoke a random chain about youtube videos

chain.invoke("Find some Rabbitmetrics videos on langchain")


