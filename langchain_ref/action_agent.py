from langchain.llms import OpenAI
from langchain.agents import initialize_agent, load_tools, AgentType


openai_api_key = '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'
llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
tools = load_tools(['serpapi', 'llm-math'], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run('Who is Leo DiCaprios girlfriend?What is her current age raised to the 0.43 power?')
