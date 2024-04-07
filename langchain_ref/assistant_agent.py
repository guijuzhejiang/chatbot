from langchain.agents import load_tools, initialize_agent, types
from langchain.chat_models import ChatOpenAI
import dotenv
dotenv.load_dotenv()


llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=types.AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
print(f'prompt: {agent.agent.llm_chain.prompt.template}')

result = agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
print(result)