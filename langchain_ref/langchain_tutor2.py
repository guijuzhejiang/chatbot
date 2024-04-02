from Chat import ChatService
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, load_tools, AgentType


openai_api_key = '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'
text='你是谁？'
llm = OpenAI(openai_api_key=openai_api_key)
llm.predict(text)

# 初始化ChatBot实例
# Chat = ChatService.ChatService(LANG="CN")
# reply_text = Chat.predict(text=text)
# print(f'reply:{reply_text}')

# chat_model = ChatOpenAI(openai_api_key=openai_api_key)
# reply_text = chat_model.predict(text)
# print(f'reply:{reply_text}')

# message = [HumanMessage(content=text)]
# message_reply = chat_model.predict(message)
# print(f'message_reply:{message_reply}')

prompt = PromptTemplate.from_template('给我一个很土但很好听的{对象}小名')
prompt_formated = prompt.format(对象='小狗')       #给我一个很土但很好听的小狗小名
print(f'prompt:{prompt_formated}')

#chain
chain = LLMChain(prompt=prompt, llm=llm)
chain.run('小狗')

#agent
tools = load_tools(['llm-math'], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run('25的3.5次方是多少？')

#chain包含对话聊天上下文
conversation = ConversationChain(llm=llm, verbose=True)
conversation.run("你好吗？")