from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from ChatGLM3 import ChatGLM3
from langchain import hub
from dotenv import load_dotenv
load_dotenv()


template = """{question}"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# default endpoint_url for a local deployed ChatGLM api server
# openai_api_base="http://127.0.0.1:8000/v1"
MODEL_PATH = '/media/zzg/GJ_disk01/pretrained_model/THUDM/chatglm3-6b'
llm = ChatGLM3()
llm.load_model(MODEL_PATH)
# prompt = hub.pull("hwchase17/structured-chat-agent")
# llm = ChatOpenAI(model=MODEL_PATH, max_tokens=1024)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "北京和上海两座城市有什么不同？"

print(llm_chain.run(question))