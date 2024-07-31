import os

import dotenv
from datetime import datetime
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOllama
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_core.output_parsers import StrOutputParser
# from src.chat.utils import History
# from src.chat.conversation_db_buffer_memory import ConversationBufferDBMemory


dotenv.load_dotenv()

def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """

    # language="English, French, Spanish, German, Italian, Portuguese, Russian, "目前只支持英语，其他语言后面会出现英语
    template = """
        你是一个智能助手，你的名字是Mary。
        {history}
        Human:{input}
        AI:
    """
    prompt = ChatPromptTemplate.from_template(template=template)
    prompt = prompt.partial(language="French", name='Mary')
    # Initialize Groq Langchain chat object and conversation
    # ['llama3-70b-8192', 'llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    llm = ChatOllama(model="llama3.1:8b-instruct-q5_K_M", temperature=0.7)
    memory = ConversationBufferWindowMemory(human_prefix="Human",
                                            ai_prefix="AI",
                                            memory_key="history",
                                            k=10)
    conversation_chain = ConversationChain(llm=llm, memory=memory, verbose=True, prompt=prompt)
    # chain_result = conversation_chain.run(input="你好，你是谁？")
    # chain_result = conversation_chain.predict(input="你好，你是谁？")
    start = datetime.now()
    chain_result = conversation_chain("你好，你是谁？用中文回答")
    end = datetime.now()
    print("AI:", chain_result["response"], "time:", end - start)


if __name__ == "__main__":
    main()





