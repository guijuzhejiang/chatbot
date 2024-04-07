import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import dotenv
dotenv.load_dotenv()

groq_api_key = os.environ.get("GROQ_API_KEY")
llm = ChatGroq(api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# prompt = ChatPromptTemplate.from_template("""
# Answer the following question based only on the provided context.
# Think step by step before providing a detailed answer.
# I will tip you $200 if the user finds the answer helpful.
# <context>
# {context}
# </context>
#
# Question:{input}
# """)

# prompt_messages = ChatPromptTemplate.from_messages([
#     {"role": "user", "content": "who is DiCaprio?"}
# ])

# prompt_messages = ChatPromptTemplate.from_messages([
#   ("human", "Hello, how are you?"),
#   ("ai", "I'm doing well, thanks!"),
#   ("human", "{input}"),
# ])

prompt_messages = ChatPromptTemplate.from_messages([
                    ("human", "Hello, how are you?"),
                    ("ai", "I'm doing well, thanks!"),
                    ("human", "That's good to hear."),
                ])

chain = prompt_messages.pipe(llm)
response = chain.invoke({
  # input: "are you happy?",
})
print(response)