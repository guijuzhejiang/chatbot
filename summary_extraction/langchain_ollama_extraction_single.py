from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.chains import create_extraction_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_mistralai import ChatMistralAI
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
import dotenv
dotenv.load_dotenv()

class LanguageLearningSummary(BaseModel):
    """Schema for summarizing language learning performance."""

    # Each field is optional, allowing the model to decline to extract it.
    # Each field has a description used by the LLM, helping to improve extraction results.

    words_used: Optional[str] = Field(default=None, description="The words used by the student.")
    review: Optional[str] = Field(default=None, description="The review of the student's performance.")
    summary: Optional[str] = Field(default=None, description="A summary of the student's learning session.")
    evaluation: Optional[str] = Field(default=None, description="An evaluation of the student's language skills.")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm specialized in language learning evaluation. "
            "Extract and summarize relevant information from the chat log. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)

# Initialize Groq Langchain chat object and conversation
# ['llama3-70b-8192', 'llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
llm = ChatGroq(model_name='llama3-70b-8192', temperature=0.0)
# qwen2:7b-instruct-q5_K_M ,llama3:70b-instruct-q5_K_M,summerwind/japanese-starling-chatv:7b-q5_K_M,qwen2:1.5b-instruct-q5_K_M
# llm = ChatOllama(model="llama3:70b-instruct-q5_K_M",
#                  num_ctx=num_ctx_llama3_70b,
#                  temperature=0,
#                  num_threads=16,
#                  verbose=True)
# llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True)
# llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
# log_text = open('data/language_learning_log.txt').read()
# 定义文本加载器，指定读取的文件路径
loader = TextLoader("data/language_learning_log.txt")
log_text = loader.load()

runnable = prompt | llm.with_structured_output(schema=LanguageLearningSummary)
results = runnable.invoke({'text': log_text})

# Print the extracted summary
print(results.dict())
