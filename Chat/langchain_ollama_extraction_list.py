from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.chains import create_extraction_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Optional, List
from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_mistralai import ChatMistralAI
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
import dotenv
dotenv.load_dotenv()

class LogEntry(BaseModel):
    """Schema for extracting information from a log entry."""

    # Each field is optional, allowing the model to decline to extract it.
    # Each field has a description used by the LLM, helping to improve extraction results.

    url: Optional[str] = Field(default=None, description="The URL that was accessed.")
    level: Optional[str] = Field(default=None, description="The log level (INFO, ERROR, etc.).")
    method: Optional[str] = Field(default=None, description="The HTTP method used in the request.")
    timestamp: Optional[str] = Field(default=None, description="The timestamp of the log entry.")
    ip_address: Optional[str] = Field(default=None, description="The IP address from which the request originated.")
    status_code: Optional[int] = Field(default=None, description="The HTTP status code returned in the response.")
    response_time: Optional[int] = Field(default=None, description="The response time in milliseconds.")


class LogEntryData(BaseModel):
    """Extracted data about LogEntry"""

    log_entries: List[LogEntry]


# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
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
# log_text = open('data/extraction_log').read()
# 定义文本加载器，指定读取的文件路径
loader = TextLoader("data/extraction_log")
log_text = loader.load()
runnable = prompt | llm.with_structured_output(schema=LogEntryData)
results = runnable.invoke({'text': log_text})
for log_entry in results.log_entries:
    print(log_entry.dict())