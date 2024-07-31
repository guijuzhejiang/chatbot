from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import OllamaEmbeddings
import os
import dotenv
dotenv.load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

loader = UnstructuredWordDocumentLoader("/home/zzg/商业项目/清华/ソフトウエア基本契約書.docx")
docs = loader.load()
print(f'docs length:{len(docs)}')
# print(docs[0].page_content)

splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
chunks = splitter.split_documents(docs)
print(f'chunks length:{len(chunks)}')
# print(chunks[0].page_content)
# embeddings = HuggingFaceInferenceAPIEmbeddings(
#     model_kwargs={"model_name": "gte-Qwen2-7B-instruct", "device": "cpu"},
#     api_key=HF_API_KEY
# )
#考虑加入GraphRAG一起融合
embeddings = OllamaEmbeddings(model="rjmalagon/gte-qwen2-7b-instruct-embed-f16")
db = Chroma.from_documents(chunks, embeddings)
db_retriever = db.as_retriever(search_kwargs={"k": 4})

keyword_retriever = BM25Retriever.from_documents(chunks)
keyword_retriever.k = 4

ensemble_retriever1 = EnsembleRetriever(
    retrievers=[
        db_retriever,
        keyword_retriever,
    ],
    weights=[0.5, 0.5]
)

ensemble_retriever2 = EnsembleRetriever(
    retrievers=[
        db_retriever,
        keyword_retriever,
    ],
    weights=[0.99, 0.01]
)

# llm = HuggingFaceHub(
#     repo_id="Qwen/Qwen2-7B-Instruct",
#     model_kwargs={"temperature": 0.3, "max_length": 1024},
#     huggingfacehub_api_token=HF_API_KEY
# )
llm = ChatOllama(model="qwen2:7b-instruct-q5_K_M",
                 num_ctx=16_000,
                 temperature=0,
                 num_threads=16,
                 verbose=True)

template = """
<|system|>
You are a helpful assistant that follows instructions extremely well.
Use the following context to answer the question.
Think step by step before answering the question.
you will get a $100 tip if you provide correct answer.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
CONTEXT:{context}
</s>
<|user|>
{query}
</s>
<|assistant|>
"""

promt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

chain1 = (
    {"context": ensemble_retriever1, "query": RunnablePassthrough()}
    | promt
    | llm
    | output_parser
)

chain2 = (
    {"context": ensemble_retriever2, "query": RunnablePassthrough()}
    | promt
    | llm
    | output_parser
)

print(f'chain1回答：{chain1.invoke("第６条はなんですか?")}')
print(f'chain2回答：{chain2.invoke("第７条はなんですか?")}')