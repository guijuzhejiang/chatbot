import os

import dotenv
from datetime import datetime
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
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


dotenv.load_dotenv()

def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """
    llm = ChatOllama(model="llama3.1:8b-instruct-q5_K_M", temperature=0)
    # llm = ChatOllama(model="llama3.1:70b-instruct-q2_K", temperature=0)

    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    # Load documents from the URLs
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # Initialize a text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    # Split the documents into chunks
    doc_splits = text_splitter.split_documents(docs_list)

    # Add the document chunks to the "vector store" using OpenAIEmbeddings
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=OllamaEmbeddings(model="nomic-embed-text")
    )
    retriever = vectorstore.as_retriever(k=4)

    # Define a tool, which we will connect to our agent
    def retrieve_documents(query: str) -> list:
        """Retrieve documents from the vector store based on the query."""
        return retriever.invoke(query)

    web_search_tool = TavilySearchResults()

    def web_search(query: str) -> list:
        """Run web search on the question."""
        web_results = web_search_tool.invoke({"query": query})
        return [
            Document(page_content=d["content"], metadata={"url": d["url"]})
            for d in web_results
        ]

    # Tool list
    tools = [retrieve_documents, web_search]
if __name__ == "__main__":
    main()





