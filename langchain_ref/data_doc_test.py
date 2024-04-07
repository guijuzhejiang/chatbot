from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
#chromadb是一个本地的向量存储解决方案


loader = PyPDFLoader('/home/zzg/商业项目/清华/在线AI绘图应用.pdf')
# pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,                 #每个块的最大长度
    chunk_overlap=20,               #块与块之间的交叠长度
    length_function=len,
)

#返回Documents对象
pages = loader.load_and_split(text_splitter=text_splitter)

# print(pages)
#
# embedding_model = OpenAIEmbeddings()
# embeddings = embedding_model.embed_documents([pages[4].page_content])
#
# print(len(embeddings))
# print(embeddings[0])

db = Chroma.from_documents(pages, OpenAIEmbeddings())

query='AI'
docs = db.similarity_search(query)