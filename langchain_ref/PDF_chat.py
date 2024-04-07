from langchain.document_loaders import PyPDFLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator, VectorStoreIndexWrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

local_persist_path = './vector_store'
pdf_path = '/home/zzg/商业项目/清华/在线AI绘图应用.pdf'

def get_index_path(index_name):
    return os.path.join(local_persist_path, index_name)

# 加载PDF文件并存储在本地向量数据库
def load_pdf_and_save_to_index(file_path, index_name):
    loader = PyPDFLoader(file_path)
    index = VectorstoreIndexCreator(vectorstore_kwargs={'persist_directory': get_index_path(index_name)}).from_loaders([loader])
    index.vectorstore.persist()
    # anser = index.query_with_sources('在线绘图应用能做什么？', chain_type='map_reduce')
    # print(anser)


# 加载本地向量数据库
def load_index(index_name):
    index_path = get_index_path(index_name)
    embedding =OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory=index_path,
        embedding_function=embedding,
    )
    return VectorStoreIndexWrapper(vectorstore=vectordb)

#查询向量数据库
def query_index_lc(index, query):
    ans = index.query_with_sources(query, chain_type='map_reduce')
    return ans['answer']

# 加载PDF文件并存储在本地向量数据库
# load_pdf_and_save_to_index(pdf_path, 'test')

# 加载本地向量数据库
# index = load_index('test')