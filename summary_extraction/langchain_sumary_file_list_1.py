import os

import dotenv

from langchain.chains import LLMChain, MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import AnalyzeDocumentChain
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_models import ChatOllama
from langchain.chat_models import ChatOpenAI
dotenv.load_dotenv()

def summarize(filename, model_name='llama3-70b-8192'):
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """
    print(f'#########################filename: {filename}##########################')
    template = """
        Document: {text}
        The above document contains a dialogue among several Japanese individuals.
        Starting with details about time, location, participants, and their relationships. 
        The conversation content follows in the format: Participant ID: Dialogue.
        You will generate concise, entity-dense summaries of the above document. 
        Also, generate summaries for each person's statements.
        Do not quote the original dialog.
        Do not discuss the guidelines for summarization in your response.
        Please answer in Japanese.
        
        Guidelines for summarization:
        - The first should be concise (3-5 sentences, ~60 words).
        - Make every word count. Do not fill with additional words which are not critical to summarize the original document.
        - Make space with fusion, compression, and removal of uninformative phrases.
        - The summaries should become highly dense and concise yet self-contained, i.e., easily understood without the article.
        - The conversation involves company names, people's names and other proper nouns. Please keep the original text and do not translate it into English.
        """
    prompt = PromptTemplate.from_template(template=template)

    # 定义文本加载器，指定读取的文件路径
    file_path = os.path.join(dir, filename)
    loader = TextLoader(file_path)
    docs_chat_jp = loader.load()
    print(f'Total length of documents: {len(docs_chat_jp[0].page_content)}')

    # 使用字符分割器确保正确分割日语文本
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=3000, chunk_overlap=10)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_RecursiveCharacterTextSplitter,
                                                   chunk_overlap=100,
                                                   separators=["\n\n", "\n", " ", ""])
    chunks = text_splitter.split_documents(docs_chat_jp)
    print(f'Total number of documents: {len(chunks)}')

    summary_chain = load_summarize_chain(llm,
                                         chain_type="map_reduce",
                                         map_prompt=prompt,
                                         combine_prompt=prompt,
                                         # document_variable_name="text",
                                         # Combine_document_variable_name="text",
                                         # Map_reduce_document_variable_name="text",
                                         # return_intermediate_steps=True,
                                         )
    # final_summary = summary_chain.run(docs_chat_jp)
    final_summary = summary_chain({"input_documents": docs_chat_jp}, return_only_outputs=True)["output_text"]
    # summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain, text_splitter=text_splitter)
    # final_summary = summarize_document_chain.run(docs_chat_jp)
    print(f'final summary is: {final_summary}')
    # 将处理后的内容写入新的文件中
    with open(summarized_file_path, 'w', encoding='utf-8') as summarized_file:
        summarized_file.write(final_summary)
    print(f'final summary length is: {len(final_summary)}')
    print(f'final summary: {final_summary}')
    print(f'Processed and saved: {filename}')

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # os.environ["NVIDIA_VISIBLE_DEVICES"] = "1"
    num_ctx_llama3_70b = 8192
    token_max_ReduceDocumentsChain = int(num_ctx_llama3_70b*0.8)
    chunk_size_RecursiveCharacterTextSplitter = int(token_max_ReduceDocumentsChain*0.9)
    # Initialize Groq Langchain chat object and conversation
    # ['llama3-70b-8192', 'llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    # llm = ChatGroq(model_name=model_name, temperature=0.0)
    # qwen2:7b-instruct-q5_K_M ,llama3:70b-instruct-q5_K_M,summerwind/japanese-starling-chatv:7b-q5_K_M,qwen2:1.5b-instruct-q5_K_M
    llm = ChatOllama(model="llama3:70b-instruct-q5_K_M",
                     num_ctx=num_ctx_llama3_70b,
                     temperature=0,
                     num_threads=16,
                     verbose=True)
    # llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True)

    dir = '/home/zzg/商业项目/清华/nucc/nucc'
    # dir = '/home/zzg/商业项目/清华/nucc/test'
    # 指定summarized目录路径
    summarized_dir = '/home/zzg/商业项目/清华/nucc/summarize_chain'
    # 如果summarized目录不存在，创建它
    if not os.path.exists(summarized_dir):
        os.makedirs(summarized_dir)
    # 遍历目录下的所有文件
    for filename in sorted(os.listdir(dir)):
        # 检查文件是否为txt文件
        if filename.endswith('.txt'):
            # 生成summarized目录中txt文件的完整路径
            summarized_file_path = os.path.join(summarized_dir, filename)
            if os.path.exists(summarized_file_path):
                print(f'Skipping {filename} as it has already been summarized.')
                continue
            # 生成文件的完整路径
            summarize(filename)