from typing import List

from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import re
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
import tempfile
from dotenv import load_dotenv
load_dotenv()
from langchain.llms import HuggingFaceLLM


class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:  # 如果传入是pdf
            text = re.sub(r"\n{3,}", "\n", text)  # 将连续出现的3个以上换行符替换为单个换行符，从而将多个空行缩减为一个空行。
            text = re.sub('\s', ' ', text)  # 将文本中的所有空白字符（例如空格、制表符、换行符等）替换为单个空格
            text = text.replace("\n\n", "")  # 将文本中的连续两个换行符替换为空字符串
        sent_sep_pattern = re.compile(
            '([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  # 用于匹配中文文本中的句子分隔符，例如句号、问号、感叹号等
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list

def new2script(transcribed_minutes):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8')
    tmp_file.write(transcribed_minutes)
    tmp_file.close()
    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
    )
    # 切分文本
    # split_texts = text_splitter.split_text(transcribed_minutes)
    loader = TextLoader(tmp_file.name)
    split_texts = loader.load_and_split(text_splitter=text_splitter)
    print(f'documents:{len(split_texts)}')
    # 使用自定义且分类切分文本
    # textSplitter = ChineseTextSplitter(False)
    # split_texts = textSplitter.split_text(transcribed_minutes)

    prompt_template = """对下面这段会议内容进行总结归纳，总结出关键点:

    "{text}"

    总结："""
    chinese_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    # max_tokens default=256
    # llm = chatglm3(max_tokens=1500)
    #引入本地模型
    llm = ChatOpenAI(model_name='/media/zzg/GJ_disk01/pretrained_model/THUDM/chatglm3-6b', max_tokens=1000)
    #chain_type='refine,map_reduce'
    # chain = load_summarize_chain(llm, chain_type='map_reduce', verbose=True)
    chain = load_summarize_chain(llm, chain_type='map_reduce', map_prompt=chinese_prompt, verbose=True)

    summary = chain.run(split_texts)

    return summary

results = new2script("该项目是一个可以实现 __完全本地化__推理的知识库增强方案, 重点解决数据安全保护，私域化部署的企业痛点。 本开源方案采用Apache License，可以免费商用，无需付费。我们支持市面上主流的本地大语言模型和Embedding模型，支持开源的本地向量数据库。")

print(f'results:{results}')