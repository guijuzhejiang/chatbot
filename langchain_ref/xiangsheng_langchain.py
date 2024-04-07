from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
import nltk
nltk.download()

def url2news(url):
    text_split = RecursiveCharacterTextSplitter(separators=['正文','撰稿'], chunk_size=1000,chunk_overlap=20,length_function=len)
    loader = UnstructuredURLLoader([url])
    # loader = UnstructuredFileLoader("./example_data/state_of_the_union.txt")
    data = loader.load_and_split(text_splitter=text_split)
    return data[1:2]

def new2script(news):
    prompt_template = """总结这段新闻的内容:
    
    "{text}"
    
    一百五十字的总结："""
    chinese_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    #max_tokens default=256
    llm = OpenAI(max_tokens=1500)
    chain = load_summarize_chain(llm, chain_type='map_reduce', map_prompt=chinese_prompt)

    summary = chain.run(news)

    openaichat = ChatOpenAI(model_name='gpt-3.5-turbo')

    template = """\
        我将给你一段新闻的概括，请按照要求把这段新闻改写成郭德纲和于谦的对口相声剧本。
        
        新闻："{新闻}"
        要求："{要求}"
        {output_instructions}
    """
    parser = PydanticOutputParser(pydantic_object=XiangSheng)

    prompt = PromptTemplate(
        template=template,
        input_variables=['新闻', '要求'],
        partial_variables={'output_instructions': parser.get_format_instructions()}
    )
    msg =[HumanMessage(content=prompt.format(新闻=summary,要求="风趣幽默，十分讽刺，剧本对话角色为郭德纲和于谦，以他们的自我介绍为开头"))]

    res = openaichat(msg)

#[Line(character='郭德纲', content='大家好，我是郭德纲！'), Line(character='于谦', content='大家好，我是于谦！'), ...]
    xiangsheng = parser.parse(res.content)
    return xiangsheng

#随便一个新浪新闻
url = 'https://news.sina.cn/2024-04-03/detail-inaqqcmn3849125.d.html?vt=4&cid=56262&node_id=56262'

class Line(BaseModel):
    character: str = Field(description='说这句台词的角色名字')
    content: str = Field(description='台词的具体内容，其中不包括角色名字')

class XiangSheng(BaseModel):
    script: list[Line] = Field(description='一段相声的台词剧本')

def url2xiangsheng(url):
    doc = url2news(url)
    xiangsheng = new2script(doc)
    return xiangsheng
