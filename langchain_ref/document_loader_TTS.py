from langchain.document_loaders import (PyPDFLoader,
                                        TextLoader,
                                        UnstructuredURLLoader,
                                        UnstructuredHTMLLoader,
                                        UnstructuredWordDocumentLoader,
                                        CSVLoader,
                                        UnstructuredPowerPointLoader
                                        )
from langchain_community.document_loaders.unstructured import UnstructuredFileIOLoader
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator, VectorStoreIndexWrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
from RealtimeTTS import TextToAudioStream, CoquiEngine
import time

def load_file(file_path):
    try:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith('doc') or file_path.endswith('docx'):
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_path.endswith('ppt') or file_path.endswith('pptx'):
            loader = UnstructuredPowerPointLoader(file_path)
        else:
            print(f'no file!!!')
            return
    except Exception:
        print(f"Load File Error")
        raise
    datas = loader.load()
    contents = ''
    for data in datas:
        contents += data.page_content
    return contents

def dummy_generator(long_text):
    yield long_text

def synthesize(generator, ref_wav_json):
    engine = CoquiEngine(voice=ref_wav_json,
                         language="zh",
                         speed=1.0,
                         length_penalty=1,
                         repetition_penalty=10.0,
                         stream_chunk_size=40,
                         overlap_wav_len=4096,
                         use_deepspeed=True,)  # using a chinese cloning reference gives better quality
    # stream = TextToAudioStream(engine)
    stream = TextToAudioStream(engine, log_characters=True, tokenizer="stanza", language="zh")
    # stream = TextToAudioStream(engine, log_characters=True, tokenizer="stanza", language="multilingual")

    print("Starting to play stream")
    stream.feed(generator)

    timeArray = time.localtime()
    timeStr = time.strftime('%Y-%m-%d_%H-%M-%S', timeArray)
    out_wav = f'TTS_wav/RealtimeTTS_stream_{timeStr}.wav'
    # ❗ use these for chinese: minimum_sentence_length = 2, minimum_first_fragment_length = 2, tokenizer="stanza", language="zh", context_size=2
    stream.play(
        minimum_sentence_length=2,
        minimum_first_fragment_length=2,
        output_wavfile=out_wav,
        # on_sentence_synthesized=lambda sentence:
        # print("Synthesized: " + sentence),
        tokenizer="stanza",
        language="zh",
        context_size=2,
        # muted=True,
    )
    # with open(f"{filename}.txt", "w", encoding="utf-8") as f:
    #     f.write(stream.text())
    engine.shutdown()


if __name__ == '__main__':
    local_persist_path = './vector_store'
    # file_path = '/home/zzg/商业项目/清华/在线AI绘图应用.pdf'
    # file_path = '/home/zzg/商业项目/清华/日语ASR技术资料.docx'
    # file_path = '/home/zzg/商业项目/清华/ソフトウエア基本契約書.doc'
    # file_path = '/home/zzg/商业项目/清华/KDDI関連内容.pptx'
    # file_path = '/home/zzg/商业项目/王老师/多莫态_video_speech_text_20230101/easynlp.txt'
    file_path = 'langchain_ref/document_test.txt'
    # 加载PDF文件并存储在本地向量数据库
    contents = load_file(file_path)

    generator = dummy_generator(contents)
    #加载TTS识别
    ref_wav_json_dir = '/home/zzg/data/Audio/reference_wav/json'
    ref_wav_json_name = 'ae3175100a4e4982aa0fe286bebad25e.json'
    ref_wav_json = os.path.join(ref_wav_json_dir, ref_wav_json_name)
    synthesize(generator, ref_wav_json)