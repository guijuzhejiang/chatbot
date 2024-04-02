import torch
import gradio as gr
import time
import torchaudio
from ASR import ASRService
from Chat import ChatService
from TTS_ref import TTService
import logging


class ChatBot():
    def __init__(self):
        logging.info('Initializing Server...')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # ASR
        self.ASR = ASRService.ASRService(self.device)
        # CHAT
        self.Chat = ChatService.ChatService(LANG="CN")
        # TTS
        self.TTS = TTService.TTService()


    def chat(self, audio):
        timeArray = time.localtime()
        timeStr = time.strftime('%Y-%m-%d_%H-%M-%S', timeArray)
        out_wav = f'../TTS_wav/test_{timeStr}.wav'
        ask_text = self.ASR.infer(audio)
        reply_text = self.Chat.predict(ask_text)
        wav = self.TTS.read(reply_text)
        torchaudio.save(out_wav, wav, 24000)
        return out_wav, reply_text


# 初始化ChatBot实例
chat_bot = ChatBot()

# 创建Gradio界面
file_transcribe = gr.Interface(
    fn=chat_bot.chat,
    inputs=[
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio file"),
    ],
    outputs="text",
    title="Transcribe Audio",
    description=("タスクを選択し、ボタンをクリックすると、マイク音声や長い音声入力を書き起こすことができます。"),
    allow_flagging="never",
)

demo = gr.Blocks()
with demo:
    gr.TabbedInterface([file_transcribe], ["Audio file"])

demo.launch(server_name='0.0.0.0', server_port=8081)
