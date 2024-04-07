import os
import time

import torch
import gradio as gr
from faster_whisper import WhisperModel
from datetime import datetime
from utils import resample_audio


device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME_ASR_JP = '/media/zzg/GJ_disk01/pretrained_model/guillaumekln/faster-whisper-large-v2'
model_ASR_JP = WhisperModel(
    MODEL_NAME_ASR_JP,
    device=device,
    compute_type="float16",
    num_workers=4,
    local_files_only=True,
)
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")
def transcribe_faster_whisper(audio, language='ja', task="transcribe", beam_size=5):
    transcribe_start = datetime.now()
    #音频重采样到16000
    # resampled_audio = resample_audio(audio)
    # 设置initial_prompt
    initial_prompt = "これから日本語の音声を認識します。"
    segments, info = model_ASR_JP.transcribe(audio,
                                             language=language,
                                             task=task,
                                             beam_size=beam_size,
                                             initial_prompt=initial_prompt,
                                             word_timestamps=True,
                                             vad_filter=True,
                                             vad_parameters=dict(min_silence_duration_ms=500),
                                             )
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    text_list = []
    text_timestamp_list = []
    for segment in segments:
        text_timestamp = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
        print(text_timestamp)
        text_timestamp_list.append(text_timestamp)
        text_list.append(segment.text)
    transcribe_end = datetime.now()
    print(f'transcribe time is:{transcribe_end - transcribe_start}')
    return '\n'.join(text_timestamp_list)

def transcribe(audio, task):
    if audio is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
        # 针对Gradio麦克风录音，`audio`参数是一个元组，其中包含文件的临时路径
    else:
        # 如果不是元组，直接使用audio参数（这取决于Gradio版本和行为）
        audio_path = audio
    return transcribe_faster_whisper(audio_path, task=task)


output = gr.Textbox(label="Output", visible=True)
audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Record Audio")

file_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio file", streaming=True),
        # gr.Audio(sources=["upload", "microphone"], type="filepath", label="Record Audio"),
        gr.Radio(["transcribe", "translate"], label="Task", value="transcribe"),
    ],
    outputs=output,
    title="Transcribe Audio",
    description=("タスクを選択し、ボタンをクリックすると、マイク音声や長い音声入力を書き起こすことができます。"),
    allow_flagging="never",
)

demo = gr.Blocks()
with demo:
    gr.TabbedInterface([file_transcribe], ["Audio file"])
    demo.launch(server_name='0.0.0.0', server_port=8081, root_path='https://www.guijutech.com/asr')
