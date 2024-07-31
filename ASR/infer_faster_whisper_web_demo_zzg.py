import argparse
import asyncio
import json
import os
import shutil
import time

import numpy as np
import torch
import uuid
import wave

import gradio as gr
from faster_whisper import WhisperModel
from datetime import datetime

from gradio import processing_utils
from pydub import AudioSegment

from src.filters import japanese_stream_filter
# from utils import resample_audio
from src.asr.faster_whisper_asr import language_codes
from src.audio_utils import save_audio_to_file
from src.vad.vad_interface import VADInterface
from src.asr.asr_factory import ASRFactory
from src.vad.vad_factory import VADFactory
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

buf_center = {}
chunk_length_seconds = 3
sampling_rate = 16000
samples_width = 2


def parse_args():
    parser = argparse.ArgumentParser(description="VoiceStreamAI Server: Real-time audio transcription using self-hosted Whisper and WebSocket")
    parser.add_argument("--vad-type", type=str, default="pyannote", help="Type of VAD pipeline to use (e.g., 'pyannote')")
    parser.add_argument("--vad-args", type=str, default='{"auth_token": "huggingface_token", "model_name": "/media/zzg/GJ_disk01/pretrained_model/pyannote/segmentation-3.0/pytorch_model.bin"}', help="JSON string of additional arguments for VAD pipeline")
    parser.add_argument("--asr-type", type=str, default="faster_whisper", help="Type of ASR pipeline to use (e.g., 'whisper')")
    parser.add_argument("--asr-args", type=str, default='{"model_size": "/media/zzg/GJ_disk01/pretrained_model/guillaumekln/faster-whisper-large-v2"}', help="JSON string of additional arguments for ASR pipeline")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for the WebSocket server")
    parser.add_argument("--port", type=int, default=8765, help="Port for the WebSocket server")
    return parser.parse_args()

args = parse_args()

vad_args = json.loads(args.vad_args)
asr_args = json.loads(args.asr_args)

vad_pipeline = VADFactory.create_vad_pipeline(args.vad_type, **vad_args)
asr_pipeline = ASRFactory.create_asr_pipeline(args.asr_type, **asr_args)

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
# model = WhisperModel(model_size, device="cuda", compute_type="int8_int8")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")
def transcribe_faster_whisper(audio, language='ja', task="transcribe", beam_size=5):
    transcribe_start = datetime.now()
    #音频重采样到16000
    # resampled_audio = resample_audio(audio)
    # 设置initial_prompt，最后写上标点符号，有助于生成标点
    # initial_prompt = "これから日本語の音声を認識します。"
    initial_prompt = "日本語での会議の記録だ。"
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
    # return "test"


def transcribe(audio, task):
    if audio is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
        # 针对Gradio麦克风录音，`audio`参数是一个元组，其中包含文件的临时路径
    else:
        # 如果不是元组，直接使用audio参数（这取决于Gradio版本和行为）
        audio_path = audio
    if task == 'translate':
        language = 'zh'
    else:
        language = 'ja'
    return transcribe_faster_whisper(audio_path, language=language)


def process_audio_async(audio_data, cid, lang, sp):
    start = time.time()

    # VAD
    audio_dir_path = f"audio_data/{cid}"
    os.makedirs(audio_dir_path, exist_ok=True)
    audio_file_path = os.path.join(audio_dir_path, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.wav')
    # merge_wav_files(audio_data, audio_file_path)
    wav_st = datetime.now()
    print(f"wav st: {str(wav_st)}")
    processing_utils.audio_to_file(
        sp, np.concatenate(audio_data), audio_file_path, format="wav"
    )
    print(f"wav merge elapsed: {str({datetime.now() - wav_st})}")
    # with wave.open(audio_file_path, 'wb') as wav_file:
    #     wav_file.setnchannels(1)  # Assuming mono audio
    #     wav_file.setsampwidth(samples_width)
    #     wav_file.setframerate(sampling_rate)
    #     wav_file.writeframes(audio_data)
    vad_st = datetime.now()
    vad_results = vad_pipeline.vad_pipeline(audio_file_path)
    vad_segments = []
    print(f"vad_results: {len(vad_results)}")
    print(f"vad_elapsed: {datetime.now() - vad_st}")

    if len(vad_results) > 0:
        vad_segments = [
            {"start": segment.start, "end": segment.end, "confidence": 1.0}
            for segment in vad_results.itersegments()
        ]

    if len(vad_segments) == 0:
        buf_center[cid]['data'].clear()
        buf_center[cid]['data_len'] = 0
        buf_center[cid]['buf_data_len'] = 0
        return

    # ASR
    last_segment_should_end_before = ((buf_center[cid]['buf_data_len'] / (sp * 1)) - 0.1)
    if vad_segments[-1]['end'] < last_segment_should_end_before:

        # transcription = await asr_pipeline.transcribe(self.client)
        language = lang
        asr_st = datetime.now()
        segments, info = asr_pipeline.asr_pipeline.transcribe(audio_file_path,
                                                      word_timestamps=True,
                                                      language=language_codes[language],
                                                      # initial_prompt=initial_prompt,
                                                      vad_filter=True,
                                                      vad_parameters=dict(min_silence_duration_ms=500)
                                                      )
        print(f"asr_elapsed: {datetime.now() - asr_st}")

        segments = list(segments)  # The transcription will actually run here.
        flattened_words = [word for segment in segments for word in segment.words]
        transcription = {
            "language": info.language,
            "language_probability": info.language_probability,
            "text": ''.join([s.text.strip() for s in segments]),
            "words":
                [
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in
                    flattened_words
                ]
        }

        buf_center[cid]['data'].clear()
        buf_center[cid]['buf_data_len'] = 0
        if transcription['text'] != '':
            end = time.time()
            transcription['processing_time'] = end - start
            print(f"processing_time: {transcription['processing_time']}")
            return transcription



def get_wav_file_size(wav_file):
    """
    获取 WAV 文件的总字节数

    Args:
        wav_file (str): WAV 文件的路径

    Returns:
        int: WAV 文件的总字节数
    """
    with wave.open(wav_file, 'r') as wav:
        num_frames = wav.getnframes()
        frame_width = wav.getsampwidth()
        return num_frames * frame_width


def mic_interface(*args, **kwargs):
    print("mic_interface")
    print(args)
    print(kwargs)


def audio_stream(*args, **kwargs):
    print("audio_stream")
    print(args)
    print(kwargs)
    if args and args[0]:
        audio_datas, lang, task, client_id = args
        sample_rate, data = audio_datas
        # sampling_rate=sample_rate
        if client_id in buf_center.keys():
            buf_center[client_id]['data'].append(data)
            buf_center[client_id]['data_len'] = buf_center[client_id]['data_len'] + len(data)
            buf_center[client_id]['buf_data_len'] = buf_center[client_id]['buf_data_len'] + len(data)

        else:
            buf_center[client_id] = {}
            buf_center[client_id]['texts'] = ''
            buf_center[client_id]['data'] = [data]
            buf_center[client_id]['data_len'] = len(data)
            buf_center[client_id]['buf_data_len'] = len(data)

        # chunk_length_in_bytes = chunk_length_seconds * sampling_rate * samples_width
        if buf_center[client_id]['data_len']/sample_rate/data.dtype.itemsize >= chunk_length_seconds:
        # audio_data = np.concatenate(buf_center[client_id]['data'])
        # audio = AudioSegment(
        #     audio_data.tobytes(),
        #     frame_rate=sample_rate,
        #     sample_width=audio_data.dtype.itemsize,
        #     channels=(1 if len(audio_data.shape) == 1 else audio_data.shape[1]),
        # )
        # if audio.duration_seconds > chunk_length_seconds:
            # loop = asyncio.get_event_loop()
            # future = asyncio.ensure_future(process_audio_async(buf_center[client_id]['data'], client_id, lang))
            # res = loop.run_until_complete(future)

            st = datetime.now()
            print(f"start: {str(st)}")
            res = process_audio_async(buf_center[client_id]['data'], client_id, lang, sample_rate)
            print(f"end: {str(datetime.now()-st)}")

            buf_center[client_id]['data_len'] = 0

            if res and 'words' in res.keys() and len(res['words']) > 0:
                words = japanese_stream_filter(''.join([w['word'] for w in res['words']]) + '\n')

                if 'texts' in buf_center[client_id].keys():
                    buf_center[client_id]['texts'] += words
                else:
                    buf_center[client_id]['texts'] = words

        return buf_center[client_id]['texts']


def audio_infer(*args, **kwargs):
    print("audio_stream")
    print(args)
    print(kwargs)
    if args and args[0]:
        audio_datas, lang, task, client_id = args
        sample_rate, data = audio_datas
        buf_center[client_id] = {"data": [data]}
        st = datetime.now()
        print(f"start: {str(st)}")
        res = process_audio_async(buf_center[client_id]['data'], client_id, lang, sample_rate)
        print(f"end: {str(datetime.now()-st)}")

        if res and 'words' in res.keys() and len(res['words']) > 0:
            words = japanese_stream_filter(''.join([w['word'] for w in res['words']]) + '\n')

            if 'texts' in buf_center[client_id].keys():
                buf_center[client_id]['texts'] += words
            else:
                buf_center[client_id]['texts'] = words

            buf_center[client_id]['data'].clear()

            return buf_center[client_id]['texts']


def merge_wav_files(input_files, output_file):
    """
    合并多个 WAV 文件为一个文件

    Args:
        input_files (list): 要合并的 WAV 文件路径列表
        output_file (str): 合并后的 WAV 文件路径
    """
    # 打开第一个 WAV 文件获取参数
    with wave.open(input_files[0], 'r') as wav:
        params = wav.getparams()

    # 创建输出 WAV 文件
    with wave.open(output_file, 'w') as output:
        output.setparams(params)

        # 将所有 WAV 文件的数据写入输出文件
        for input_file in input_files:
            with wave.open(input_file, 'r') as wav:
                output.writeframes(wav.readframes(wav.getnframes()))


if __name__ == '__main__':
    shutil.rmtree("audio_data", ignore_errors=True)
    text_file_output = gr.Textbox(label="Output", elem_classes="text_output", visible=True, scale=1, lines=20, autoscroll=True)
    audio_file_input = gr.Audio(sources=["upload"], type="filepath", label="Record Audio", streaming=False)

    file_transcribe = gr.Interface(
        fn=transcribe,
        inputs=[
            audio_file_input,
            gr.Radio(["transcribe", "translate"], label="Task", value="transcribe"),
        ],
        outputs=text_file_output,
        title="Transcribe Audio",
        description=("タスクを選択し、ボタンをクリックすると、マイク音声や長い音声入力を書き起こすことができます。"),
        allow_flagging="never",
    )

    # audio_mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Record Audio",
    #                            streaming=True,
    #                            waveform_options={"sample_rate": 16000})
    # task_mic_input = gr.Radio(["transcribe", "translate"], label="Task", value="transcribe")
    # client_id_mic_input = gr.Text(str(uuid.uuid4()), visible=False)
    # lang_mic_input = gr.Radio(language_codes.keys(), label="lang", value="japanese", visible=False)
    # text_mic_output = gr.TextArea(label="Output", elem_classes="text_output", visible=True, scale=1,
    #                               lines=20,
    #                               autoscroll=True)

    # mic_transcribe = gr.Interface(
    #     fn=audio_stream,
    #     inputs=[
    #         audio_mic_input,
    #         lang_mic_input,
    #         task_mic_input,
    #         client_id_mic_input
    #     ],
    #     outputs=text_mic_output,
    #     title="Transcribe Audio",
    #     description=("タスクを選択し、ボタンをクリックすると、マイク音声や長い音声入力を書き起こすことができます。"),
    #     allow_flagging="never",
    #     live=True
    # )

    def clear(id):
        print(id)
        return gr.update("asd")

    with gr.Blocks(fill_height=True, title="Transcribe Audio") as mic_demo:
        client_id = str(uuid.uuid4())

        with gr.Row():
            gr.Markdown("タスクを選択し、ボタンをクリックすると、マイク音声や長い音声入力を書き起こすことができます。")
        with gr.Row():
            # input
            with gr.Column(scale=1):
                audio_mic_input = gr.Audio(sources=["microphone"], type="numpy", label="Record Audio",
                                           streaming=True,
                                           waveform_options={"sample_rate": 16000})
                task_mic_input = gr.Radio(["transcribe", "translate"], label="Task", value="transcribe")
                client_id_mic_input = gr.Text(client_id, visible=False)
                lang_mic_input = gr.Radio(language_codes.keys(), label="lang", value="japanese", visible=False)

            with gr.Column(scale=1):
                text_mic_output = gr.TextArea(label="Output", elem_classes="text_output", visible=True, scale=1,
                                              lines=20,
                                              elem_id=client_id,
                                              autoscroll=True)

                clear_btn = gr.Button("Clear", visible=False)
                clear_btn.click(fn=clear, inputs=client_id_mic_input, outputs=text_mic_output)

        audio_mic_input.stream(audio_stream,
                               inputs=[
                                   audio_mic_input,
                                   lang_mic_input,
                                   task_mic_input,
                                   client_id_mic_input
                               ],
                               outputs=text_mic_output, )
        # audio_mic_input.stop_recording(audio_infer,
        #                        inputs=[
        #                            audio_mic_input,
        #                            lang_mic_input,
        #                            task_mic_input,
        #                            client_id_mic_input
        #                        ],
        #                        outputs=text_mic_output, )

    with gr.Blocks(fill_height=True, css="style.css") as demo:
        gr.TabbedInterface([file_transcribe, mic_demo], ["Audio file", "Microphone"])
        # audio_input.stream(audio_stream, inputs=audio_input, outputs=[text_output])
        # audio_input.upload(file_upload, inputs=audio_input, outputs=[text_output])

    # demo.launch(server_name='0.0.0.0', server_port=8082, root_path='https://www.guijutech.com/asr')
    demo.launch(server_name='0.0.0.0', server_port=8082)
