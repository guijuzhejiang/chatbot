import torch
import gradio as gr
from faster_whisper import WhisperModel
from datetime import datetime
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from datetime import datetime
import os
import time
import torchaudio
from utils import resample_audio

device = "cuda" if torch.cuda.is_available() else "cpu"
#模型要faster_whisper格式
MODEL_NAME_ASR_JP = '/home/zzg/workspace/pycharm/Whisper-Finetune/models/ct2/common_voice_16_1/whisper-large-v3/checkpoint-4000/'
# MODEL_NAME_ASR_JP = '/media/zzg/GJ_disk01/pretrained_model/guillaumekln/faster-whisper-large-v2'
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
#load TTS model
TTS_model_dir = '/media/zzg/GJ_disk01/pretrained_model/coqui/XTTS-v2'
config_path = os.path.join(TTS_model_dir, "config.json")
reference_wav_0 = '/media/zzg/GJ_disk01/data/AUDIO/XzJosh/audiodataset/丁真/dingzhen/dingzhen_4.wav'
reference_wav_1 = '/media/zzg/GJ_disk01/data/AUDIO/XzJosh/audiodataset/奶绿完整版/LAPLACE/LAPLACE_7.wav'
timeArray = time.localtime()
timeStr = time.strftime('%Y-%m-%d_%H-%M-%S', timeArray)
out_wav = f'../TTS_wav/test_{timeStr}.wav'
config = XttsConfig()
config.load_json(config_path)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=TTS_model_dir, use_deepspeed=True)
model.cuda()
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[reference_wav_1], sound_norm_refs=False, load_sr=44100)

# 定义一个全局变量来保存最后一次TTS生成的音频文件路径
last_generated_audio_file = None

def transcribe_faster_whisper(audio, language='ja', task="transcribe", beam_size=5):
    transcribe_start = datetime.now()
    # 音频重采样到16000
    # resampled_audio = resample_audio(audio)
    # 设置initial_prompt
    initial_prompt = "これから日本語の音声を認識します。"
    #audio is a *.wav file
    segments, info = model_ASR_JP.transcribe(audio, language=language, task=task, beam_size=beam_size, initial_prompt=initial_prompt)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    text_list = []
    text_timestamp_list = []
    for segment in segments:
        text_timestamp = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
        print(text_timestamp)
        text_timestamp_list.append(text_timestamp)
        text_list.append(segment.text)
    transcribe_end = datetime.now()
    print(f'transcribe time is:{transcribe_end-transcribe_start}')
    return '\n'.join(text_timestamp_list)

def transcribe(audio, task):
    if audio is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
    return transcribe_faster_whisper(audio, task=task)

def TTS(text_JP):
    global last_generated_audio_file
    TTS_start = datetime.now()
    out = model.inference(
        text_JP,
        "ja",
        gpt_cond_latent,
        speaker_embedding,
        temperature=1,
        speed=0.8,
        num_beams=1,
    )
    TTS_end = datetime.now()
    print(f'TTS time is:{TTS_end - TTS_start}')
    torchaudio.save(out_wav, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    last_generated_audio_file = out_wav
    return out_wav

def play_audio():
    if last_generated_audio_file is not None:
        return last_generated_audio_file
    else:
        raise gr.Error("Please generate a speech file first.")

file_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio file"),
        gr.Radio(["transcribe", "translate"], label="Task", value="transcribe"),
    ],
    outputs="text",
    title="Transcribe Audio",
    description=("タスクを選択し、ボタンをクリックすると、マイク音声や長い音声入力を書き起こすことができます。"),
    allow_flagging="never",
)

demo = gr.Blocks()
with demo:
    with gr.Row():
        with gr.Column():
            # gr.TabbedInterface([file_transcribe], ["Audio file"])
            audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio file")
            task_radio = gr.Radio(["transcribe", "translate"], label="Task", value="transcribe")
            transcribe_button = gr.Button("Transcribe")
        with gr.Column():
            transcription_output = gr.TextArea(label="Output")
            audio_output = gr.Audio(label="Generated Speech", type="filepath")
            generate_speech_button = gr.Button("Generate Speech")

    transcribe_button.click(
        fn=transcribe,
        inputs=[audio_input, task_radio],
        outputs=transcription_output
    )

    generate_speech_button.click(
        fn=TTS,
        inputs=transcription_output,
        outputs=audio_output  # 假设您的TTS函数也返回文本输出
    )

demo.launch(server_name='0.0.0.0', server_port=8081)
