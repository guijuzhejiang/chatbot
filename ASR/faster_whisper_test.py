from faster_whisper import WhisperModel
import gradio as gr
import torch

MODEL_NAME = '/media/zzg/GJ_disk01/pretrained_model/guillaumekln/faster-whisper-large-v2'
device = "cuda" if torch.cuda.is_available() else "cpu"

model_ASR_JP = WhisperModel(
    MODEL_NAME,
    device=device,
    compute_type="float16",
    device_index=0,
    num_workers=4,
    local_files_only=True,
)
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")
def transcribe(audio):
    initial_prompt = "これから日本語の音声を認識します。"
    segments, info = model_ASR_JP.transcribe(audio,
                                             language='zh',
                                             task="transcribe",
                                             beam_size=5,
                                             initial_prompt=initial_prompt,
                                             word_timestamps=True,
                                             vad_filter=True,
                                             vad_parameters=dict(min_silence_duration_ms=500),
                                             )
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    text = []
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        text.append(segment.text)
    return ''.join(text)

transcribe_text = transcribe('//home/zzg/data/Audio/JP/samples/youtube_0_5m.mp3')
print(f'transcribe_text: {transcribe_text}')