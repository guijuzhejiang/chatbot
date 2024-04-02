from faster_whisper import WhisperModel
import gradio as gr
import torch

MODEL_NAME = '/media/zzg/GJ_disk01/pretrained_model/zh-plus/faster-whisper-large-v2-japanese-5k-steps'
device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperModel(
    MODEL_NAME,
    device=device,
    compute_type="float16",
    device_index=1,
    num_workers=4,
    local_files_only=True,
)
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")
def transcribe(audio):
    segments, info = model.transcribe(audio, beam_size=5)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    text = []
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        text.append(segment.text)
    return ''.join(text)

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath"),
    outputs="text",
    title="Whisper Small Hindi",
    description="Realtime demo for Hindi speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()