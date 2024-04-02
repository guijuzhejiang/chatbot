import os.path

import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
TTS_model = '/media/zzg/GJ_disk01/pretrained_model/coqui/XTTS-v2'
tts = TTS(model_path=TTS_model, config_path=os.path.join(TTS_model, "config.json"), progress_bar=True).to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
# wav = tts.tts(text="我是谁，我在哪，我干啥呢？", speaker_wav="/media/zzg/GJ_disk01/data/AUDIO/XzJosh/audiodataset/坏女人/badXT/badXT_5.wav", language="zh")
# Text to speech to a file
# tts.tts_to_file(text="我是谁，我在哪，我干啥呢？", speaker_wav="/media/zzg/GJ_disk01/data/AUDIO/XzJosh/audiodataset/坏女人/badXT/badXT_5.wav", language="zh", file_path="TTS_wav/badgirl.wav")
tts.tts_to_file(text="我是谁，我在哪，我干啥呢？", speaker_wav="/media/zzg/GJ_disk01/data/AUDIO/XzJosh/audiodataset/丁真/dingzhen/dingzhen_4.wav", language="zh", file_path="../TTS_wav/dingzhen.wav")
