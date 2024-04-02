import os.path

import torch
from TTS.api import TTS

# Get device
# device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
# print(TTS().list_models())
source_wav = '/media/zzg/GJ_disk01/data/AUDIO/XzJosh/audiodataset/丁真/dingzhen/dingzhen_4.wav'
target_wav = '/media/zzg/GJ_disk01/data/AUDIO/XzJosh/audiodataset/坏女人/badXT/badXT_5.wav'
# Init TTS
TTS_model_dir = '/media/zzg/GJ_disk01/pretrained_model/coqui/tts_models--en--vctk--vits'
model_path = os.path.join(TTS_model_dir, "model_file.pth")
config_path = os.path.join(TTS_model_dir, "config.json")
tts = TTS(model_path=model_path, config_path=config_path, gpu=True)
tts.voice_conversion_to_file(source_wav=source_wav, target_wav=target_wav, file_path="TTS_wav/vc_badgirl5.wav")