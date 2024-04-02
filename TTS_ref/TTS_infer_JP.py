import os
import time

import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from datetime import datetime


print("Loading model...")
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

print("Computing speaker latents...")
#一定注意参考音频的采样率为多少，建议统一重采样到一个数，比如24000,读入的时候设置load_sr=24000
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[reference_wav_1], sound_norm_refs=False, load_sr=44100)

print("Inference...")
time_start=datetime.now()
out = model.inference(
    "いらすとやは季節のイベント・動物・子供などのかわいいイラストが沢山見つかるフリー素材サイトです。",
    "ja",
    gpt_cond_latent,
    speaker_embedding,
    temperature=1,
    speed=0.8,
    num_beams=1,
)
time_end=datetime.now()
print(f"Inference time is :{time_end-time_start}")
torchaudio.save(out_wav, torch.tensor(out["wav"]).unsqueeze(0), 24000)