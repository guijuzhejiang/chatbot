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
root = '/home/zzg/商业项目/外语对话练习/reference_audio/'
sex = 'women'
wav_name = 'household_10S.wav'
reference_wav = os.path.join(root, sex, wav_name)
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
# gpt_cond_latent_44100, speaker_embedding_44100 = model.get_conditioning_latents(audio_path=[reference_wav], sound_norm_refs=False, load_sr=44100)
gpt_cond_latent_22050, speaker_embedding_22050 = model.get_conditioning_latents(audio_path=[reference_wav], sound_norm_refs=False, load_sr=22050)

print("Inference...")
time_start=datetime.now()
text="""
    Currently using roughly 10S of audio as a reference doesn't feel like it works very well. How long of a reference should I use to get better results?
    """
# out_44100 = model.inference(
#     text,
#     "en",
#     gpt_cond_latent_44100,
#     speaker_embedding_44100,
#     temperature=0.1,
#     speed=1,
#     num_beams=1,
#     enable_text_splitting=True
# )
# time_end=datetime.now()
# print(f"Inference time is :{time_end-time_start}")
# out_wav_44100 = f'../TTS_wav/test_44100_{timeStr}.wav'
# torchaudio.save(out_wav_44100, torch.tensor(out_44100["wav"]).unsqueeze(0), 24000)

out_22050 = model.inference(
    text,
    "en",
    gpt_cond_latent_22050,
    speaker_embedding_22050,
    temperature=0.7,
    speed=1,
    num_beams=1,
    enable_text_splitting=True
)
out_wav_22050 = f'../TTS_wav/test_22050_{timeStr}.wav'
torchaudio.save(out_wav_22050, torch.tensor(out_22050["wav"]).unsqueeze(0), 24000)