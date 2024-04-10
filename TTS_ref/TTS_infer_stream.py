import os
import time
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("Loading model...")
TTS_model_dir = '/media/zzg/GJ_disk01/pretrained_model/coqui/XTTS-v2'
config_path = os.path.join(TTS_model_dir, "config.json")
reference_wav = '/media/zzg/GJ_disk01/data/AUDIO/XzJosh/audiodataset/坏女人/badXT/badXT_5.wav'
timeArray = time.localtime()
timeStr = time.strftime('%Y-%m-%d_%H-%M-%S', timeArray)
out_wav = f'../TTS_wav/test_{timeStr}.wav'
config = XttsConfig()
config.load_json(config_path)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=TTS_model_dir, use_deepspeed=True)
# model.load_checkpoint(config=config, checkpoint_dir=TTS_model_dir)
model.cuda()

print("Computing speaker latents...")
#一定注意参考音频的采样率为多少，建议统一重采样到一个数，比如24000,读入的时候设置load_sr=24000
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[reference_wav], sound_norm_refs=False, load_sr=44100)

print("Inference...")
t0 = time.time()
chunks = model.inference_stream(
    "我是艾露莎·史塔雷特，你可以叫我艾露莎。作为魔法骑士团的S级魔法师，我负责研究和开发新型的魔导器。我的发色是蓝色的，瞳色也是蓝色的。身高大约1.60米。根据官方设定，我现在22岁，生日是在2月2日，所以我是水瓶座。血型是O型。至于体重、三围等身体指标，由于没有具体的设定，所以无法提供。我在魔法骑士团中担任着非常重要的角色。",
    "zh",
    gpt_cond_latent,
    speaker_embedding,
    temperature=1,
    speed=1,
    enable_text_splitting=True
)

wav_chuncks = []
for i, chunk in enumerate(chunks):
    if i == 0:
        print(f"Time to first chunck: {time.time() - t0}")
    print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
    wav_chuncks.append(chunk)
wav = torch.cat(wav_chuncks, dim=0)
torchaudio.save(out_wav, wav.squeeze().unsqueeze(0).cpu(), 24000)