import sys
import time

sys.path.append('TTS/vits')

import soundfile
import os
os.environ["PYTORCH_JIT"] = "0"
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import logging
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


class TTService():
    def __init__(self, TTS_model_dir = '/media/zzg/GJ_disk01/pretrained_model/coqui/XTTS-v2',
                 reference_wav='/media/zzg/GJ_disk01/data/AUDIO/XzJosh/audiodataset/坏女人/badXT/badXT_5.wav',
                 temperature=1,
                 speed=1.0,
                 language="zh"
                 ):
        logging.info('Initializing TTS Service')
        config_path = os.path.join(TTS_model_dir, "config.json")
        config = XttsConfig()
        config.load_json(config_path)
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=TTS_model_dir, use_deepspeed=True)
        self.model.cuda()
        # 一定注意参考音频的采样率为多少，建议统一重采样到一个数，比如24000,读入的时候设置load_sr=24000
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(audio_path=[reference_wav],
                                                                            sound_norm_refs=False, load_sr=44100)
        self.temperature = temperature
        self.speed = speed
        self.language = language

    def read(self, text):
        with torch.no_grad(),torch.cuda.amp.autocast():
            chunks = self.model.inference_stream(
                text=text,
                language=self.language,
                gpt_cond_latent=self.gpt_cond_latent,
                speaker_embedding=self.speaker_embedding,
                temperature=self.temperature,
                speed=self.speed,
            )
        wav_chuncks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                print(f"Time to first chunck: {time.time() - t0}")
            print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
            wav_chuncks.append(chunk)
        wav = torch.cat(wav_chuncks, dim=0)
        # wav = wav.squeeze().unsqueeze(0).cpu()
        return wav

    def read_save(self, text, filename):
        stime = time.time()
        wav = self.read(text)
        # soundfile.write(filename, au, sr)
        torchaudio.save(filename, wav.squeeze().unsqueeze(0).cpu(), 24000)
        logging.info('TTS Done, time used %.2f' % (time.time() - stime))




