import json
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import os
from glob import glob

TTS_model_dir = '/media/zzg/GJ_disk01/pretrained_model/coqui/XTTS-v2'
config_path = os.path.join(TTS_model_dir, "config.json")
config = XttsConfig()
config.load_json(config_path)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=TTS_model_dir, use_deepspeed=True)
model.cuda()

dir_root = '/home/zzg/data/Audio/reference_wav'
subdir_wav = 'wav'
subdir_json = 'json'
dir_wav = os.path.join(dir_root, subdir_wav)
for reference_wav in glob(os.path.join(dir_wav, '*.wav')):

    json_save_path = os.path.join(dir_root, subdir_json, os.path.basename(reference_wav).split('.')[0] +'.json')
    if not os.path.exists(json_save_path):
        # 一定注意参考音频的采样率为多少，建议统一重采样到一个数，比如24000,读入的时候设置load_sr=24000
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[reference_wav],
                                                                            sound_norm_refs=False, load_sr=44100)
        json_ob = {"gpt_cond_latent": gpt_cond_latent.squeeze().tolist(), "speaker_embedding": speaker_embedding.squeeze().tolist()}
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(json_ob, f, ensure_ascii=False)
            print(f'saved {json_save_path}')