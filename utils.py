import torchaudio

def resample_audio(audio_path, target_sample_rate=16000):
    # 加载音频文件
    audio, sample_rate = torchaudio.load(audio_path)
    if sample_rate!=target_sample_rate:
        # 重采样音频
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        resampled_audio = resampler(audio)
    else:
        resampled_audio=audio_path
    return resampled_audio