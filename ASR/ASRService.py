import logging
import time
from faster_whisper import WhisperModel


class ASRService():
    def __init__(self, device, MODEL_NAME_ASR_JP = '/home/zzg/workspace/pycharm/Whisper-Finetune/models/ct2/common_voice_16_1/whisper-large-v3-japanese-4k-steps/checkpoint-10000/'):
        logging.info('Initializing ASR Service...')
        self.model = WhisperModel(
            MODEL_NAME_ASR_JP,
            device=device,
            compute_type="float16",
            num_workers=4,
            local_files_only=True,
        )

    def infer(self, audio, language='ja', task='transcribe', beam_size=5, initial_prompt="これから日本語の音声を認識します。"):
        stime = time.time()
        # audio is a *.wav file
        segments, info = self.model.transcribe(audio, language=language, task=task, beam_size=beam_size, initial_prompt=initial_prompt)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        text = []
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            text.append(segment.text)
        result = ''.join(text)
        logging.info('ASR Result: %s. time used %.2f.' % (result, time.time() - stime))
        return result

if __name__ == '__main__':
    config_path = 'ASR/resources/config.yaml'

    service = ASRService(config_path)

    # print(wav_path)
    wav_path = 'ASR/test_wavs/0478_00017.wav'
    result = service.infer(wav_path)
    print(result)