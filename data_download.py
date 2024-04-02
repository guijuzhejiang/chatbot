import os.path

from datasets import load_dataset, DatasetDict, load_from_disk, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments


data_repo = 'mozilla-foundation/common_voice_16_1'
data_name = data_repo.split('/')[1]
data_path = '/home/zzg/data/Audio/JP'
data_path_train = os.path.join(data_path, data_name, 'train')
data_path_test = os.path.join(data_path, data_name, 'test')
print(f'data_path:{data_path}')
print(f'data_name:{data_name}')
print(f'data_path_train:{data_path_train}')
print(f'data_path_test:{data_path_test}')

common_voice = DatasetDict()
common_voice["train"] = load_dataset(data_repo, "ja", split="train+validation", use_auth_token='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')
common_voice["train"].save_to_disk(data_path_train)
# common_voice["train"] = load_from_disk(data_path_train)

common_voice["test"] = load_dataset(data_repo, "ja", split="test", use_auth_token='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')
common_voice["test"].save_to_disk(data_path_test)
# common_voice["test"] = load_from_disk(data_path_test)

print(common_voice)