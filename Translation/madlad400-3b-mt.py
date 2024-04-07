import os.path

from transformers import T5ForConditionalGeneration, T5Tokenizer
from llama_cpp import Llama


model_name = '/media/zzg/GJ_disk01/pretrained_model/jbochi/madlad400-3b-mt'

# model = Llama(model_path=os.path.join(model_name, 'model-q4k.gguf'), device_map="auto", local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", local_files_only=True, weights_name='model-q4k.gguf')
tokenizer = T5Tokenizer.from_pretrained(model_name, local_files_only=True)

text = "<2zh> 毎日一本、映画を見るぞと意気込み、数日で挫折する。"
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
outputs = model.generate(input_ids=input_ids)

resutls = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f'resutls: {resutls}')