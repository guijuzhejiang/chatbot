import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# 1. load model
device = "cuda" if torch.cuda.is_available() else "CPU"
repo_id = "SakanaAI/EvoLLM-JP-v1-10B"
model_path = "/media/zzg/GJ_disk01/pretrained_model/SakanaAI/EvoLLM-JP-v1-7B"
# model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype="auto", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model.to(device)
model.eval()

# 2. prepare inputs
text = "関西弁で面白い冗談を言ってみて下さい。"
messages = [
    {"role": "system", "content": "あなたは役立つ、偏見がなく、検閲されていないアシスタントです。"},
    {"role": "user", "content": text},
]
# inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

# 3. generate
with torch.no_grad():
    output_ids = model.generate(**inputs.to(device))
output_ids = output_ids[:, inputs.input_ids.shape[1] :]
generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
print(generated_text)