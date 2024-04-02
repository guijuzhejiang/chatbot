import os.path

from datasets import load_dataset, DatasetDict, load_from_disk, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, get_scheduler
from data_process import DataCollatorSpeechSeq2SeqWithPadding
import evaluate
import torch


model_name = '/media/zzg/GJ_disk01/pretrained_model/drewschaub/whisper-large-v3-japanese-4k-steps'
# model_name = '/media/zzg/GJ_disk01/pretrained_model/vumichien/whisper-large-v2-jp'
data_path = '/home/zzg/data/Audio/JP'
data_path_train = os.path.join(data_path, 'train')
data_path_test = os.path.join(data_path, 'test')

common_voice = DatasetDict()
common_voice["train"] = load_from_disk(data_path_train)
common_voice["test"] = load_from_disk(data_path_test)
common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
print(common_voice)

#音频重采样到16kHz
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Japanese", task="transcribe")

processor = WhisperProcessor.from_pretrained(model_name, language="Japanese", task="transcribe")

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=8)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric_wer = evaluate.load("wer")
metric_cer = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
    cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}

model = WhisperForConditionalGeneration.from_pretrained(model_name,
                                                        # torch_dtype=torch.bfloat16,
                                                        torch_dtype=torch.float16,
                                                        load_in_8bit=False
                                                        )
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

lr=1e-5
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-v3-JP", # change to a repo name of your choice
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16, # increase by 2x for every 2x decrease in batch size
    learning_rate=lr,
    warmup_steps=500,
    max_steps=10000,
    gradient_checkpointing=True,
    fp16=True,
    # bf16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

processor.save_pretrained(training_args.output_dir)


# 2. 启用混合精度训练加速
# scaler = torch.cuda.amp.GradScaler()

# 5. 准备优化器和学习率调度器
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=training_args.max_steps,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    optimizers=(optimizer, lr_scheduler),
)

# 7. 使用混合精度训练
# model, optimizer, train_dataloader = trainer.amp.model_prepare(model, optimizer, common_voice["train"])
trainer.model = trainer.model.to(trainer.model.device)
trainer.train(resume_from_checkpoint=None)

# for epoch in range(training_args.num_train_epochs):
#     for batch in train_dataloader:
#         with torch.cuda.amp.autocast():
#             outputs = model(**batch)
#             loss = outputs.loss
#
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         optimizer.zero_grad()