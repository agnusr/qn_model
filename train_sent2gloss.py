from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

MODEL_NAME = "t5-small"
OUTDIR = "models/sent2gloss"

print("✅ Loading tokenizer & model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

print("✅ Loading dataset...")
dataset = load_dataset("json", data_files={
    "train": "data/sent2gloss_train.jsonl",
    "validation": "data/sent2gloss_valid.jsonl"
})

# ✅ ✅ TASK PREFIX (VERY IMPORTANT)
TASK_PREFIX = "translate English to Gloss: "

def preprocess(example):
    input_text = TASK_PREFIX + example["input"]

    input_enc = tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=128
    )

    target_enc = tokenizer(
        example["output"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

print("✅ Tokenizing dataset...")
tokenized_data = dataset.map(
    preprocess,
    remove_columns=dataset["train"].column_names
)

training_args = TrainingArguments(
    output_dir=OUTDIR,
    learning_rate=3e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,     # ✅ MUST be higher
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=20,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"]
)

print("✅ Training started...")
trainer.train()

model.save_pretrained(OUTDIR)
tokenizer.save_pretrained(OUTDIR)

print("✅ SENT → GLOSS MODEL TRAINED SUCCESSFULLY ✅")



