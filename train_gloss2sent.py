import os
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments
)

MODEL_NAME = "t5-small"
OUTDIR = "models/gloss2sent"

os.makedirs(OUTDIR, exist_ok=True)

print("✅ Loading dataset...")

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/gloss2sent_train.jsonl",
        "validation": "data/gloss2sent_valid.jsonl"
    }
)

print("✅ Loading tokenizer & model...")

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# ✅ ✅ IMPORTANT: ADD TASK PREFIX
TASK_PREFIX = "translate Gloss to English: "

def preprocess(example):
    # ✅ Input with prefix
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

print("✅ Starting training...")

args = TrainingArguments(
    output_dir=OUTDIR,
    per_device_train_batch_size=4,   # ✅ better gradients
    per_device_eval_batch_size=4,
    num_train_epochs=10,            # ✅ NEED MORE EPOCHS
    save_strategy="epoch",
    logging_steps=20,
    learning_rate=3e-4,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"]
)

trainer.train()

trainer.save_model(OUTDIR)
tokenizer.save_pretrained(OUTDIR)

print("✅ Gloss → English model trained successfully!")

# -----------------------------
# ✅ PROPER TESTING
# -----------------------------
print("\n✅ Testing model with sample inputs...")

sample_inputs = [
    "I GO MARKET MORNING",
    "I PLAY FOOTBALL WEEKEND",
    "I WRITE EMAIL TOMORROW"
]

for gloss in sample_inputs:
    test_input = TASK_PREFIX + gloss

    inputs = tokenizer(test_input, return_tensors="pt", truncation=True, padding=True)

    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Gloss: {gloss}")
    print(f"✅ Predicted English: {decoded}\n")




