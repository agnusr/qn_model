# src/train_template.py
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import argparse

def main(args):
    model_name = args.model_name  # e.g., "t5-small"
    out_dir = args.output_dir     # e.g., "models/gloss2sent"
    train_file = args.train_file
    valid_file = args.valid_file

    print("Loading dataset...")
    data_files = {"train": train_file, "validation": valid_file}
    ds = load_dataset("json", data_files=data_files)

    print("Loading tokenizer and model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    max_len = args.max_length

    def preprocess(examples):
        # tokenization
        inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=max_len)
        labels = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=max_len)
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        predict_with_generate=True,
        fp16=False,   # set True if GPU with CUDA and mixed precision supported
        push_to_hub=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
    )

    trainer.train()
    print("Saving model to", out_dir)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="t5-small")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()
    main(args)