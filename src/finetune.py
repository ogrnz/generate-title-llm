import argparse
import json
from functools import partial

import evaluate
import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


def load_split_dataset(
    hf_dataset=None, data_path=None, test_size=0.15, valid_size=0.0588
):
    if data_path is not None:  # Load custom jsonl
        dataset = load_dataset("json", data_files=data_path)
    else:  # Import from hub
        dataset = load_dataset(hf_dataset)

    # To match t5 nomenclature
    dataset = dataset.rename_columns({"message": "text", "title": "summary"})

    train_test = dataset["train"].train_test_split(test_size=test_size)
    train_valid = train_test["train"].train_test_split(test_size=valid_size)
    dataset = DatasetDict(
        {
            "train": train_valid["train"],
            "test": train_test["test"],
            "valid": train_valid["test"],
        }
    )

    return dataset


def preprocess(examples, tokenizer, prefix="summarize: "):
    inputs = [prefix + example for example in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=examples["summary"], max_length=32, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred, tokenizer, metric):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def train_model(
    checkpoint,
    dataset,
    training_args: dict = {},
):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenized_dataset = dataset.map(
        partial(preprocess, tokenizer=tokenizer), batched=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=checkpoint)
    rouge = evaluate.load("rouge")

    # Actual training
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    training_args = Seq2SeqTrainingArguments(**training_args)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer, metric=rouge),
    )

    trainer.train()


def parse_json_config(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def main(args):
    default_training_args = {
        "output_dir": args.output_dir,
        "evaluation_strategy": "epoch",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "weight_decay": 0.01,
        "save_total_limit": 3,
        "num_train_epochs": 40,
        "predict_with_generate": True,
        "fp16": True,
    }

    # Override defaults with user provided config if available
    if args.training_config:
        user_args = parse_json_config(args.training_config)
        default_training_args.update(user_args)

    # Proceed with training
    dataset = load_split_dataset(args.hf_data)
    train_model(
        args.checkpoint,
        dataset,
        default_training_args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model on a dataset")
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="google-t5/t5-small",
        help="HF Model checkpoint or name",
    )
    parser.add_argument(
        "-d",
        "--hf-data",
        type=str,
        default="ogrnz/chat-titles",
        help="HF dataset to fine-tune on",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "-tc",
        "--training-config",
        type=str,
        help="Path to JSON file containing training configurations (`Seq2SeqTrainingArguments`)",
    )
    args = parser.parse_args()

    main(args)
