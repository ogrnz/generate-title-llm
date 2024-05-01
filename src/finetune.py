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


def load_split_dataset(data_path, test_size=0.15, valid_size=0.0588):
    dataset = load_dataset("json", data_files=data_path)

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
    output_dir="./results",
):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenized_dataset = dataset.map(
        partial(preprocess, tokenizer=tokenizer), batched=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=checkpoint)
    rouge = evaluate.load("rouge")

    # Actual training
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir, evaluation_strategy="epoch", **training_args
    )

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


if __name__ == "__main__":
    dataset = load_split_dataset("./data/dataset.jsonl")
    checkpoint = "google-t5/t5-small"

    training_args = {
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "weight_decay": 0.01,
        "save_total_limit": 3,
        "num_train_epochs": 40,
        "predict_with_generate": True,
        "fp16": True,
    }
    train_model(
        checkpoint, dataset, training_args=training_args, output_dir="./results"
    )
