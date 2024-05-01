import json
from transformers import AutoTokenizer, pipeline


def generate_title(message, summarizer, max_length=16):
    return summarizer(message, max_length=max_length, min_length=3)


def load_messages(file_path):
    messages = []
    with open(file_path, "r", encoding="utf-8") as f:
        for l in f:
            text = json.loads(l)["message"]
            messages.append(text)
    return messages


if __name__ == "__main__":
    messages = load_messages("./data/test.jsonl")

    checkpoint = "./results/checkpoint-20000"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    summarizer = pipeline("summarization", model=checkpoint, tokenizer=tokenizer)

    for message in messages:
        print("=" * 10)
        title = generate_title(message, summarizer)[0]["summary_text"]
        print(f"{title}:\n {message}\n")
        print("=" * 10)
