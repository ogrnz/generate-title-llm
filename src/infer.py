import json
from transformers import pipeline


def generate_title(message, summarizer, max_length=32):
    return summarizer(message, max_length=max_length, min_length=5)


def load_messages(file_path):
    messages = []
    with open(file_path, "r", encoding="utf-8") as f:
        for l in f:
            text = json.loads(l)["message"]
            messages.append(text)
    return messages


if __name__ == "__main__":
    messages = load_messages("./data/test.jsonl")
    summarizer = pipeline("summarization", model="./results/checkpoint-20000")

    messages = [
        "J'ai mal au genou après avoir repris mon programme de course. J'ai peur d'avoir trop poussé. Que dois-je faire?"
    ]
    title = generate_title(messages[0], summarizer, max_length=16)[0]["summary_text"]
    print(f"{title}:\n {messages[0]}\n")

    for message in messages[:20]:
        print("=" * 10)
        title = generate_title(message, summarizer, max_length=16)[0]["summary_text"]
        print(f"{title}:\n {message}\n")
        print("=" * 10)
