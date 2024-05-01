import argparse
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


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    summarizer = pipeline("summarization", model=args.checkpoint, tokenizer=tokenizer)

    if args.message:  # Directly use the message string provided
        messages = [args.message]
    else:  # Load messages from file if path is provided
        messages = load_messages(args.file_path)

    for message in messages:
        print("=" * 10)
        title = generate_title(message, summarizer)[0]["summary_text"]
        print(f"{title}:\n{message}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate summaries from texts using a transformer model."
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="./results/checkpoint-20000",
        help="Model checkpoint path or compatible huggingface model url",
    )
    parser.add_argument(
        "-f",
        "--file-path",
        type=str,
        help="(optional) Path to a file containing JSONL messages",
    )
    parser.add_argument(
        "-m",
        "--message",
        type=str,
        help="(optional) Direct message string to summarize",
    )
    args = parser.parse_args()

    if not args.file_path and not args.message:
        parser.error("Either --file-path or --message must be provided")

    main(args)
