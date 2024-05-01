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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    summarizer = pipeline("summarization", model=args.checkpoint, tokenizer=tokenizer)

    if args.message:  # Directly use the message string provided
        messages = [args.message]
    else:  # Load messages from file if path is provided
        messages = load_messages(args.file_path)

    for message in messages:
        title = generate_title(message, summarizer)[0]["summary_text"]
        if args.pretty:
            print(f"title: {title}\nmessage: {message}")
            print("-" * 10)
        else:
            print({"title": title, "message": message})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate summaries from texts using a transformer model."
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="ogrnz/t5-chat-titles",
        help="Model checkpoint path or compatible huggingface model",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        default="google-t5/t5-small",
        help="Model tokenizer path",
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
    parser.add_argument(
        "-p",
        "--pretty",
        action=argparse.BooleanOptionalAction,
        help="(optional) Output format in human-readable pretty format",
    )
    args = parser.parse_args()

    if not args.file_path and not args.message:
        parser.error("Either --file-path or --message must be provided")

    main(args)
