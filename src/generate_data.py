import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tinydb import TinyDB, Query

logging.basicConfig(
    filename="app.log",
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("generate_data")
logger.addHandler(logging.StreamHandler(sys.stdout))
token_logger = logging.getLogger("token_logger")
token_logger.addHandler(logging.FileHandler("tokens.log"))

for muted_logger in ("httpx", "httpcore.connection"):
    logging.getLogger(muted_logger).setLevel(logging.ERROR)


def read_jsonl(jsonl_path) -> list:
    conversations = []
    with open(jsonl_path, "r", encoding="utf-8") as jsonl_file:
        for conversation in jsonl_file:
            first_msg = json.loads(conversation)["conversations"][0]["value"]
            data = {"message": first_msg, "title": ""}
            conversations.append(data)
    return conversations


def query_openai(query: str, client: OpenAI, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON.",
            },
            {
                "role": "user",
                "content": f"Create a concise, 3-5 word phrase as a header for the following query, strictly adhering to the 3-5 word limit and avoiding the use of the word 'title'. The json output should always have the key 'title'. Do NOT answer the query, only find a header: <query>{query}</query>",
            },
        ],
    )
    logger.debug(response)
    token_logger.info(response.usage.total_tokens)
    return json.loads(response.choices[0].message.content)


def generate_title(conversations: list, db: TinyDB):
    Prompt = Query()
    for conversation in conversations:
        # Check if the prompt is already in db
        message = conversation["message"]
        search = db.search(Prompt.message == message)
        if len(search) > 0:
            logger.info(f"Message already exists in db (doc_id {search[0].doc_id})")
            continue

        # All good, query openai
        response_json = query_openai(message, client=OpenAI())

        # Parse response
        title = response_json.get("title")

        # Insert in db if necessary
        if title:
            query = {"message": message, "title": title}
            logger.info(f"Title generated: {query['title']}")
            db.insert(query)
        else:
            logger.warning(f"Something went wrong while generating title for {message}")


def display_titles(db):
    for observation in db.all():
        print(observation["title"], observation["message"])


if __name__ == "__main__":
    load_dotenv()
    logger.info("App started")

    jsonl_path = Path("./data/chatalpaca-10k.json")
    conversations = read_jsonl(jsonl_path)
    conversations = conversations[:100]

    db = TinyDB("./data/db.json")

    # generate_title(conversations, db=db)
    display_titles(db)
