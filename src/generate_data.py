import json
import logging
import sys
import time
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
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

# Mute verbose logger
for muted_logger in ("httpx", "httpcore.connection"):
    logging.getLogger(muted_logger).setLevel(logging.ERROR)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.2f} s")
        return result

    return timeit_wrapper


def read_jsonl(jsonl_path) -> list:
    conversations = []
    with open(jsonl_path, "r", encoding="utf-8") as jsonl_file:
        for conversation in jsonl_file:
            first_msg = json.loads(conversation)
            conversations.append(first_msg)
    return conversations


async def query_openai(query: str, client: AsyncOpenAI, model="gpt-3.5-turbo"):
    response = await client.chat.completions.create(
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


@timeit
def generate_titles(conversations: list, db: TinyDB):
    Prompt = Query()
    client = AsyncOpenAI()
    for conversation in conversations:
        # Check if the prompt is already in db
        message = conversation["message"]
        search = db.search(Prompt.message == message)
        if len(search) > 0:
            logger.info(f"Message already exists in db (doc_id {search[0].doc_id})")
            continue

        # All good, query openai
        response_json = query_openai(message, client=client)

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
        print(observation["title"], ":", observation["message"])


if __name__ == "__main__":
    load_dotenv()
    logger.info("App started")

    jsonl_path = Path("./data/dataset.jsonl")
    conversations = read_jsonl(jsonl_path)
    conversations = conversations[5:10]

    db = TinyDB("./data/db.json")

    generate_titles(conversations, db=db)
    # display_titles(db)
