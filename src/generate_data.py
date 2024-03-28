import asyncio
import json
import logging
import sys
import time
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tinydb import TinyDB, Query

logging.basicConfig(
    filename="app.log",
    filemode="a",
    level=logging.INFO,
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


async def generate_titles(conversations: list, db: TinyDB):
    Prompt = Query()
    client = AsyncOpenAI()

    to_query = []
    for conversation in conversations:
        # Check if the prompt is already in db
        message = conversation["message"]
        search = db.search(Prompt.message == message)
        if len(search) > 0:
            logger.info(f"Message already exists in db (doc_id {search[0].doc_id})")
            continue
        else:
            to_query.append(message)

    tasks = [query_openai(message, client=client) for message in to_query]
    results = await asyncio.gather(*tasks)

    titles = [result.get("title") for result in results]
    return to_query, titles


def insert_titles_into_db(messages, titles):
    queries = []
    for message, title in zip(messages, titles):
        if title:
            query = {"message": message, "title": title}
            logger.info(f"Title generated and inserted: {query['title']}")
            queries.append(query)
        else:
            logger.warning(f"Something went wrong while generating title for {message}")
    db.insert_multiple(queries)


def display_titles(db):
    for observation in db.all():
        print(observation["title"])
        print(observation["message"])
        print("\n")


if __name__ == "__main__":
    load_dotenv()
    logger.info("App started")

    jsonl_path = Path("./data/dataset.jsonl")
    db = TinyDB("./data/db.json")
    conversations = read_jsonl(jsonl_path)

    chunksize = 100  # Shouldn't be too high, otherwise we'll get rate limited
    for i in range(0, len(conversations), chunksize):
        t0 = time.perf_counter()
        messages, titles = asyncio.run(
            generate_titles(conversations[i : i + chunksize], db=db)
        )
        insert_titles_into_db(messages, titles)
        t1 = time.perf_counter()
        print(f"Chunk generation took {t1 - t0:.2f}s")

    # display_titles(db)
