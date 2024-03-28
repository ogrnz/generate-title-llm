import asyncio
import time

from openai import AsyncOpenAI, OpenAI


async def async_query_openai(query: str, model="gpt-3.5-turbo"):
    response = await async_client.chat.completions.create(
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
    return response.choices[0].message.content


def query_openai(prompt: str, client: OpenAI, model="gpt-3.5-turbo"):
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
                "content": f"Create a concise, 3-5 word phrase as a header for the following query, strictly adhering to the 3-5 word limit and avoiding the use of the word 'title'. The json output should always have the key 'title'. Do NOT answer the query, only find a header: <query>{prompt}</query>",
            },
        ],
    )
    return response.choices[0].message.content


async def async_run(N, prompt):
    tasks = [async_query_openai(prompt) for _ in range(N)]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(r)


def normal_run(N, prompt):
    for _ in range(N):
        result = query_openai(prompt, client=normal_client)
        print(result)


if __name__ == "__main__":
    N = 5
    async_client = AsyncOpenAI()
    normal_client = OpenAI()
    prompt = "A current-carrying helix has 200 turns, a diameter of 4 cm, and a length of 20 cm. The current through each turn is 2 A. Calculate the magnetic field at a point on the axis of the helix, which is at a distance of 8 cm from the center of the helix."

    # Normal run
    t0 = time.perf_counter()
    normal_run(N, prompt)
    t1 = time.perf_counter()
    print(f"{t1 - t0:.2f}s")

    # Async run
    t0 = time.perf_counter()
    asyncio.run(async_run(N, prompt))
    t1 = time.perf_counter()
    print(f"{t1 - t0:.2f}s")
