
from openai import OpenAI
import os
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pprint import pprint
from pathlib import Path
import click
import logging
import uuid
from bickford.config import PROJECT_ROOT
from bickford.telemetry import setup_tracing
from openinference.instrumentation import using_session

# Setup tracing for OpenAI and smolagents
setup_tracing()

logger = logging.getLogger("bickford")
load_dotenv()

session_id = str(uuid.uuid4())  # or your own conversation ID


def generate_response(client: OpenAI, messages: List[Dict[str, str]]):
    logger.debug(f"Generating response for messages: {messages}")
    with using_session(session_id):
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
    return response

def chat(client: OpenAI, initial_prompt: Optional[str] = None):
    if initial_prompt:
        question = initial_prompt
    else:
        question = input("Enter your question: ")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question}
    ]

    try:
        while True:
            start_time = time.time()
            response = generate_response(client, messages)

            response_message = ""
            usage = None
            for chunk in response:
                if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                    response_message += chunk.choices[0].delta.content
                    print(chunk.choices[0].delta.content, end="", flush=True)
                if chunk.usage is not None:
                    usage = chunk.usage

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            print("\n--------------------------------")
            messages.append({"role": "assistant", "content": response_message})

            # Keep only the last 10 messages
            if len(messages) > 10:
                messages.pop(0)

            if usage is not None:
                cached_tokens = usage.prompt_tokens_details.cached_tokens
                input_tokens = usage.prompt_tokens - cached_tokens

                # Based on https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/#pricing
                # Pricing per million tokens: Input $2.50, Cached Input $1.25, Output $10.00
                input_cost = input_tokens * 2.50 / 1_000_000
                cached_cost = cached_tokens * 1.25 / 1_000_000
                output_cost = usage.completion_tokens * 10.00 / 1_000_000
                total_cost = input_cost + cached_cost + output_cost

                print("\n--------------------------------")
                print(f"[stats] prompt={usage.prompt_tokens} cached={cached_tokens} completion={usage.completion_tokens} cost=${total_cost:.12f} latency={latency_ms:.2f}ms")

            question = input("Enter your question: ")
            messages.append({"role": "user", "content": question})


    except KeyboardInterrupt:
        pprint(messages)
        print("\nExiting...")

@click.command()
@click.argument('prompt', required=False, default=None)
def main(prompt: Optional[str] = None):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    chat(client, initial_prompt=prompt)


if __name__ == "__main__":
    main()
