"""Step 1 - The chat loop.

This is NOT an agent yet. It's the heartbeat every agent is built on: read a
line from the user, send the whole conversation to the model, print the reply,
repeat. The model has no memory of its own - we keep the conversation and
resend it every turn.

Run it:
    python agent.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("MODEL", "anthropic/claude-sonnet-4.5")

BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY in your environment or .env file.")

    # OpenRouter speaks the OpenAI API, so we just point the OpenAI client at it.
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    conversation: list[dict] = []

    print("Chat with the agent (use 'ctrl-c' to quit)")
    while True:
        try:
            user_input = input(f"{BLUE}You{RESET}: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        conversation.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=MODEL,
            messages=conversation,
        )
        message = response.choices[0].message
        conversation.append(message.model_dump())

        print(f"{YELLOW}Agent{RESET}: {message.content}")


if __name__ == "__main__":
    main()
