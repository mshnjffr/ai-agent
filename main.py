"""Step 3 - The tool that closes the loop: edit_file (entry point).

read_file and list_files let the agent observe. edit_file lets it ACT - to
change something outside the model's context window, which is the actual
definition of an agent.

The implementation (in tools.py) is deliberately dumb: replace an exact
substring (old_str) with new_str. If old_str is empty and the file doesn't
exist, we create it. Plain string replacement is enough for the model to write
and rewrite real code.

The loop in agent.py is unchanged; we just hand it one more tool here.

This step ships three tools: read_file, list_files, edit_file (all in tools.py).
That's the whole inner loop of a code-editing agent.

Run it:
    python main.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from agent import Agent
from tools import EDIT_FILE, LIST_FILES, READ_FILE

load_dotenv()

MODEL = os.getenv("MODEL", "anthropic/claude-sonnet-4.5")


def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY in your environment or .env file.")

    # OpenRouter speaks the OpenAI API, so we just point the OpenAI client at it.
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    tools = [READ_FILE, LIST_FILES, EDIT_FILE]
    Agent(client, MODEL, tools).run()


if __name__ == "__main__":
    main()
