"""Step 2 - A second tool: list_files (entry point).

The loop in agent.py doesn't change at all - we just add another tool in
tools.py and hand it to the agent here. The interesting part is what the model
does with it: given read_file AND list_files, it will chain them on its own
(e.g. list the directory, then read the files it finds) without us scripting
that behaviour.

Note there's no fixed format for a tool's output. list_files returns JSON with a
trailing "/" on directories simply because it's easy for the model to parse.
Picking good tool output is an experiment, not a rule.

This step ships two tools: read_file and list_files (both in tools.py).

Run it:
    python main.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from agent import Agent
from tools import LIST_FILES, READ_FILE

load_dotenv()

MODEL = os.getenv("MODEL", "anthropic/claude-sonnet-4.5")


def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY in your environment or .env file.")

    # OpenRouter speaks the OpenAI API, so we just point the OpenAI client at it.
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    tools = [READ_FILE, LIST_FILES]
    Agent(client, MODEL, tools).run()


if __name__ == "__main__":
    main()
