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

Want to see what the API is actually doing? Watch the log file while you chat:

    python main.py
    # in another terminal:
    tail -f agent.log
"""

import logging
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from agent import Agent
from tools import EDIT_FILE, LIST_FILES, READ_FILE

load_dotenv()

MODEL = os.getenv("MODEL", "anthropic/claude-sonnet-4.5")


def setup_logging() -> str:
    """Send the agent's behind-the-scenes events to a log file.

    The console stays clean (just the conversation); the log file gets the full
    story: every request, response, decision, and tool call. Override the path
    with AGENT_LOG if you like.
    """
    log_path = os.getenv("AGENT_LOG", "agent.log")

    logger = logging.getLogger("agent")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)

    logger.info("")
    logger.info("#" * 72)
    logger.info("# NEW SESSION  (model=%s)", MODEL)
    logger.info("#" * 72)
    return log_path


def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY in your environment or .env file.")

    log_path = setup_logging()
    print(f"(API activity is logged to ./{log_path} - try `tail -f {log_path}`)")

    # OpenRouter speaks the OpenAI API, so we just point the OpenAI client at it.
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    tools = [READ_FILE, LIST_FILES, EDIT_FILE]
    Agent(client, MODEL, tools).run()


if __name__ == "__main__":
    main()
