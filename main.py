"""Step 1 - The agent (entry point).

We add two things to the bare chat loop on `main`:

  1. Tools. A tool is just: a name, a description, a JSON schema for its
     inputs, and a function that runs it. We tell the model which tools exist.
     (See tools.py.)
  2. The agentic loop. When the model wants a tool, it doesn't run it - it
     replies with a `tool_call` asking US to. We run the function, send the
     result back, and ask the model again. We keep looping until the model
     stops asking for tools and just talks to the user. (See agent.py.)

That second point is the whole idea of an agent. Everything else is decoration.

This file is just the entry point: it sets up logging, loads config, builds the
OpenRouter client, picks which tools to hand the agent, and starts it.

Want to see what the API is actually doing? Watch the log file while you chat:

    python main.py
    # in another terminal:
    tail -f agent.log

This step ships one tool: read_file. On `main` you saw the chat loop with no
tools at all - start there if you haven't.
"""

import logging
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from agent import Agent
from tools import READ_FILE

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

    tools = [READ_FILE]
    Agent(client, MODEL, tools).run()


if __name__ == "__main__":
    main()
