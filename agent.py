"""Step 1 - The agent.

We add two things to the bare chat loop on `main`:

  1. Tools. A tool is just: a name, a description, a JSON schema for its
     inputs, and a function that runs it. We tell the model which tools exist.
  2. The agentic loop. When the model wants a tool, it doesn't run it - it
     replies with a `tool_call` asking US to. We run the function, send the
     result back, and ask the model again. We keep looping until the model
     stops asking for tools and just talks to the user.

That second point is the whole idea of an agent. Everything else is decoration.

This step ships one tool: read_file. (On `main` you saw the chat loop with no
tools at all - start there if you haven't.)

Run it:
    python agent.py
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("MODEL", "anthropic/claude-sonnet-4.5")

BLUE = "\033[94m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"


# --- Tools --------------------------------------------------------------------


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    function: Callable[[dict], str]

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


def read_file(args: dict) -> str:
    """Read the contents of a relative file path."""
    with open(args["path"], "r", encoding="utf-8") as f:
        return f.read()


READ_FILE = Tool(
    name="read_file",
    description=(
        "Read the contents of a given relative file path. Use this when you "
        "want to see what's inside a file. Do not use this with directory names."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The relative path of a file in the working directory.",
            },
        },
        "required": ["path"],
    },
    function=read_file,
)


# --- Agent --------------------------------------------------------------------


class Agent:
    def __init__(self, client: OpenAI, model: str, tools: list[Tool]) -> None:
        self.client = client
        self.model = model
        self.tools = {tool.name: tool for tool in tools}

    def run(self) -> None:
        conversation: list[dict] = []

        print("Chat with the agent (use 'ctrl-c' to quit)")
        read_user_input = True
        while True:
            if read_user_input:
                try:
                    user_input = input(f"{BLUE}You{RESET}: ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                conversation.append({"role": "user", "content": user_input})

            message = self._run_inference(conversation)
            conversation.append(message.model_dump())

            if message.content:
                print(f"{YELLOW}Agent{RESET}: {message.content}")

            tool_calls = message.tool_calls or []
            if not tool_calls:
                # No tool requested - hand the turn back to the user.
                read_user_input = True
                continue

            # The model asked for one or more tools. Run them, append each
            # result, and loop WITHOUT asking the user for input so the model
            # can act on the results.
            for call in tool_calls:
                conversation.append(self._execute_tool(call))
            read_user_input = False

    def _run_inference(self, conversation: list[dict]):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=conversation,
            tools=[tool.to_openai() for tool in self.tools.values()],
        )
        return response.choices[0].message

    def _execute_tool(self, call) -> dict:
        name = call.function.name
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            args, parse_error = {}, exc
        else:
            parse_error = None

        tool = self.tools.get(name)
        if tool is None:
            content = f"tool not found: {name}"
        elif parse_error is not None:
            content = f"could not parse arguments: {parse_error}"
        else:
            print(f"{GREEN}tool{RESET}: {name}({json.dumps(args)})")
            try:
                content = tool.function(args)
            except Exception as exc:  # surface the error back to the model
                content = f"error: {exc}"

        return {
            "role": "tool",
            "tool_call_id": call.id,
            "content": content,
        }


def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY in your environment or .env file.")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    tools = [READ_FILE]
    Agent(client, MODEL, tools).run()


if __name__ == "__main__":
    main()
