"""Step 2 - A second tool: list_files.

The loop doesn't change at all - we just add another tool to the registry. The
interesting part is what the model does with it: given read_file AND list_files,
it will chain them on its own (e.g. list the directory, then read the files it
finds) without us scripting that behaviour.

Note there's no fixed format for a tool's output. list_files returns JSON with a
trailing "/" on directories simply because it's easy for the model to parse.
Picking good tool output is an experiment, not a rule.

This step ships two tools: read_file and list_files.

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


def list_files(args: dict) -> str:
    """List files and directories under a path, recursively."""
    root = args.get("path") or "."
    entries: list[str] = []
    for current, dirs, files in os.walk(root):
        for d in dirs:
            rel = os.path.relpath(os.path.join(current, d), root)
            entries.append(rel + "/")
        for f in files:
            rel = os.path.relpath(os.path.join(current, f), root)
            entries.append(rel)
    return json.dumps(sorted(entries))


LIST_FILES = Tool(
    name="list_files",
    description=(
        "List files and directories at a given path. If no path is provided, "
        "lists files in the current directory."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Optional relative path to list files from. Defaults to the "
                    "current directory if not provided."
                ),
            },
        },
    },
    function=list_files,
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

    tools = [READ_FILE, LIST_FILES]
    Agent(client, MODEL, tools).run()


if __name__ == "__main__":
    main()
