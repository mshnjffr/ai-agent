"""Step 4 - The tool that closes the loop: edit_file.

read_file and list_files let the agent observe. edit_file lets it ACT - to
change something outside the model's context window, which is the actual
definition of an agent.

The implementation is deliberately dumb: replace an exact substring (old_str)
with new_str. If old_str is empty and the file doesn't exist, we create it.
Plain string replacement is enough for the model to write and rewrite real code.

This step ships three tools: read_file, list_files, edit_file. That's the whole
inner loop of a code-editing agent.

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


def edit_file(args: dict) -> str:
    """Replace old_str with new_str in a file, creating it if needed."""
    path = args.get("path")
    old_str = args.get("old_str", "")
    new_str = args.get("new_str", "")

    if not path or old_str == new_str:
        raise ValueError("invalid input: need a path and old_str != new_str")

    if not os.path.exists(path):
        if old_str == "":
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_str)
            return f"Successfully created file {path}"
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if old_str not in content:
        raise ValueError("old_str not found in file")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content.replace(old_str, new_str))

    return "OK"


EDIT_FILE = Tool(
    name="edit_file",
    description=(
        "Make edits to a text file.\n\n"
        "Replaces 'old_str' with 'new_str' in the given file. 'old_str' and "
        "'new_str' MUST be different from each other.\n\n"
        "If the file specified with path doesn't exist, it will be created "
        "(pass an empty 'old_str')."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file.",
            },
            "old_str": {
                "type": "string",
                "description": (
                    "Text to search for - must match exactly and must have "
                    "exactly one match. Leave empty to create a new file."
                ),
            },
            "new_str": {
                "type": "string",
                "description": "Text to replace old_str with.",
            },
        },
        "required": ["path", "new_str"],
    },
    function=edit_file,
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

    tools = [READ_FILE, LIST_FILES, EDIT_FILE]
    Agent(client, MODEL, tools).run()


if __name__ == "__main__":
    main()
