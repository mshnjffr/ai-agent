"""Tools the agent can use.

A tool is just four things:
  - a name
  - a description (so the model knows when to reach for it)
  - a JSON schema describing its inputs
  - a function that runs it and returns a string

Keeping tools in their own file means agent.py can focus on the loop, and
adding a new capability is simply: write a function, wrap it in a Tool, and
append it to the list in agent.py.
"""

import json
import os
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    function: Callable[[dict], str]

    def to_openai(self) -> dict:
        """Render this tool in the shape the OpenAI/OpenRouter API expects."""
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
