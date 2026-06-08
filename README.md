# Build an Agent (in Python, with OpenRouter)

A tiny, readable companion to Thorsten Ball's
[*How to Build an Agent*](https://ampcode.com/notes/how-to-build-an-agent) —
ported from Go/Anthropic to **Python** and **[OpenRouter](https://openrouter.ai)**
so you can use whatever model you like (Claude, GPT, etc.) behind one API.

The whole point of the original post: an agent is not magic. It's

> **an LLM, a loop, and enough tokens.**

This repo proves it in ~200 lines of Python, built up one tool at a time.

## What you'll build

A terminal chat agent that can read files, list directories, and edit/create
files — enough to have it write and modify real code for you.

## The progression (one branch per step)

Each step lives on its own feature branch and builds on the previous one. Read
them in order, or `git diff` between them to see exactly what each step adds.

| Branch | What it adds |
| --- | --- |
| `main` (start)      | The bare chat loop: talk to a model in your terminal. Not an agent yet. |
| `step-1-read-file`  | The tool-use loop + the first tool, `read_file`. **Now it's an agent.** |
| `step-2-list-files` | A `list_files` tool. Watch it chain tools on its own. |
| `step-3-edit-file`  | An `edit_file` tool (string replace + file creation). It can now write code. |

**`main` is the starting point** - just the project scaffold and the bare chat
loop. From here you build forward through the step branches. The final, complete
agent lives on `step-3-edit-file`.

```bash
git log --oneline --all --graph   # see the whole story
git checkout step-1-read-file     # the first real agent
git checkout step-3-edit-file     # the finished agent
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then put your OpenRouter key in .env
```

## Run

On `main` (the single-file chat loop):

```bash
python agent.py
```

On the step branches, the code is split into modules and you run the entry
point instead:

```bash
python main.py
```

Either way, just talk to it:

```
Chat with the agent (use 'ctrl-c' to quit)
You: what files are in this directory?
```

## File layout (on the step branches)

| File | Responsibility |
| --- | --- |
| `main.py`  | Entry point: load config, build the OpenRouter client, pick tools, run. |
| `agent.py` | The `Agent` class — the tool-use loop. Reusable, no config/CLI concerns. |
| `tools.py` | The `Tool` type and the tool implementations. |

## How tool calling works (the one idea)

You tell the model what tools exist. When it wants one, it doesn't run it — it
*asks you to* by returning a `tool_call`. You run the function, send the result
back, and loop. That's the entire trick; everything else is polish.

## Notes

- This uses the OpenAI Python SDK pointed at OpenRouter's OpenAI-compatible
  endpoint (`https://openrouter.ai/api/v1`).
- It's intentionally minimal: no sandboxing, no confirmation prompts. The agent
  will read and edit files in the current directory. Run it somewhere safe and
  play in a throwaway folder.
