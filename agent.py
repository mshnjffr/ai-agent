"""The Agent - an LLM, its tools, and the loop that connects them.

This is the reusable core. It deliberately knows nothing about how it was
configured: no environment variables, no argument parsing, no client creation.
You hand it a ready-to-use client, a model name, and a list of tools, and it
runs the conversation. main.py is responsible for wiring those pieces together.

Keeping the class in its own module (separate from the entry point) follows the
single-responsibility principle: agent.py owns the agent loop, main.py owns
startup/config, tools.py owns the tools.
"""

import json

from openai import OpenAI

from tools import Tool

BLUE = "\033[94m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"


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
