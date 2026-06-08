"""The Agent - an LLM, its tools, and the loop that connects them.

This is the reusable core. It deliberately knows nothing about how it was
configured: no environment variables, no argument parsing, no client creation.
You hand it a ready-to-use client, a model name, and a list of tools, and it
runs the conversation. main.py is responsible for wiring those pieces together.

Keeping the class in its own module (separate from the entry point) follows the
single-responsibility principle: agent.py owns the agent loop, main.py owns
startup/config, tools.py owns the tools.

Behind-the-scenes logging
-------------------------
The console only shows the conversation. To actually SEE what the API is doing,
this module logs every step to a logger named "agent": the request we send, the
raw response, whether the response is a plain reply or a tool call, the decision
the loop makes, and each tool execution + result. main.py points that logger at
a file (agent.log) so you can read the whole story.
"""

import json
import logging

from openai import OpenAI

from tools import Tool

log = logging.getLogger("agent")

BLUE = "\033[94m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"


def _truncate(text: str, limit: int = 600) -> str:
    """Make long / multi-line content readable on a single log line."""
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... ({len(text)} chars total)"


def _format_messages(messages: list[dict]) -> str:
    """Render the conversation we're about to send, the way the API sees it."""
    lines = []
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "tool":
            lines.append(
                f"    [{i}] tool   (result for {m.get('tool_call_id')}): "
                f"{_truncate(str(m.get('content')))}"
            )
        elif role == "assistant" and m.get("tool_calls"):
            calls = ", ".join(
                f"{c['function']['name']}({c['function']['arguments']})"
                for c in m["tool_calls"]
            )
            text = _truncate(m.get("content") or "")
            lines.append(f"    [{i}] assistant: text={text!r} tool_calls=[{calls}]")
        else:
            lines.append(f"    [{i}] {role}: {_truncate(str(m.get('content')))}")
    return "\n".join(lines)


def _format_response(message) -> str:
    """Render the model's reply: its text and any tool calls it asked for."""
    lines = []
    lines.append(f"    assistant text: {_truncate(message.content) if message.content else '(none)'}")
    if message.tool_calls:
        lines.append(f"    tool_calls ({len(message.tool_calls)}):")
        for c in message.tool_calls:
            lines.append(
                f"      - id={c.id} name={c.function.name} args={c.function.arguments}"
            )
    else:
        lines.append("    tool_calls: none")
    return "\n".join(lines)


class Agent:
    def __init__(self, client: OpenAI, model: str, tools: list[Tool]) -> None:
        self.client = client
        self.model = model
        self.tools = {tool.name: tool for tool in tools}

    def run(self) -> None:
        conversation: list[dict] = []

        # Show, once, the exact tool schemas the model receives.
        log.info("TOOLS REGISTERED (these JSON schemas are sent to the model on every request):")
        for tool in self.tools.values():
            log.info("%s", json.dumps(tool.to_openai(), indent=2))

        print("Chat with the agent (use 'ctrl-c' to quit)")
        read_user_input = True
        while True:
            if read_user_input:
                try:
                    user_input = input(f"{BLUE}You{RESET}: ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                log.info("=" * 72)
                log.info("USER INPUT: %s", user_input)
                conversation.append({"role": "user", "content": user_input})

            message = self._run_inference(conversation)
            conversation.append(message.model_dump())

            if message.content:
                print(f"{YELLOW}Agent{RESET}: {message.content}")

            tool_calls = message.tool_calls or []
            if not tool_calls:
                # No tool requested - hand the turn back to the user.
                log.info(
                    "DECISION: response has NO tool calls -> show the reply and "
                    "wait for the user's next message.\n"
                )
                read_user_input = True
                continue

            # The model asked for one or more tools. Run them, append each
            # result, and loop WITHOUT asking the user for input so the model
            # can act on the results.
            log.info(
                "DECISION: response HAS %d tool call(s) -> run the tool(s), append "
                "the result(s), and call the API again WITHOUT prompting the user.",
                len(tool_calls),
            )
            for call in tool_calls:
                conversation.append(self._execute_tool(call))
            read_user_input = False

    def _run_inference(self, conversation: list[dict]):
        log.info(
            "--> API REQUEST (model=%s, %d message(s), tools offered=%s)\n%s",
            self.model,
            len(conversation),
            list(self.tools.keys()),
            _format_messages(conversation),
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=conversation,
            tools=[tool.to_openai() for tool in self.tools.values()],
        )
        choice = response.choices[0]
        log.info(
            "<-- API RESPONSE (finish_reason=%s)\n%s",
            choice.finish_reason,
            _format_response(choice.message),
        )
        return choice.message

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
            log.info("TOOL EXEC: %s -> NOT FOUND", name)
        elif parse_error is not None:
            content = f"could not parse arguments: {parse_error}"
            log.info("TOOL EXEC: %s -> bad arguments: %s", name, parse_error)
        else:
            print(f"{GREEN}tool{RESET}: {name}({json.dumps(args)})")
            log.info("TOOL EXEC: %s(%s)", name, json.dumps(args))
            try:
                content = tool.function(args)
                log.info("TOOL RESULT (%d chars): %s", len(content), _truncate(content))
            except Exception as exc:  # surface the error back to the model
                content = f"error: {exc}"
                log.info("TOOL RESULT: error: %s", exc)

        return {
            "role": "tool",
            "tool_call_id": call.id,
            "content": content,
        }
