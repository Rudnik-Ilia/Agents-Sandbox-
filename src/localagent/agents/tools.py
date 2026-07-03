"""Two tool-calling agents sharing a common set of example tools.

* :class:`ReActToolAgent` shows the "text" approach: the model emits a parseable
  ``ACTION:`` line and the runner executes the tool and feeds back an observation.
* :class:`NativeToolAgent` shows the "regular" approach using LangChain's native
  ``bind_tools`` / structured tool-calls supported by the Ollama model.

Both share the tools defined in :mod:`localagent.tools.registry`: calculator,
shell, read_file and web_search.
"""

from __future__ import annotations

import asyncio
import json
import re
import time

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from localagent.agents.base import Agent
from localagent.approval import confirm_tool
from localagent.cli import run_repl
from localagent.llm import build_chat_llm, build_reliable_chat, token_usage, with_reliability
from localagent.logging_setup import AgentLogger, get_agent_logger
from localagent.mcp_tools import load_mcp_tools
from localagent.memory import BufferMemory
from localagent.tools.registry import ALL_TOOLS, DANGEROUS_TOOLS, TOOL_USAGE

_MAX_STEPS = 6
_ACTION_RE = re.compile(r"ACTION:\s*(\w+)\s+(.+)", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"ANSWER:\s*(.*)", re.IGNORECASE | re.DOTALL)


def _invoke_tool(tool: BaseTool, argument: object) -> str:
    """Invoke a tool synchronously, using the async path (required by MCP tools)."""
    return str(asyncio.run(tool.ainvoke(argument)))


def _build_toolset(
    logger: AgentLogger, extra_tools: list[BaseTool] | None = None
) -> tuple[list[BaseTool], dict[str, BaseTool], str]:
    """Combine built-in tools, any extra tools, and configured MCP tools.

    Built-in/extra tools take precedence; MCP tools whose names collide are skipped
    so ``bind_tools`` never sees duplicate names.
    """
    tools: list[BaseTool] = [*ALL_TOOLS, *(extra_tools or [])]
    by_name: dict[str, BaseTool] = {t.name: t for t in tools}
    for tool in load_mcp_tools(logger):
        if tool.name in by_name:
            logger.info("skipped MCP tool with duplicate name", tool=tool.name)
            continue
        tools.append(tool)
        by_name[tool.name] = tool

    lines = []
    for tool in tools:
        hint = TOOL_USAGE.get(tool.name)
        if hint is None:
            first_line = (tool.description or "").splitlines()[0][:70]
            hint = f"{tool.name} <input> ({first_line})"
        lines.append(f"- {hint}")
    return tools, by_name, "\n".join(lines)


def _react_instructions(tool_list: str) -> str:
    return (
        "You can use these tools:\n"
        f"{tool_list}\n\n"
        "Follow this protocol strictly, one step per reply:\n"
        "- To use a tool, reply with a single line: ACTION: <tool> <input>\n"
        "- <input> is the tool's text argument, or a JSON object for multi-field tools.\n"
        "- After you receive a line 'OBSERVATION: <result>', continue.\n"
        "- When you have the final answer, reply: ANSWER: <your answer>\n"
        "If no tool is needed, reply directly with ANSWER: <your answer>."
    )


def _native_instructions(names: list[str]) -> str:
    return (
        f"You are an assistant with access to tools: {', '.join(names)}. "
        "Call a tool when it helps; otherwise answer directly. Be concise."
    )


class ReActToolAgent(Agent):
    """Text-protocol tool use: parse an ACTION line, run the tool, feed back."""

    def __init__(self, logger: AgentLogger) -> None:
        self._llm = build_reliable_chat()
        self._logger = logger
        self._memory = BufferMemory(session="tools-react", persist=False)
        self._tools, self._tools_by_name, tool_list = _build_toolset(logger)
        self.instructions = _react_instructions(tool_list)

    def context_window(self, skill_context: str = "") -> list[BaseMessage]:
        return [SystemMessage(self.system_prompt(skill_context)), *self._memory.messages()]

    def respond(self, user_text: str, skill_context: str = "") -> str:
        messages: list[BaseMessage] = [
            SystemMessage(self.system_prompt(skill_context)),
            *self._memory.messages(),
            HumanMessage(user_text),
        ]
        answer = "stopped: too many tool steps"
        for _ in range(_MAX_STEPS):
            start = time.perf_counter()
            reply = self._llm.invoke(messages)
            latency_ms = (time.perf_counter() - start) * 1000
            text = reply.content.strip()
            self._logger.llm_call(str(messages[-1].content), text, latency_ms, token_usage(reply))

            action = _ACTION_RE.search(text)
            if action:
                name = action.group(1).strip().lower()
                tool_input = action.group(2).strip().strip("`\"'")
                result = self._run_tool(name, tool_input)
                messages.append(AIMessage(text))
                messages.append(HumanMessage(f"OBSERVATION: {result}"))
                continue

            match = _ANSWER_RE.search(text)
            answer = match.group(1).strip() if match else text
            break

        self._memory.add_user(user_text)
        self._memory.add_ai(answer)
        return answer

    def _run_tool(self, name: str, tool_input: str) -> str:
        tool = self._tools_by_name.get(name)
        if tool is None:
            result = f"error: unknown tool '{name}'"
        elif name in DANGEROUS_TOOLS and not confirm_tool(name, tool_input):
            result = "denied by user"
            self._logger.route("tool_denied", name)
        else:
            argument: object = tool_input
            if tool_input.startswith("{"):
                try:
                    argument = json.loads(tool_input)
                except json.JSONDecodeError:
                    argument = tool_input
            try:
                result = _invoke_tool(tool, argument)
            except Exception as exc:  # noqa: BLE001 - surface tool errors as observations
                result = f"error: {exc}"
        self._logger.tool_call(name, tool_input, result)
        return result


class NativeToolAgent(Agent):
    """Native tool-calling via LangChain `bind_tools` and structured tool calls."""

    def __init__(
        self, logger: AgentLogger, extra_tools: list[BaseTool] | None = None, session: str = "tools-native"
    ) -> None:
        self._logger = logger
        self._memory = BufferMemory(session=session, persist=False)
        self._tools, self._tools_by_name, _ = _build_toolset(logger, extra_tools)
        self._llm = with_reliability(build_chat_llm().bind_tools(self._tools))
        self.instructions = _native_instructions(list(self._tools_by_name))

    def context_window(self, skill_context: str = "") -> list[BaseMessage]:
        return [SystemMessage(self.system_prompt(skill_context)), *self._memory.messages()]

    def respond(self, user_text: str, skill_context: str = "") -> str:
        messages: list[BaseMessage] = [
            SystemMessage(self.system_prompt(skill_context)),
            *self._memory.messages(),
            HumanMessage(user_text),
        ]
        answer = "stopped: too many tool steps"
        for _ in range(_MAX_STEPS):
            start = time.perf_counter()
            reply: AIMessage = self._llm.invoke(messages)
            # breakpoint()
            latency_ms = (time.perf_counter() - start) * 1000
            self._logger.llm_call(user_text, reply.content or "[tool call]", latency_ms, token_usage(reply))
            messages.append(reply)

            if not reply.tool_calls:
                answer = reply.content.strip()
                break

            for call in reply.tool_calls:
                result = self._run_tool(call["name"], call["args"])
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

        self._memory.add_user(user_text)
        self._memory.add_ai(answer)
        return answer

    def _run_tool(self, name: str, arguments: dict) -> str:
        tool = self._tools_by_name.get(name)
        if tool is None:
            result = f"error: unknown tool '{name}'"
        elif name in DANGEROUS_TOOLS and not confirm_tool(name, arguments):
            result = "denied by user"
            self._logger.route("tool_denied", name)
        else:
            try:
                result = _invoke_tool(tool, arguments)
            except Exception as exc:  # noqa: BLE001 - surface tool errors to the model
                result = f"error: {exc}"
        self._logger.tool_call(name, arguments, result)
        return result


def run_tools(mode: str) -> None:
    """Launch the tool-calling REPL in either 'react' or 'native' mode."""
    logger = get_agent_logger(f"tools-{mode}")
    agent: Agent = ReActToolAgent(logger) if mode == "react" else NativeToolAgent(logger)
    title = "Tools agent (ReAct text protocol)" if mode == "react" else "Tools agent (native tool-calling)"
    run_repl(agent, logger, title=title, subtitle="tools: calculator, shell, read_file, web_search")
