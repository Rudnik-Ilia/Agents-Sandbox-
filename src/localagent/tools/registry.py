"""Central registry of the example tools available to the tool-calling agents.

Every tool takes a single string argument, which keeps the ReAct text protocol
trivial to parse while still exercising native structured tool-calls.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from localagent.tools.calculator import calculator
from localagent.tools.files import read_file
from localagent.tools.memory_tool import remember
from localagent.tools.shell import shell
from localagent.tools.web import web_search

ALL_TOOLS: list[BaseTool] = [calculator, shell, read_file, web_search, remember]
TOOLS_BY_NAME: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}

#: Tools that require human approval before running.
DANGEROUS_TOOLS: set[str] = {"shell"}

#: One-line usage hints used to build the ReAct system prompt.
TOOL_USAGE: dict[str, str] = {
    "calculator": "calculator <arithmetic expression>",
    "shell": "shell <command line>",
    "read_file": "read_file <path to a text file>",
    "web_search": "web_search <search query>",
    "remember": "remember <durable fact to store in long-term memory>",
}
