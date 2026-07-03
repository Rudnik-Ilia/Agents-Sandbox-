"""Interactive REPL with a `/` slash-command menu for loading skills.

The menu is a plain numbered list (no colors/emojis); type `/` to see skills and
`/<number>` or `/<name>` to load one into the active context.
"""

from __future__ import annotations

from localagent.agents.base import Agent
from localagent.logging_setup import AgentLogger
from localagent.rules import load_rules
from localagent.skills import Skill, SkillRegistry

_PROMPT = "you> "
_HELP = (
    "commands: /skills (list) | /<n> or /<name> (load skill) | /add [path] (add doc) | /remember <fact> | "
    "/active | /clear | /win (show context window) | /rules | /help | /exit"
)
_BORDER = "-" * 78


def _print_context_window(agent: Agent, active: list[Skill]) -> None:
    """Print every message that would currently be sent to the model."""
    skill_context = "\n\n".join(s.body for s in active)
    messages = agent.context_window(skill_context)
    chars = 0
    print(_BORDER)
    for message in messages:
        content = str(message.content)
        chars += len(content)
        print(f"[{message.type}]")
        print(content)
        print(_BORDER)
    print(f"  messages={len(messages)} | ~{chars} chars | ~{chars // 4} est. tokens")


def _print_skill_menu(registry: SkillRegistry) -> None:
    skills = registry.list()
    if not skills:
        print("  (no skills found in the skills/ folder)")
        return
    print("  available skills (type /<number> or /<name> to load):")
    for i, skill in enumerate(skills, start=1):
        desc = f" - {skill.description}" if skill.description else ""
        print(f"    {i}. {skill.name}{desc}")


def _handle_slash(
    command: str, agent: Agent, registry: SkillRegistry, active: list[Skill], logger: AgentLogger
) -> bool:
    """Process a slash command. Return False to signal the REPL should exit."""
    body = command[1:].strip()
    lowered = body.lower()

    if lowered in {"exit", "quit", "q"}:
        return False
    if lowered in {"", "skills", "s"}:
        _print_skill_menu(registry)
        return True
    if lowered == "win":
        _print_context_window(agent, active)
        return True
    if lowered == "help":
        print(f"  {_HELP}")
        return True
    if lowered == "rules":
        rules = load_rules()
        print(f"  {rules}" if rules else "  (no rules loaded)")
        return True
    if lowered == "active":
        print("  active skills: " + (", ".join(s.name for s in active) if active else "(none)"))
        return True
    if lowered == "clear":
        active.clear()
        print("  cleared active skills")
        return True
    if lowered.startswith("add"):
        path = body[3:].strip()
        print(f"  {agent.ingest_document(path)}")
        return True
    if lowered.startswith("remember"):
        note = body[len("remember"):].strip()
        if not note:
            print("  usage: /remember <fact to store in SOUL.md>")
        else:
            print(f"  {agent.remember(note)}")
        return True

    skill = registry.get_by_index(int(body)) if body.isdigit() else registry.get(body)
    if skill is None:
        print(f"  unknown skill '{body}'. Type /skills to list.")
        return True
    if skill not in active:
        active.append(skill)
    logger.info("skill loaded", skill=skill.name)
    print(f"  loaded skill '{skill.name}'")
    return True


def run_repl(agent: Agent, logger: AgentLogger, title: str, subtitle: str = "") -> None:
    """Run the interactive loop for the given agent until the user exits."""
    registry = SkillRegistry()
    active: list[Skill] = []

    logger.banner(title, subtitle)
    print(f"  {_HELP}\n")

    while True:
        try:
            user = input(_PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user.startswith("/"):
            if not _handle_slash(user, agent, registry, active, logger):
                break
            continue

        skill_context = "\n\n".join(s.body for s in active)
        try:
            answer = agent.respond(user, skill_context)
        except Exception as exc:  # noqa: BLE001 - surface any runtime failure to the user
            logger.error(f"agent failed: {exc}", exc=exc)
            print(f"bot> [error] {exc}\n")
            continue
        print(f"bot> {answer}\n")

    logger.info("session end")
    print("bye")
