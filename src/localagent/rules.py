"""Always-on rules: markdown files auto-loaded into every agent's system prompt."""

from __future__ import annotations

from localagent.config import get_settings


def load_rules() -> str:
    """Concatenate all markdown files in the rules directory into one block.

    Returns an empty string when no rules are present.
    """
    rules_dir = get_settings().rules_dir
    if not rules_dir.exists():
        return ""

    sections: list[str] = []
    for path in sorted(rules_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            sections.append(text)
    return "\n\n".join(sections)
