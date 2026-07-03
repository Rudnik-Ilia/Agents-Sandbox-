"""Opt-in skills: markdown instructions a user loads on demand via the `/` menu."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from localagent.config import get_settings


@dataclass(frozen=True)
class Skill:
    """A single loadable skill parsed from a markdown file."""

    name: str
    description: str
    body: str
    path: Path


def _parse_skill(path: Path) -> Skill:
    """Parse a skill file, reading optional `name`/`description` frontmatter."""
    raw = path.read_text(encoding="utf-8")
    name = path.stem
    description = ""
    body = raw

    if raw.startswith("---"):
        _, _, rest = raw.partition("---")
        front, sep, remainder = rest.partition("---")
        if sep:
            body = remainder.strip()
            for line in front.strip().splitlines():
                key, _, value = line.partition(":")
                key, value = key.strip().lower(), value.strip()
                if key == "name" and value:
                    name = value
                elif key == "description" and value:
                    description = value
    return Skill(name=name, description=description, body=body.strip(), path=path)


class SkillRegistry:
    """Discovers skills on disk and exposes them for the slash-command menu."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self.reload()

    def reload(self) -> None:
        """Rescan the skills directory."""
        self._skills.clear()
        skills_dir = get_settings().skills_dir
        if not skills_dir.exists():
            return
        for path in sorted(skills_dir.glob("*.md")):
            skill = _parse_skill(path)
            self._skills[skill.name] = skill

    def list(self) -> list[Skill]:
        """Return all known skills sorted by name."""
        return sorted(self._skills.values(), key=lambda s: s.name)

    def get(self, name: str) -> Skill | None:
        """Look up a skill by exact name."""
        return self._skills.get(name)

    def get_by_index(self, index: int) -> Skill | None:
        """Look up a skill by its 1-based position in :meth:`list`."""
        skills = self.list()
        if 1 <= index <= len(skills):
            return skills[index - 1]
        return None
