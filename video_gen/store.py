"""Durable, human-editable archives for productions and cast members.

Productions live in ``video_scripts/<id>.md``; cast members live in
``characters/<id>.md`` next to their reference art, so a character can be
reused across productions and edited by hand between renders.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .model import Character, Production, character_key, split_frontmatter

LOGGER = logging.getLogger("video_gen.store")


class ProductionStore:
    """Production archive with a regenerated index."""

    def __init__(self, root: Path = Path("video_scripts")) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, production_id: str) -> Path:
        return self.root / f"{production_id}.md"

    def save(self, production: Production) -> Path:
        path = self.path_for(production.id)
        path.write_text(production.to_markdown(), encoding="utf-8")
        self.write_index()
        return path

    def load(self, production_id: str) -> Production:
        path = (
            Path(production_id)
            if str(production_id).endswith(".md")
            else self.path_for(production_id)
        )
        if not path.exists():
            raise FileNotFoundError(f"no production at {path}")
        return Production.from_markdown(path.read_text(encoding="utf-8"))

    def list_productions(self) -> List[Production]:
        productions: List[Production] = []
        for path in sorted(self.root.glob("*.md")):
            if path.name == "INDEX.md":
                continue
            try:
                productions.append(Production.from_markdown(path.read_text(encoding="utf-8")))
            except Exception as exc:  # noqa: BLE001 - one bad file must not hide the rest
                LOGGER.warning("skipping unreadable production %s: %s", path.name, exc)
        productions.sort(key=lambda p: p.created, reverse=True)
        return productions

    def accept(self, production: Production, discard_siblings: bool = True) -> Path:
        """Mark accepted; demote the other candidates from the same batch."""
        production.status = "accepted"
        batch = production.config.get("batch")
        if discard_siblings and batch:
            for other in self.list_productions():
                if (
                    other.id != production.id
                    and other.config.get("batch") == batch
                    and other.status in ("draft", "candidate")
                ):
                    other.status = "rejected"
                    self.path_for(other.id).write_text(other.to_markdown(), encoding="utf-8")
        return self.save(production)

    def record_render(self, production: Production, record: Dict[str, object]) -> Path:
        production.renders.append(record)
        production.status = "rendered"
        return self.save(production)

    def write_index(self) -> Path:
        rows = [
            "# Production index",
            "",
            "Regenerated automatically on every save. One row per production, so both",
            "you and the model can scan the whole history without opening files.",
            "",
            "| id | title | format | style | status | cast | shots | est. | renders |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for production in self.list_productions():
            rows.append(
                f"| [`{production.id}`]({production.id}.md) | {production.title} "
                f"| {production.format} | {production.style} | {production.status} "
                f"| {len(production.cast)} | {len(production.shots)} "
                f"| {production.estimated_duration:.0f}s | {len(production.renders)} |"
            )
        path = self.root / "INDEX.md"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path


class CharacterStore:
    """Reusable cast members, one markdown file each.

    Casting decisions are written back here so a character keeps the same voice
    across every production they appear in -- consistency by persistence rather
    than by re-deriving it identically each time.
    """

    def __init__(self, root: Path = Path("characters")) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, character_id: str) -> Path:
        return self.root / f"{character_id}.md"

    def save(self, character: Character) -> Path:
        path = self.path_for(character.id)
        front = yaml.safe_dump(
            character.to_dict(), sort_keys=False, allow_unicode=True, width=88
        )
        body = [f"# {character.name}", ""]
        if character.role:
            body.append(f"**Role:** {character.role}")
        if character.description:
            body.append(f"\n{character.description}")
        if character.appearance:
            body.append(f"\n**Appearance:** {character.appearance}")
        if character.reference_image:
            body.append(f"\n![{character.name}]({Path(character.reference_image).name})")
        path.write_text(f"---\n{front}---\n\n" + "\n".join(body) + "\n", encoding="utf-8")
        return path

    def load(self, character_id: str) -> Character:
        path = self.path_for(character_id)
        if not path.exists():
            raise FileNotFoundError(f"no character at {path}")
        front, _ = split_frontmatter(path.read_text(encoding="utf-8"))
        return Character.from_dict(front)

    def list_characters(self) -> List[Character]:
        characters: List[Character] = []
        for path in sorted(self.root.glob("*.md")):
            if path.name == "INDEX.md":
                continue
            try:
                front, _ = split_frontmatter(path.read_text(encoding="utf-8"))
                if front.get("name"):
                    characters.append(Character.from_dict(front))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("skipping unreadable character %s: %s", path.name, exc)
        return characters

    def find(self, name: str) -> Optional[Character]:
        key = character_key(name)
        return next((c for c in self.list_characters() if c.key == key), None)

    def import_reference(self, character: Character, image_path: Path) -> Character:
        """Copy a user-supplied reference image in beside the character file."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"no image at {image_path}")
        destination = self.root / f"{character.id}{image_path.suffix.lower()}"
        if image_path.resolve() != destination.resolve():
            shutil.copy2(image_path, destination)
        character.reference_image = str(destination)
        self.save(character)
        return character

    def merge_into(self, production: Production) -> Production:
        """Adopt stored details for any cast member already on file.

        Only fills gaps -- a production that specifies its own appearance or
        voice keeps it, so a stored character never silently overrides the
        script in front of it.
        """
        for i, member in enumerate(production.cast):
            stored = self.find(member.name)
            if stored is None:
                continue
            if not member.description:
                member.description = stored.description
            if not member.appearance:
                member.appearance = stored.appearance
            if not member.role:
                member.role = stored.role
            if not member.voice.voice:
                member.voice = stored.voice
            member.reference_image = member.reference_image or stored.reference_image
            member.identity_ref = member.identity_ref or stored.identity_ref
            member.lora = member.lora or stored.lora
            member.palette = member.palette or stored.palette
            production.cast[i] = member
        return production

    def remember(self, production: Production) -> None:
        """Persist every cast member of a production for reuse."""
        for member in production.cast:
            self.save(member)
