from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from .config import Settings


@dataclass(frozen=True, slots=True)
class StylePreset:
    id: str
    name: str
    mode: str
    artist: str
    style: str


def _slug(value: str) -> str:
    ascii_part = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return ascii_part or "style-" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def load_style_catalog(settings: Settings) -> list[StylePreset]:
    """Load only the non-secret style section from NewBie-LLM-Formatter.

    The studio remains independent from the ComfyUI desktop process: this reads a
    JSON file and never imports or executes the custom node package.
    """
    path = Path(settings.style_catalog_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        styles = raw.get("styles", {})
    except (OSError, ValueError, TypeError):
        styles = {}
    result: list[StylePreset] = []
    used: set[str] = set()
    for name, value in styles.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            continue
        mode = str(value.get("mode", "Both")).strip().lower()
        mode = mode if mode in {"newbie", "anima", "both"} else "both"
        preset_id = _slug(name)
        suffix = 2
        while preset_id in used:
            preset_id = f"{_slug(name)}-{suffix}"
            suffix += 1
        used.add(preset_id)
        result.append(StylePreset(
            id=preset_id,
            name=name,
            mode=mode,
            artist=str(value.get("artist", "")).strip(),
            style=str(value.get("style", "")).strip(),
        ))
    return result


def style_catalog_json(settings: Settings) -> list[dict[str, str]]:
    return [
        {"id": item.id, "name": item.name, "mode": item.mode,
         "artist": item.artist, "style": item.style}
        for item in load_style_catalog(settings)
    ]


def find_style(settings: Settings, preset_id: str) -> StylePreset | None:
    return next((item for item in load_style_catalog(settings) if item.id == preset_id), None)


def _artist_parts(value: str) -> list[tuple[str, float | None]]:
    """Parse the three weighting syntaxes supported by LLM-Formatter."""
    result: list[tuple[str, float | None]] = []
    for part in value.split(","):
        cleaned = re.sub(r"(?i)^artist:\s*", "", part.strip())
        weight: float | None = None
        match = re.match(r"^([\d.]+)::(.+)::$", cleaned)
        if match:
            weight, cleaned = float(match.group(1)), match.group(2).strip()
        else:
            match = re.match(r"^\((.+):([\d.]+)\)$", cleaned)
            if match:
                cleaned, weight = match.group(1).strip(), float(match.group(2))
            else:
                scale = 1.0
                while len(cleaned) >= 2 and ((cleaned[0], cleaned[-1]) in {("(", ")"), ("[", "]")}):
                    scale *= 1.2 if cleaned[0] == "(" else 0.9
                    cleaned = cleaned[1:-1].strip()
                if scale != 1.0:
                    weight = round(scale, 2)
        item = (cleaned, weight)
        if cleaned and item not in result:
            result.append(item)
    return result


def _format_artists(value: str, anima: bool) -> str:
    formatted: list[str] = []
    for content, weight in _artist_parts(value):
        # Recipe sources may already escape literal parentheses. Canonicalize
        # those name escapes once; adding another backslash changes the text
        # reaching the encoder. Weighting wrappers are added separately below.
        content = re.sub(r"\\*([()])", r"\\\1", content)
        if anima:
            content = "@" + content.replace("_", " ").removeprefix("@")
        formatted.append(f"({content}:{weight:g})" if weight is not None and weight != 1 else content)
    return ("use the fixed style of " + " and ".join(formatted)) if anima and formatted else ", ".join(formatted)


def inject_newbie_style(xml_prompt: str, preset: StylePreset, *, preserve_current_style: bool = False) -> str:
    try:
        start, end = xml_prompt.index("<img>"), xml_prompt.rindex("</img>") + len("</img>")
        root = ET.fromstring(xml_prompt[start:end])
    except (ValueError, ET.ParseError):
        return xml_prompt
    general = root.find("general_tags")
    if general is None:
        general = ET.SubElement(root, "general_tags")
    for tag, text in (("style", preset.style), ("artist", _format_artists(preset.artist, anima=False))):
        if not text:
            continue
        node = general.find(tag)
        if node is None:
            node = ET.SubElement(general, tag)
        if tag == 'style' and preserve_current_style and node.text and node.text != text:
            node.text = node.text + ', ' + text
        else:
            node.text = text
    return xml_prompt[:start] + ET.tostring(root, encoding="unicode", short_empty_elements=False) + xml_prompt[end:]


def inject_anima_style(prompt: str, preset: StylePreset) -> str:
    additions: list[str] = []
    artists = _format_artists(preset.artist, anima=True)
    if artists:
        additions.append(artists)
    if preset.style:
        additions.append(preset.style)
    return ", ".join([part for part in (prompt.strip(), *additions) if part])


def inject_tag_style(prompt: str, preset: StylePreset) -> str:
    """Inject a Danbooru/plain-tag recipe for SDXL-like and generic routes."""
    additions = [_format_artists(preset.artist, anima=False), preset.style]
    return ", ".join([part for part in (prompt.strip(), *additions) if part])
