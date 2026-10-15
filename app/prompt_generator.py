from __future__ import annotations

import os
import re

import httpx

from .config import Settings
from .prompting import DEFAULT_NEGATIVE, ensure_system_prefix, validate_xml_prompt
from .schemas import DraftPromptRequest, ModelProfile


class PromptGeneratorUnavailable(RuntimeError):
    pass


SYSTEM = """You convert a user's image description into one English XML prompt for the NewBie image model.
Return XML only, rooted at <img>. Include <character_1> and <general_tags>. Use concise Danbooru-compatible
English tags, one coherent style family, and no artist-name pileups. Put very_aesthetic, masterpiece, no_text
in <quality> and max_high_resolution in <resolution>. Never hide hands when the user requests visible hands.
Do not add characters, props, expressions, actions, clothing, text, or scenery that the user did not request."""

NATURAL_SYSTEM = """You write one precise English image-generation prompt. Preserve the requested character identity,
canonical design, clothing, action, camera and scene. Do not add unrequested props, text, characters or style families.
When hands are requested, explicitly keep both complete hands inside frame with a simple anatomically testable pose.
Return the prompt only, with no markdown or explanation."""


def _extract_xml(text: str) -> str:
    cleaned = re.sub(r"^```(?:xml)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start = cleaned.find("<img>")
    end = cleaned.rfind("</img>")
    if start < 0 or end < start:
        raise ValueError("大模型没有返回完整的 <img> XML")
    return cleaned[start : end + len("</img>")]


async def draft_prompt(payload: DraftPromptRequest, settings: Settings) -> dict[str, object]:
    if not settings.llm_base_url or not settings.llm_model:
        raise PromptGeneratorUnavailable("尚未在 config/settings.json 配置 llm_base_url 与 llm_model")
    key = os.getenv(settings.llm_api_key_env, "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    user = (
        f"Character: {payload.character or 'original character'}\n"
        f"Description: {payload.description}\nStyle: {payload.style}\n"
        f"Both hands must be visible: {'yes' if payload.visible_hands else 'not required'}"
    )
    system_prompt = SYSTEM if payload.model_id == ModelProfile.NEWBIE else NATURAL_SYSTEM
    body = {
        "model": settings.llm_model,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
        "temperature": 0.2,
        "max_tokens": 8000,
    }
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(url, headers=headers, json=body)
        if response.is_error:
            raise RuntimeError(f"提示词模型调用失败：{response.status_code} {response.text[:500]}")
        content = response.json()["choices"][0]["message"]["content"]
    if payload.model_id == ModelProfile.NEWBIE:
        xml = _extract_xml(content)
        warnings = validate_xml_prompt(xml)
        if any(item.startswith("XML 解析失败") for item in warnings):
            raise ValueError("提示词模型返回了无效 XML")
        return {"prompt": xml, "full_prompt": ensure_system_prefix(xml), "negative_prompt": DEFAULT_NEGATIVE, "warnings": warnings}
    prompt = re.sub(r"^```(?:text)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    negative = (
        "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration"
        if payload.model_id == ModelProfile.ANIMA_TURBO
        else "blurry, ugly, bad anatomy, bad hands, text, watermark"
    )
    return {"prompt": prompt, "full_prompt": prompt, "negative_prompt": negative, "warnings": []}
