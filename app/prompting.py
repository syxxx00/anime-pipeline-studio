from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree

from .schemas import TestMode


SYSTEM_PREFIX = (
    "You are an assistant designed to generate high-quality anime images with "
    "the highest degree of image-text alignment based on xml format textual prompts. "
    "<Prompt Start>\n"
)

DEFAULT_NEGATIVE = (
    "<danbooru_tags>furry, english text, chinese text, korean text, speech bubble, "
    "dated, logo, signature, watermark, web address, artist name, character name, "
    "copyright name, twitter username, low score rate, worst quality, low quality, "
    "bad quality, lowres, pixelated, blurry, compression artifacts, jpeg artifacts, "
    "bad anatomy, worst hands, deformed hands, deformed fingers, deformed feet, "
    "extra limbs, extra arms, extra legs, extra fingers, fused fingers, missing limbs, "
    "wrong hands, twisted hands, abstract, sequence, lineup, 2koma, 4koma, artifacts, "
    "adversarial noise, resized, image sample, low aesthetic</danbooru_tags>\n"
    "<resolution>low_resolution</resolution>"
)


@dataclass(frozen=True, slots=True)
class Preset:
    id: str
    name: str
    purpose: str
    prompt: str
    model_id: str = "newbie"


PRESETS: tuple[Preset, ...] = (
    Preset(
        id="baseline",
        name="官方最小基线",
        purpose="确认模型、文本编码器、采样与 VAE 是否正常",
        prompt="""<img>
<character_1>
<n>original_character</n>
<gender>1girl, solo</gender>
<appearance>grey_hair, red_eyes, shoulder_length_hair</appearance>
<clothing>black_jacket, white_shirt, black_skirt</clothing>
<expression>neutral_expression, closed_mouth</expression>
<action>standing</action>
<position>center</position>
</character_1>
<general_tags>
<count>1girl, solo</count>
<style>anime_style, game_illustration, clean_lineart, cel_shading</style>
<background>simple_background</background>
<atmosphere>neutral_lighting</atmosphere>
<quality>very_aesthetic, masterpiece, no_text</quality>
<resolution>max_high_resolution</resolution>
<artist></artist>
<objects></objects>
<other></other>
</general_tags>
</img>""",
    ),
    Preset(
        id="hands",
        name="双手压力测试",
        purpose="让双手占据足够像素，禁止用袖口、背手或裁切掩盖失败",
        prompt="""<img>
<character_1>
<n>original_character</n>
<gender>1girl, solo</gender>
<appearance>grey_hair, red_eyes, shoulder_length_hair</appearance>
<clothing>short_sleeves, simple_shirt</clothing>
<expression>neutral_expression, looking_at_viewer</expression>
<action>upper_body, both_hands_visible, open_hands, spread_fingers, hands_in_front</action>
<position>center, front_view</position>
</character_1>
<general_tags>
<count>1girl, solo</count>
<style>anime_style, clean_lineart, flat_color</style>
<background>plain_light_background</background>
<atmosphere>even_lighting</atmosphere>
<quality>very_aesthetic, masterpiece, no_text</quality>
<resolution>max_high_resolution</resolution>
<artist></artist>
<objects></objects>
<other>hands_focus, detailed_hands</other>
</general_tags>
</img>""",
    ),
    Preset(
        id="specter",
        name="幽灵鲨角色基线",
        purpose="以现有角色 LoRA 检查身份保持，动作保持简单且双手可见",
        prompt="""<img>
<character_1>
<n>specter_the_unchained_(arknights)</n>
<gender>1girl, solo</gender>
<appearance>long_white_hair, red_eyes, pale_skin</appearance>
<clothing>black_nun_outfit, black_dress, cross_necklace</clothing>
<expression>calm, closed_mouth, looking_at_viewer</expression>
<action>upper_body, both_hands_visible, relaxed_hands</action>
<position>center, front_view</position>
</character_1>
<general_tags>
<count>1girl, solo</count>
<style>anime_style, game_illustration, clean_lineart, cel_shading</style>
<background>simple_background</background>
<atmosphere>soft_lighting</atmosphere>
<quality>very_aesthetic, masterpiece, no_text</quality>
<resolution>max_high_resolution</resolution>
<artist></artist>
<objects></objects>
<other>arknights</other>
</general_tags>
</img>""",
    ),
    Preset(
        id="w_arknights",
        name="W 角色基线",
        purpose="不使用复杂场景，先验证 NewBie 基模的角色知识和手部结构",
        prompt="""<img>
<character_1>
<n>w_(arknights)</n>
<gender>1girl, solo</gender>
<appearance>grey_hair, short_hair, red_eyes, sarkaz_horns</appearance>
<clothing>black_jacket, open_jacket, white_shirt, black_skirt, black_pantyhose, fingerless_gloves</clothing>
<expression>smirk, looking_at_viewer</expression>
<action>upper_body, both_hands_visible, relaxed_hands</action>
<position>center, front_view</position>
</character_1>
<general_tags>
<count>1girl, solo</count>
<style>anime_style, game_illustration, clean_lineart, cel_shading</style>
<background>simple_background</background>
<atmosphere>soft_lighting</atmosphere>
<quality>very_aesthetic, masterpiece, no_text</quality>
<resolution>max_high_resolution</resolution>
<artist></artist>
<objects></objects>
<other>arknights</other>
</general_tags>
</img>""",
    ),
    Preset(
        id="z_w_arknights",
        name="Z-Image · W 角色压力测试",
        purpose="使用自然语言测试通用模型的 W 身份知识、无文字遵从与双手结构",
        model_id="z_image_turbo",
        prompt="""A clean anime game character illustration of W from Arknights, solo, upper body, front view. She has short silver-gray hair, vivid red eyes, black Sarkaz horns, a mischievous smirk, and her canonical black, white and red tactical outfit with fingerless gloves. Both hands are fully visible in front of her with relaxed natural fingers. Plain light-gray studio background, even soft lighting, crisp clean line art, restrained cel shading, accurate character design, no typography, no logo, no frame.""",
    ),
    Preset(
        id="anima_w_arknights",
        name="Anima Turbo · W 角色压力测试",
        purpose="使用原生二次元新模型测试 W 的角色知识、双手和严格服装还原",
        model_id="anima_turbo",
        prompt="""masterpiece, best quality, score_7, safe, newest, 1girl, solo, W (Arknights), Arknights, short silver-gray hair, red eyes, black Sarkaz horns, mischievous smirk, canonical black white and red tactical outfit, fingerless gloves, upper body, front view, both hands fully visible, relaxed natural fingers, plain light-gray background, even soft lighting, clean lineart, cel shading, no text, no logo, no frame""",
    ),
    Preset(
        id="anima_hands",
        name="Anima Turbo · 双手压力测试",
        purpose="用与 NewBie 相同的正面张掌构图测试手掌、手腕和指长",
        model_id="anima_turbo",
        prompt="""masterpiece, best quality, score_7, safe, newest, 1girl, solo, original character, shoulder-length gray hair, red eyes, short-sleeved simple shirt, upper body, front view, both complete hands in front of the shoulders, palms facing viewer, five clearly separated fingers on each hand, natural finger lengths, anatomically connected wrists, symmetrical pose, plain light background, even lighting, clean lineart, flat cel shading, no text, no logo""",
    ),
)


def ensure_system_prefix(prompt: str) -> str:
    cleaned = prompt.strip()
    if cleaned.startswith(SYSTEM_PREFIX.strip()):
        return cleaned
    return SYSTEM_PREFIX + cleaned


def validate_xml_prompt(prompt: str) -> list[str]:
    warnings: list[str] = []
    xml = prompt.strip()
    if "<Prompt Start>" in xml:
        xml = xml.split("<Prompt Start>", 1)[1].strip()
    if xml.startswith("{") and xml.endswith("}"):
        xml = xml[1:-1].strip()
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        return [f"XML 解析失败：{exc}"]
    if root.tag not in {"img", "image"}:
        warnings.append("根标签建议使用 <img> 或 <image>")
    if root.find("character_1") is None:
        warnings.append("缺少 <character_1>")
    general = root.find("general_tags")
    if general is None:
        warnings.append("缺少 <general_tags>")
    else:
        quality = general.findtext("quality", "")
        resolution = general.findtext("resolution", "")
        if "very_aesthetic" not in quality or "masterpiece" not in quality:
            warnings.append("质量字段未包含 very_aesthetic, masterpiece")
        if "max_high_resolution" not in resolution:
            warnings.append("分辨率字段未包含 max_high_resolution")
    risky = {
        "multiple_expressions": "多表情会干扰单一面部结构",
        "maximalist detail": "极繁细节容易在瓦砾/植被场景产生碎片化",
        "delicate floral details": "花卉细节可能污染非花卉场景",
        "holding_blade": "持械会显著提高双手结构难度",
    }
    lower = prompt.lower()
    for token, message in risky.items():
        if token in lower:
            warnings.append(message)
    return warnings


def matrix_prompt(base: str, variant: str) -> str:
    additions = {
        "identity": ("other", "character_reference_accuracy, consistent_character_design"),
        "hands": ("other", "hands_focus, both_hands_visible, open_hands, spread_fingers"),
        "scene": ("background", "simple_outdoor_background, distant_sky"),
        "style": ("style", "anime_style, clean_lineart, cel_shading"),
    }
    if variant not in additions:
        raise ValueError(f"未知矩阵变量: {variant}")
    xml = base.strip()
    prefix = ""
    if "<Prompt Start>" in xml:
        prefix, xml = xml.split("<Prompt Start>", 1)
        prefix = prefix + "<Prompt Start>\n"
        xml = xml.strip()
    root = ElementTree.fromstring(xml)
    general = root.find("general_tags")
    if general is None:
        raise ValueError("矩阵测试需要 <general_tags>")
    field, content = additions[variant]
    node = general.find(field)
    if node is None:
        node = ElementTree.SubElement(general, field)
    node.text = content
    return prefix + ElementTree.tostring(root, encoding="unicode")


def natural_matrix_prompt(base: str, variant: str) -> str:
    additions = {
        "identity": " Preserve the exact canonical character design and signature accessories.",
        "hands": " Both complete hands are inside frame, palms visible, fingers spread in a simple symmetrical pose.",
        "scene": " Use only a plain outdoor sky background with no extra objects.",
        "style": " Use one consistent clean anime cel-shading style with crisp line art.",
    }
    if variant not in additions:
        raise ValueError(f"未知矩阵变量: {variant}")
    return base.rstrip() + additions[variant]


def list_presets() -> list[dict[str, str]]:
    return [
        {"id": item.id, "name": item.name, "purpose": item.purpose, "prompt": item.prompt, "model_id": item.model_id}
        for item in PRESETS
    ]
