from __future__ import annotations

import json
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree.ElementTree import Element, SubElement, tostring

from .config import Settings
from .prompting import DEFAULT_NEGATIVE
from .schemas import NewbieObjectSpec, NewbieRelationSpec, NewbieSemanticSpec, NewbieStyleMode
from .intent_guards import matches, preserve_explicit_intent

if TYPE_CHECKING:
    from .scene_contract import SceneContractV3


CHARACTERS = {
    "w": {
        "aliases": ("明日方舟的w", "明日方舟w", "w_(arknights)", "w from arknights", "角色w"),
        "name": "W", "tag": "w_(arknights)", "source": "arknights",
        "appearance": [], "clothing": [],
        "expression": ["mischievous smile"],
    },
    "specter": {
        "aliases": ("幽灵鲨", "specter", "specter_(arknights)"),
        "name": "幽灵鲨", "tag": "specter_(arknights)", "source": "arknights",
        "appearance": [], "clothing": [],
        "expression": ["calm expression"],
    },
    "specter_unchained": {
        "aliases": ("归溟幽灵鲨", "specter the unchained", "specter_the_unchained_(arknights)"),
        "name": "归溟幽灵鲨", "tag": "specter_the_unchained_(arknights)", "source": "arknights",
        "appearance": [], "clothing": [],
        "expression": ["calm expression"],
    },
    "wisadel": {
        "aliases": ("维什戴尔", "wisadel", "wisadel_(arknights)"),
        "name": "维什戴尔", "tag": "wisadel_(arknights)", "source": "arknights",
        "appearance": [], "clothing": [],
        "expression": ["confident expression"],
    },
    "amiya": {
        "aliases": ("阿米娅", "amiya"),
        "name": "阿米娅", "tag": "amiya_(arknights)", "source": "arknights",
        "appearance": [], "clothing": [],
        "expression": ["gentle determined expression"],
    },
    "viviana": {
        "aliases": ("薇薇安娜", "viviana", "viviana_(arknights)"),
        "name": "薇薇安娜", "tag": "viviana_(arknights)", "source": "arknights",
        "appearance": [], "clothing": [],
        "expression": ["composed expression"],
    },
    "kangel": {
        "aliases": ("超天酱", "超てんちゃん", "kangel", "omgkawaiiangel"),
        "name": "超天酱", "tag": "omgkawaiiangel_(needy_streamer_overload)", "source": "needy streamer overload",
        # These former blue twin-tail/outfit defaults were unsourced and
        # reinforced the Miku-like drift observed in job 1b94b98eb2c4.
        # Keep the identity tag, but do not invent replacement visual facts.
        "appearance": [], "clothing": [],
        "expression": ["bright idol smile"],
    },
    "misaka": {
        "aliases": ("御坂美琴", "misaka mikoto", "mikoto misaka"),
        "name": "御坂美琴", "tag": "misaka_mikoto", "source": "toaru kagaku no railgun",
        "appearance": [], "clothing": [],
        "expression": ["confident expression"],
    },
}


BASE_STYLE = ["detailed anime game illustration", "standard anime character proportions", "refined lineart", "soft painterly cel shading", "detailed face and costume"]
SEMANTIC_INFERENCE_LOCK = threading.Lock()

POSE_TOKENS = ("坐", "站", "跑", "奔跑", "躺", "跪", "蹲", "倚", "行走", "远眺", "眺望", "远望", "sitting", "standing", "running", "lying", "kneeling", "walking", "gazing")
HAND_ACTION_TOKENS = ("手", "手指", "掌心", "拿", "握", "按", "抱", "捧", "抓", "挥", "holding", "hand", "finger", "thumb", "grip", "press", "carry", "hug")
SCENE_TOKENS = ("礁石", "海边", "海岸", "海面", "海洋", "房间", "室内", "街道", "城市", "仓库", "森林", "天空", "屋顶", "教堂", "车站", "seashore", "coast", "ocean", "room", "street", "warehouse", "forest", "sky", "rooftop", "church", "station")
ABSTRACT_TOKENS = ("像", "仿佛", "似乎", "如同", "把意识", "尚未开口", "metaphor", "like a")
OBJECT_TOKENS = ("起爆器", "引爆器", "炸弹", "提灯", "书", "巨剑", "剑", "圆锯", "任务报告", "文件袋", "心形抱枕", "硬币", "怀表", "玻璃瓶", "瓶", "白花", "收音机", "海鸟", "帆船", "灯塔", "detonator", "bomb", "lantern", "book", "sword", "circular saw", "report", "document folder", "heart cushion", "coin", "watch", "bottle", "flower", "radio", "bird", "sailboat", "lighthouse")
OBJECT_GROUPS = (("起爆器", "引爆器", "detonator"), ("炸弹", "bomb"), ("提灯", "lantern"), ("书", "book"), ("巨剑", "剑", "sword"), ("圆锯", "circular saw"), ("任务报告", "report"), ("文件袋", "document folder"), ("心形抱枕", "heart cushion"), ("硬币", "coin"), ("怀表", "watch"), ("玻璃瓶", "瓶", "bottle"), ("白花", "flower"), ("收音机", "radio"), ("海鸟", "bird"), ("帆船", "sailboat"), ("灯塔", "lighthouse"))
OBJECT_GROUPS += (("显示器", "monitor", "computer display"), ("手机", "phone", "smartphone"))
OBJECT_TOKENS += tuple(alias for group in OBJECT_GROUPS[-2:] for alias in group)


def _has_any(text: str, tokens: tuple[str, ...]) -> bool:
    return matches(text, tokens)


def _dedupe_limit(items: list[str], limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = re.sub(r"\s+", " ", item.strip())
        key = cleaned.lower().rstrip(".,; ")
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _character(sentence: str) -> dict:
    lower = sentence.lower().replace(" ", "")
    matches: list[tuple[int, dict]] = []
    for item in CHARACTERS.values():
        for alias in item["aliases"]:
            normalized = alias.lower().replace(" ", "")
            if normalized in lower:
                matches.append((len(normalized), item))
    if matches:
        # Longer aliases carry more identity information: 归溟幽灵鲨 must not
        # be captured by the shorter 幽灵鲨 alias.
        return max(matches, key=lambda match: match[0])[1]
    return {
        "name": "original character", "tag": "original_character", "source": "",
        "appearance": [], "clothing": [], "expression": ["natural expression"],
    }


def fallback_semantic(sentence: str, style_mode: NewbieStyleMode) -> NewbieSemanticSpec:
    char = _character(sentence)
    lower = sentence.lower()
    # Rendering safeguards for a requested 1girl, not claims about a specific
    # character's canon design. Canonical facts only enter through the audited
    # knowledge pack.
    appearance = _dedupe_limit([
        *char["appearance"],
        "delicate feminine face",
        "soft tapered jawline",
        "balanced feminine facial proportions",
    ], 12)
    action = ["standing"]
    framing = ["medium shot", "front view", "centered composition"]
    background = ["simple coherent background"]
    left_hand: list[str] = []
    right_hand: list[str] = []
    objects: list[NewbieObjectSpec] = []
    relations: list[NewbieRelationSpec] = []

    if _has_any(lower, ("全身", "full body")):
        framing = ["full body", "front view", "entire figure inside frame"]
    elif _has_any(lower, ("半身", "腰部以上", "medium shot", "waist up")):
        framing = ["waist-up medium shot", "front view", "hands and held object large enough to inspect"]
    elif _has_any(lower, ("近景", "特写", "close-up", "portrait")):
        framing = ["upper body portrait", "front view", "centered composition"]
    if _has_any(lower, ("大远景", "远景", "wide environmental", "extreme wide")):
        framing = ["wide environmental shot", "vast environment dominates the frame", "character fully visible but small in the landscape"]
    elif _has_any(lower, ("环境为主", "以景为主", "以环境为", "environment-focused", "environment dominates")):
        # "环境为主"没有明说远景但意图相同：环境占画面主导 → 复用大远景构图
        framing = ["wide environmental shot", "vast environment dominates the frame", "character fully visible but small in the landscape"]
    if _has_any(lower, ("微微侧身", "侧身", "侧对", "side profile", "three-quarter")):
        framing = framing + ["slight three-quarter side view"]
    if _has_any(lower, ("俯视", "俯瞰", "top-down", "high angle")):
        framing = framing + ["high angle view looking down"]
    if _has_any(lower, ("仰视", "low angle")):
        framing = framing + ["low angle view looking up"]
    has_sitting = _has_any(lower, ("坐", "sitting", "sit"))
    has_reef = _has_any(lower, ("礁石", "reef rock", "rocky outcrop"))
    has_coast = _has_any(lower, ("海边", "海岸", "海面", "seashore", "coast", "ocean"))
    if has_sitting:
        action = ["sitting naturally"]
    if _has_any(lower, ("跑", "奔跑", "running")):
        action = ["running", "balanced dynamic pose"]
    if _has_any(lower, ("跃起", "跳", "跳跃", "jump")):
        action = ["jumping dynamically", "airborne pose with clear balance"]
    if _has_any(lower, ("踢", "kick")):
        action = action + ["kicking with one leg", "stable supporting leg"] if action != ["standing"] else ["kicking with one leg", "stable supporting leg"]
    if _has_any(lower, ("侧躺", "躺在", "lying")):
        action = ["lying down naturally", "relaxed reclining pose"]
    if _has_any(lower, ("趴在", "lying face down")):
        action = ["leaning forward with upper body resting on the surface", "relaxed prone posture"]
    if _has_any(lower, ("靠在", "leaning")):
        action = ["leaning against the surface", "relaxed standing lean"]
    if _has_any(lower, ("盘腿", "cross-legged")):
        action = ["sitting cross-legged", "balanced grounded posture"]
    if _has_any(lower, ("比心", "heart hands")):
        action = action + ["forming a heart shape with both hands toward the viewer"] if action != ["standing"] else ["facing the viewer", "forming a heart shape with both hands"]
        left_hand = ["left hand forms half of the heart shape", "five anatomically connected fingers"]
        right_hand = ["right hand forms the other half of the heart shape", "five anatomically connected fingers"]
    if _has_any(lower, ("走", "walking")) and action == ["standing"]:
        action = ["walking naturally", "mid-stride pose"]
    if _has_any(lower, ("回头", "looking back")):
        action.append("looking back over shoulder")
    if _has_any(lower, ("远眺", "眺望", "远望", "gazing into")):
        action.append("gazing into the distance")
    if _has_any(lower, ("手", "拿", "握", "按", "holding", "hand")):
        left_hand = ["left hand visible", "natural fingers"]
        right_hand = ["right hand visible", "natural fingers"]
    if _has_any(lower, ("起爆器", "引爆器", "detonator")):
        objects.append(NewbieObjectSpec(
            id="detonator", name="remote detonator",
            appearance=["small rectangular black handheld device", "one clearly visible red push button", "rigid manufactured edges"],
            position=["fully visible in right hand", "separated from clothing"],
        ))
        right_hand = ["right hand firmly holds the sides of the detonator", "right thumb rests on the red button", "five anatomically connected fingers"]
        relations.append(NewbieRelationSpec(subject="right thumb", predicate="presses", object="detonator red button"))
    if _has_any(lower, ("炸弹", "bomb")):
        objects.append(NewbieObjectSpec(
            id="bomb", name="compact demolition charge",
            appearance=["clearly manufactured explosive device", "rectangular casing", "visible wire and indicator light"],
            position=["separate foreground object", "not fused with the character"],
        ))
    if _has_any(lower, ("圆锯", "circular saw")):
        objects.append(NewbieObjectSpec(
            id="circular_saw", name="large industrial circular saw weapon",
            appearance=["one complete circular toothed metal blade", "separate rigid handle and motor housing", "clearly connected mechanical parts"],
            position=["fully visible in front of the character", "held with both hands at the handle", "not fused with arms or clothing"],
        ))
        left_hand = ["left hand grips the circular saw handle", "five anatomically connected fingers"]
        right_hand = ["right hand grips the circular saw handle", "five anatomically connected fingers"]
        relations.extend([
            NewbieRelationSpec(subject="left hand", predicate="grips", object="circular_saw.handle"),
            NewbieRelationSpec(subject="right hand", predicate="grips", object="circular_saw.handle"),
        ])
    if _has_any(lower, ("任务报告", "报告纸", "mission report", "written report")):
        objects.append(NewbieObjectSpec(
            id="mission_report", name="paper mission report",
            appearance=["several clean rectangular paper sheets", "clear unbroken paper edges", "blank surface without readable text"],
            position=["held in front of the character", "separate from desk and clothing"],
        ))
        left_hand = ["left hand holds the left edge of the report", "five anatomically connected fingers"]
        right_hand = ["right hand holds the right edge of the report", "five anatomically connected fingers"]
        relations.extend([
            NewbieRelationSpec(subject="left hand", predicate="holds", object="mission_report.left_edge"),
            NewbieRelationSpec(subject="right hand", predicate="holds", object="mission_report.right_edge"),
        ])
    if _has_any(lower, ("文件袋", "document folder")):
        objects.append(NewbieObjectSpec(
            id="document_folder", name="closed document folder",
            appearance=["flat rectangular paper folder", "clean closed edges", "no readable text"],
            position=["held securely against the torso", "fully distinct from clothing"],
        ))
        left_hand = ["left arm supports the document folder", "left hand visible at its lower edge"]
        right_hand = ["right hand steadies the side of the document folder", "natural fingers"]
        relations.append(NewbieRelationSpec(subject="both arms", predicate="hold", object="document_folder"))
    if _has_any(lower, ("心形抱枕", "心形枕", "heart cushion", "heart-shaped cushion")):
        objects.append(NewbieObjectSpec(
            id="heart_cushion", name="heart-shaped cushion",
            appearance=["one complete symmetrical pink heart shape", "soft fabric volume", "clean continuous outline"],
            position=["hugged in front of the chest", "fully visible and separate from the body"],
        ))
        left_hand = ["left arm wraps around the heart cushion", "left hand visible"]
        right_hand = ["right arm wraps around the heart cushion", "right hand visible"]
        relations.append(NewbieRelationSpec(subject="both arms", predicate="hug", object="heart_cushion"))
    if _has_any(lower, ("硬币", "coin")):
        objects.append(NewbieObjectSpec(
            id="coin", name="single metal coin",
            appearance=["one small round metal coin", "clear circular silhouette", "thin metallic rim"],
            position=["directly above the right thumb", "not hidden by light effects"],
        ))
        right_hand = ["right thumb directly below the coin in a flicking pose", "other fingers naturally curled", "five anatomically connected fingers"]
        relations.append(NewbieRelationSpec(subject="right thumb", predicate="directly below", object="coin"))
    if _has_any(lower, ("小提琴", "violin")):
        objects.append(NewbieObjectSpec(
            id="violin", name="wooden violin with bow",
            appearance=["brown wooden violin body", "separate slender bow", "four visible strings"],
            position=["violin rests on left shoulder", "bow held in right hand"],
        ))
        left_hand = ["left hand fingers press the violin fingerboard", "five anatomically connected fingers"]
        right_hand = ["right hand holds the bow by its grip", "five anatomically connected fingers"]
    if _has_any(lower, ("拉面", "筷子", "noodles", "ramen", "chopsticks")):
        objects.append(NewbieObjectSpec(
            id="noodles", name="bowl of ramen with chopsticks",
            appearance=["steaming bowl of noodles", "pair of wooden chopsticks"],
            position=["bowl on the table or held in left hand", "chopsticks held in right hand"],
        ))
        if not right_hand or "natural" in " ".join(right_hand):
            right_hand = ["right hand holds chopsticks naturally", "five anatomically connected fingers"]
    if _has_any(lower, ("手机", "phone", "smartphone")):
        objects.append(NewbieObjectSpec(
            id="phone", name="modern smartphone",
            appearance=["flat rectangular smartphone", "glowing screen"],
            position=["held in one hand at readable distance"],
        ))
        if not right_hand or "natural" in " ".join(right_hand):
            right_hand = ["right hand holds the smartphone", "fingers wrap around its edges", "five anatomically connected fingers"]
    if _has_any(lower, ("巨剑", "大剑", "greatsword", "broadsword")):
        objects.append(NewbieObjectSpec(
            id="greatsword", name="large two-handed greatsword",
            appearance=["broad heavy blade", "long two-handed grip", "metallic edge highlights"],
            position=["gripped by both hands", "clearly separate from the body silhouette"],
        ))
        left_hand = ["left hand grips the lower half of the sword hilt", "five anatomically connected fingers"]
        right_hand = ["right hand grips the upper half of the sword hilt", "five anatomically connected fingers"]
    if _has_any(lower, ("猫", "cat")):
        objects.append(NewbieObjectSpec(
            id="cat", name="small cat",
            appearance=["soft fur", "clearly defined head, body and tail"],
            position=["held securely in the character's arms", "in natural contact with the character"],
        ))
        left_hand = ["left hand supports the cat from below", "natural fingers"]
        right_hand = ["right hand gently holds the cat", "natural fingers"]
    if _has_any(lower, ("自动售货机", "售货机", "vending machine")):
        objects.append(NewbieObjectSpec(
            id="vending_machine", name="drinks vending machine",
            appearance=["tall rectangular vending machine", "lit product display window", "coin slot panel"],
            position=["standing beside the character", "clearly grounded"],
        ))
    if _has_any(lower, ("书", "book", "reading")):
        objects.append(NewbieObjectSpec(
            id="book", name="open hardcover book",
            appearance=["open book with two visible pages", "hard cover"],
            position=["held open with both hands or resting on the lap"],
        ))
        if not left_hand:
            left_hand = ["left hand supports the book", "natural fingers"]
        if not right_hand or "natural" in " ".join(right_hand):
            right_hand = ["right hand holds the book edge", "natural fingers"]
    if _has_any(lower, ("玩偶", "布偶", "plush")):
        objects.append(NewbieObjectSpec(
            id="plush", name="soft plush toy",
            appearance=["rounded soft plush toy", "stitched details"],
            position=["hugged against the character's chest"],
        ))
    if _has_any(lower, ("钢笔", "写信", "信纸", "letter", "pen")):
        objects.append(NewbieObjectSpec(
            id="letter", name="letter paper and pen",
            appearance=["sheet of letter paper on the desk", "slender pen"],
            position=["paper flat on the desk", "pen held in right hand"],
        ))
        right_hand = ["right hand holds the pen above the paper", "writing posture", "five anatomically connected fingers"]
    if _has_any(lower, ("城市", "街道", "city", "street")):
        background = ["coherent city street", "clear ground plane", "recognizable buildings", "no floating debris"]
    elif _has_any(lower, ("室内", "房间", "room", "indoors")):
        background = ["coherent room interior", "clear walls and floor", "few recognizable props"]
    elif _has_any(lower, ("仓库", "warehouse")):
        background = ["dim coherent warehouse interior", "clear floor plane", "recognizable storage shelves", "few background objects", "no floating debris"]
    elif _has_any(lower, ("便利店", "convenience store")):
        background = ["coherent convenience store interior", "shelves with recognizable products", "large window facing the street", "clear floor plane"]
    elif _has_any(lower, ("公园", "park")):
        background = ["coherent park", "trees and walking paths", "park benches", "clear ground plane"]
    elif _has_any(lower, ("天台", "rooftop")):
        background = ["coherent rooftop", "protective railing", "open sky", "distant skyline"]
    elif _has_any(lower, ("草地", "草原", "grass")):
        background = ["open grassy field", "clear natural horizon", "scattered plants"]
    elif _has_any(lower, ("雪原", "snowfield")):
        background = ["vast open snowfield", "snow-covered ground to the horizon", "cold blue-white palette"]
    elif _has_any(lower, ("钟楼", "clock tower")):
        background = ["top of a clock tower", "city rooftops far below", "open sky"]
    elif _has_any(lower, ("沙发", "sofa", "couch")):
        background = ["coherent living room", "soft sofa the character rests on", "a few recognizable household props"]
    elif _has_any(lower, ("湖边", "湖畔", "栈道", "lakeside", "lake")):
        background = ["calm lakeside", "wooden boardwalk over the water", "distant shoreline"]
    elif _has_any(lower, ("废墟", "ruins")):
        background = ["coherent urban ruins", "collapsed concrete structures", "clear ground plane", "dramatic open sky"]
    if has_reef:
        background = ["rocky seashore", "large solid coastal reef rock beneath the character", "open ocean", "clearly visible distant horizon", "coherent natural rock formations"]
        framing = ["medium-wide character-focused environmental shot", "character occupies roughly half of frame height", "entire seated pose visible", "face detectable for later detail pass", "reef rock and ocean horizon inside frame", "eye-level view"]
        if has_sitting:
            framing.insert(1, "entire seated pose visible")
    elif has_coast:
        background = ["coherent seashore", "open ocean", "clearly visible distant horizon"]
        framing = ["medium-wide character-focused environmental shot", "character occupies roughly half of frame height", "face detectable for later detail pass", "character and coastline inside frame", "eye-level view"]
    if _has_any(lower, ("暴风雨的前奏", "暴风雨前", "暴风雨尚未抵达", "风暴尚未开口", "风暴将至", "storm approaching", "before the storm")):
        background += ["towering layered storm clouds gathering over the sea", "dark blue-grey cloud masses", "dramatic breaks between clouds", "clear readable cloud structure"]
    if _has_any(lower, ("风中都带着轻快", "轻快的风", "风是轻快", "海风", "brisk breeze", "sea breeze")):
        background += ["lively sea breeze creating small ripples", "a few loose hair strands and veil edges lifted gently by the wind"]
    storm_approaching = _has_any(lower, ("暴风雨的前奏", "暴风雨前", "暴风雨尚未抵达", "风暴尚未开口", "风暴将至", "storm approaching", "before the storm"))
    if not storm_approaching and _has_any(lower, ("暴雨", "大雨", "倾盆", "雨水", "heavy rain", "downpour")):
        background += ["heavy rain", "clearly visible rain streaks", "wet reflective ground", "splashing raindrops"]
    elif not storm_approaching and _has_any(lower, ("下雨", "雨天", "雨中", "小雨", "rain", "rainy")):
        background += ["rainy weather", "visible falling rain", "wet reflective surfaces"]
    if _has_any(lower, ("暴雪", "暴风雪", "大雪", "下雪", "雪中", "blizzard", "heavy snow", "snowing")):
        background += ["heavy falling snow", "snow accumulating on surfaces", "cold misty air"]
    if _has_any(lower, ("大雾", "浓雾", "雾", "fog", "mist")):
        background += ["dense fog", "very low visibility", "soft diffused silhouettes in the distance"]
    if _has_any(lower, ("闪电", "雷电", "lightning", "thunder")):
        background += ["lightning bolts in the sky", "turbulent storm clouds"]
    if _has_any(lower, ("霓虹", "neon")):
        background += ["glowing neon signs", "colorful neon reflections"]
    lighting = ["soft directional lighting"]
    if _has_any(lower, ("深夜", "夜晚", "夜空", "半夜", "夜里", "晚上", "night", "midnight")):
        lighting = ["night time", "dark night sky", "artificial lights illuminating the scene", "readable night ambience"]
    if _has_any(lower, ("闪电", "雷电", "lightning", "thunder")):
        lighting = lighting + ["dramatic lightning flashes illuminating the scene"] if lighting != ["soft directional lighting"] else ["dramatic lightning illumination", "storm-lit scene"]
    if _has_any(lower, ("日光璀璨", "阳光璀璨", "天边日光", "sunlit horizon", "brilliant sunlight", "glowing horizon")):
        lighting = ["brilliant golden sunlight at the horizon", "sparkling highlights across the ocean", "warm rim light on hair and clothing", "clear luminous sky"]
    elif _has_any(lower, ("黄昏", "暮色", "傍晚", "dusk", "twilight")):
        lighting = ["late dusk", "warm orange last light along the horizon", "cool blue shadows", "dramatic twilight sky", "soft warm rim light on the character"]
    expression = list(char["expression"])
    if _has_any(lower, ("睡着", "入睡", "sleeping", "asleep")):
        expression = ["sleeping peacefully", "closed eyes", "relaxed face"]
    elif _has_any(lower, ("大笑", "laughing")):
        expression = ["open hearty laugh", "joyful expression"]
    if _has_any(lower, ("放空", "发呆", "出神", "平静", "blank mind", "distant gaze", "calmly")):
        expression = ["quiet unfocused distant gaze", "calm relaxed face", "closed mouth", "not looking at viewer"]
    if _has_any(lower, ("雕塑", "statue", "一动不动", "motionless")):
        if has_sitting and has_reef:
            action = ["sitting completely still on the reef rock", "upright relaxed posture", "motionless contemplative pose"]
        elif has_sitting:
            action = ["sitting completely still", "upright relaxed posture", "motionless contemplative pose"]
        else:
            action = ["completely motionless composed pose", "statue-like stillness without literal stone material"]

    mode_style = {
        NewbieStyleMode.STANDARD: BASE_STYLE,
        NewbieStyleMode.CHIBI: ["chibi", "super deformed", "cute anime style", "simple clean shapes"],
        NewbieStyleMode.POSTER: ["anime game poster", "clean lineart", "controlled cel shading", "graphic composition"],
    }[style_mode]
    mode_style = list(mode_style)
    if any(token in " ".join(background).lower() for token in ("seashore", "ocean", "city", "warehouse", "forest")):
        mode_style += ["balanced character and environment composition", "atmospheric perspective", "detailed coherent scenery", "complete continuous environment surrounding the character"]
    return preserve_explicit_intent(NewbieSemanticSpec(
        source_sentence=sentence, character_name=char["name"], character_tag=char["tag"], source=char["source"],
        appearance=appearance, clothing=list(char["clothing"]), expression=expression,
        body_action=action, left_hand=left_hand, right_hand=right_hand, objects=objects, relations=relations,
        framing=framing, background=background, lighting=lighting, style_mode=style_mode,
        style=mode_style, exclusions=[],
    ))


def _clean(items: list[str], default: str = "") -> str:
    values = [re.sub(r"\s+", " ", item.strip()) for item in items if item.strip()]
    return ", ".join(values) or default


def compile_xml(spec: NewbieSemanticSpec) -> str:
    """Historical matrix formatter. New creation uses response_for's native channel."""
    root = Element("img")
    character = SubElement(root, "character_1")
    fields = {
        "n": spec.character_tag,
        "source": spec.source,
        "gender": spec.count,
        "appearance": _clean(spec.appearance),
        "clothing": _clean(spec.clothing),
        "expression": _clean(spec.expression, "natural expression"),
        "action": _clean(spec.body_action, "standing"),
        "left_hand": _clean(spec.left_hand),
        "right_hand": _clean(spec.right_hand),
        "position": _clean(spec.framing, "centered composition"),
    }
    for name, value in fields.items():
        SubElement(character, name).text = value
    general = SubElement(root, "general_tags")
    SubElement(general, "count").text = spec.count
    SubElement(general, "style").text = _clean(spec.style, _clean(BASE_STYLE))
    SubElement(general, "background").text = _clean(spec.background, "simple coherent background")
    SubElement(general, "atmosphere").text = _clean(spec.lighting, "soft lighting")
    exact_text_requested = bool(re.search(r"(?:写着|显示文字|文字为|text reads)[^“\"]*[“\"][^”\"]+[”\"]", spec.source_sentence, re.I))
    canonical_marking = any(token in " ".join(spec.clothing).lower() for token in ("lettering", "written", "logo text", "inscription"))
    if exact_text_requested:
        text_policy = "clean unobstructed surface reserved for later typography, no pseudo-text"
    elif canonical_marking:
        text_policy = "preserve the requested canonical costume marking, suppress unrelated pseudo-text"
    else:
        text_policy = "no_text"
    SubElement(general, "quality").text = f"very_aesthetic, masterpiece, coherent shapes, {text_policy}"
    SubElement(general, "resolution").text = "max_high_resolution"
    SubElement(general, "artist").text = ""
    object_parts = []
    for obj in spec.objects:
        object_parts.append(f"{obj.id}: {obj.name}; appearance: {_clean(obj.appearance)}; position: {_clean(obj.position)}")
    SubElement(general, "objects").text = " | ".join(object_parts)
    relation_parts = [f"{rel.subject} {rel.predicate} {rel.object}" for rel in spec.relations]
    SubElement(general, "other").text = "; ".join(relation_parts)
    return tostring(root, encoding="unicode", short_empty_elements=False)


def build_negative(spec: NewbieSemanticSpec) -> str:
    dynamic = ["floating fragments", "black shards", "collage", "comic panel", "diorama", "unrecognizable object", "masculine face", "square jaw", "long chin", "prominent nose", "asymmetrical eyes", "distorted facial features"]
    if spec.style_mode != NewbieStyleMode.CHIBI:
        dynamic += ["chibi", "super deformed", "sticker", "mascot", "icon"]
    if spec.left_hand or spec.right_hand:
        dynamic += ["hidden hands", "cropped hands", "extra fingers", "fused fingers", "broken wrists", "object fused with hand"]
    if spec.objects:
        dynamic += ["abstract orb", "flower-shaped object", "rose-shaped object", "melted prop", "ambiguous prop"]
    background_text = " ".join(spec.background).lower()
    if any(token in background_text for token in ("seashore", "ocean", "city", "warehouse", "room interior", "forest")):
        dynamic += ["simple background", "plain background", "white background", "studio background", "close-up portrait", "empty background"]
    dynamic += spec.exclusions
    unique = list(dict.fromkeys(dynamic))
    addition = ", ".join(unique)
    base = DEFAULT_NEGATIVE
    canonical_marking = any(token in " ".join(spec.clothing).lower() for token in ("lettering", "written", "logo text", "inscription"))
    if canonical_marking:
        for token in ("english text, ", "chinese text, ", "korean text, ", "logo, ", "character name, ", "copyright name, "):
            base = base.replace(token, "")
    return base.replace("<danbooru_tags>", f"<danbooru_tags>{addition}, ", 1)


def compile_anima_prompt(spec: NewbieSemanticSpec) -> str:
    canonical_marking = any(token in " ".join(spec.clothing).lower() for token in ("lettering", "written", "logo text", "inscription"))
    text_policy = "preserve requested costume marking, no unrelated pseudo-text" if canonical_marking else "no readable text"
    parts = [
        "masterpiece", "best quality", "score_7", "safe", "newest", spec.count,
        spec.character_tag, spec.source, _clean(spec.appearance), _clean(spec.clothing),
        _clean(spec.expression), _clean(spec.body_action), _clean(spec.left_hand), _clean(spec.right_hand),
        _clean(spec.framing), _clean(spec.background), _clean(spec.lighting),
        "canonical character design", "character remains the clear focal point",
        "balanced character and environment", "coherent spatial depth", "continuous full-bleed environment", "natural anatomy",
        "no cutout frame", "no circular vignette", "no collage", text_policy,
    ]
    return ", ".join(dict.fromkeys(item.strip() for item in parts if item and item.strip()))


def lint(spec: NewbieSemanticSpec) -> list[str]:
    warnings: list[str] = []
    object_ids = {obj.id for obj in spec.objects}
    if len(spec.objects) > 2:
        warnings.append("画面包含多个独立道具，请检查各自外观与位置；尚无充分实验能给出统一的道具数量上限。")
    for obj in spec.objects:
        if len(obj.appearance) < 2:
            warnings.append(f"道具 {obj.id} 的外观约束不足，至少补充形状、颜色或材质中的两项。")
    for rel in spec.relations:
        mentioned = f"{rel.subject} {rel.object}".lower()
        if object_ids and not any(obj_id.lower() in mentioned for obj_id in object_ids):
            warnings.append(f"关系“{rel.subject} {rel.predicate} {rel.object}”未明确引用对象 ID。")
        if any(token in f"{rel.subject} {rel.predicate}".lower() for token in ("hand", "finger", "thumb", "手", "指")):
            warnings.append("包含手指与道具的精确接触关系：XML 能明确意图，但 NewBie 单次采样仍可能接触错位；请检查成图，必要时进入局部重绘。")
    framing = " ".join(spec.framing).lower()
    if "full body" in framing and (spec.left_hand or spec.right_hand):
        warnings.append("全身构图中手部像素较少；若手/道具是重点，建议改为半身或腰部以上。")
    all_text = json.dumps(spec.model_dump(), ensure_ascii=False).lower()
    source = spec.source_sentence.lower()
    if spec.character_tag == "original_character":
        warnings.append("未识别到明确角色身份；将按原创角色处理，不会自动猜测已有角色。")
    if spec.background == ["simple coherent background"]:
        warnings.append("没有明确场景；当前使用中性简单背景。")
    if not _has_any(source, POSE_TOKENS):
        warnings.append("没有明确姿态；系统不会猜测坐、站或奔跑，请在生成前确认中性姿态是否合适。")
    if not _has_any(source, SCENE_TOKENS):
        warnings.append("没有识别到明确地点；模型补出的场景可能不是原意，请检查 background。")
    contradictions = (
        (("正午", "noon"), ("黄昏", "暮色", "dusk", "twilight"), "时间同时包含正午与黄昏"),
        (("站", "奔跑", "running", "standing"), ("坐", "sitting"), "姿态同时要求站立/奔跑与坐下"),
        (("大笑", "laughing"), ("面无表情", "expressionless"), "表情同时要求大笑与面无表情"),
        (("正面", "front view"), ("背对", "back view"), "视角同时要求正面与背面"),
        (("万里无云", "clear sky"), ("暴风", "storm cloud"), "天空同时要求无云与风暴云"),
        (("特写", "close-up"), ("全身", "full body"), "构图同时要求特写与完整全身"),
    )
    for left, right, message in contradictions:
        if any(token in source for token in left) and any(token in source for token in right):
            warnings.append(f"矛盾：{message}；请在生成前选择其一。")
    visible_objects = [group for group in OBJECT_GROUPS if _has_any(source, group)]
    if len(visible_objects) > 2:
        warnings.append(f"识别到至少 {len(visible_objects)} 类物体；数量本身不是删减用户要求的依据，请核对是否遗漏。")
    clause_count = len([item for item in re.split(r"[，,；;。]", spec.source_sentence) if item.strip()])
    if clause_count > 9 or len(spec.source_sentence) > 220:
        warnings.append(f"描述包含约 {clause_count} 个分句，需要检查细节保留情况；尚未测得当前模型的可靠文本长度边界。")
    if _has_any(source, ABSTRACT_TOKENS) and not (_has_any(source, POSE_TOKENS) or _has_any(source, SCENE_TOKENS)):
        warnings.append("描述主要依赖比喻且缺少明确姿态/地点；系统只转换可见线索，不保证还原诗意。")
    if "gloves" in all_text and "palm visible" in all_text:
        warnings.append("服装要求手套但动作要求掌心可见，需明确是手套掌面而非裸露掌心。")
    if not spec.objects and any(word in spec.source_sentence.lower() for word in ("拿", "握", "起爆器", "炸弹", "holding")):
        warnings.append("原句似乎包含持物动作，但语义规格没有独立道具；请补充 objects。")
    return warnings


def split_clauses(sentence: str) -> list[str]:
    return [item.strip() for item in re.split(r"[，,；;。！？!?\n]+", sentence) if item.strip()]


def _coverage_availability(spec: NewbieSemanticSpec) -> dict:
    """Structural field presence only; this cannot establish semantic fidelity."""
    return {
        "identity": spec.character_tag != "original_character",
        "action": bool(spec.body_action) and spec.body_action not in (["standing"], ["unspecified neutral pose"]),
        "hands": bool(spec.left_hand or spec.right_hand),
        "objects": bool(spec.objects),
        "background": bool(spec.background) and spec.background != ["simple coherent background"],
        "lighting": bool(spec.lighting) and spec.lighting != ["soft directional lighting"],
        "expression": bool(spec.expression) and spec.expression != ["natural expression"],
        "framing": bool(spec.framing) and spec.framing != ["medium shot", "front view", "centered composition"],
    }


# 语义模型自报字段 → 覆盖检查字段。identity 不在内：角色身份由 CHARACTERS 表确定性判定，模型说了不算
_CLAUSE_FIELD_MAP = {
    "expression": "expression", "body_action": "action",
    "left_hand": "hands", "right_hand": "hands",
    "objects": "objects", "relations": "objects",
    "framing": "framing", "background": "background", "lighting": "lighting",
}


def _contradictory_tokens(source: str) -> set[str]:
    contradiction_pairs = (
        (("正午", "noon"), ("黄昏", "暮色", "dusk", "twilight")),
        (("站", "奔跑", "running", "standing"), ("坐", "sitting")),
        (("大笑", "laughing"), ("面无表情", "expressionless")),
        (("正面", "front view"), ("背对", "back view")),
        (("万里无云", "clear sky"), ("暴风", "storm cloud")),
        (("特写", "close-up"), ("全身", "full body")),
    )
    return {
        token for left, right in contradiction_pairs
        if _has_any(source, left) and _has_any(source, right)
        for token in (*left, *right)
    }


def _missing_named_objects(spec: NewbieSemanticSpec, clause: str) -> list[str]:
    # A prop can legitimately be described in the background. Do not count
    # source_sentence itself, or unrelated fields, as evidence it was rendered.
    represented = " ".join([*spec.background, *[
        " ".join([obj.name, *obj.appearance, *obj.position]) for obj in spec.objects
    ]])
    return [group[0] for group in OBJECT_GROUPS
            if _has_any(clause, group) and not _has_any(represented, group)]


def analyze_coverage(spec: NewbieSemanticSpec, clause_map: list[dict] | None = None) -> dict:
    """Report structural coverage, never present model self-report as proof.

    Named-object checks are a bounded diagnostic, not a semantic judge. A
    nonempty field or a successful JSON parse cannot prove faithful translation.
    """
    clauses = split_clauses(spec.source_sentence)
    clause_lower = spec.source_sentence.lower()
    contradictory_tokens = _contradictory_tokens(clause_lower)
    available = _coverage_availability(spec)
    rows: list[dict] = []
    score = 0.0

    if clause_map is not None:
        mode = "llm_reported"
        by_text = {}
        for item in clause_map:
            if isinstance(item, dict):
                by_text[re.sub(r"\s+", "", str(item.get("clause", "")))] = item
        for index, original in enumerate(clauses):
            clause = original.lower()
            entry = by_text.get(re.sub(r"\s+", "", original))
            if any(token in clause for token in contradictory_tokens):
                rows.append({"text": original, "status": "contradictory", "fields": [], "reason": "与描述中的另一项要求冲突"})
                continue
            if not isinstance(entry, dict):
                rows.append({"text": original, "status": "unmapped", "fields": [], "reason": "编译器未交代此分句，请人工确认"})
                continue
            raw_fields = entry.get("fields", [])
            if not isinstance(raw_fields, list):
                raw_fields = []
            claimed = {_CLAUSE_FIELD_MAP.get(str(field)) for field in raw_fields}
            claimed.discard(None)
            if any(alias.lower() in clause for char in CHARACTERS.values() for alias in char["aliases"]):
                claimed.add("identity")
            claimed = sorted(claimed)
            verified = [field for field in claimed if available[field]]
            empty = [field for field in claimed if not available[field]]
            status = str(entry.get("status", "unmapped"))
            note = str(entry.get("note", "")).strip()
            uncovered = _missing_named_objects(spec, clause)
            if uncovered:
                score += 0.5 if verified else 0
                rows.append({"text": original, "status": "partial" if verified else "unmapped", "fields": verified,
                             "missing_objects": uncovered, "issue_kind": "compiler_omission",
                             "reason": f"原文已指定，但物体未写入规格：{'、'.join(uncovered)}；不是用户未提供信息"})
            elif status not in {"mapped", "partial"} or not claimed:
                rows.append({"text": original, "status": "unmapped", "fields": [],
                             "reason": "编译器未提供可核对的字段映射" + (f"（自报：{note}）" if note else "")})
            elif empty and not verified:
                rows.append({"text": original, "status": "unmapped", "fields": [],
                             "reason": f"编译器声称写入 {', '.join(empty)}，但最终规格对应字段为空" + (f"（{note}）" if note else "")})
            elif empty:
                score += 0.5
                rows.append({"text": original, "status": "partial", "fields": verified,
                             "reason": f"已写入 {', '.join(verified)}；声称的 {', '.join(empty)} 在最终规格中为空"})
            else:
                score += 0.5
                rows.append({"text": original, "status": "partial", "fields": verified,
                             "reason": f"仅确认字段存在：{', '.join(verified)}；语义是否一致尚未独立验证" + (f"（模型自报：{note}）" if note else "")})
        total = len(rows)
        return {
            "total": total,
            "grounded": sum(row["status"] == "mapped" for row in rows),
            "risky": sum(row["status"] != "mapped" for row in rows),
            "ratio": round(score / total, 2) if total else 0,
            "mode": mode,
            "semantic_verified": False,
            "notice": "这是编译器自报与有限结构检查，不是语义准确率或出图质量评分。",
            "clauses": rows,
        }

    mode = "tokens_fallback"
    for original in clauses:
        clause = original.lower()
        expected: list[str] = []
        if any(alias.lower() in clause for char in CHARACTERS.values() for alias in char["aliases"]): expected.append("identity")
        if _has_any(clause, POSE_TOKENS) or _has_any(clause, ("回头", "挥手", "拿", "握", "按", "雕塑", "一动不动", "holding", "looking back", "statue", "motionless")): expected.append("action")
        if _has_any(clause, ("手", "手指", "掌心", "hand", "finger", "thumb")): expected.append("hands")
        if _has_any(clause, OBJECT_TOKENS): expected.append("objects")
        if _has_any(clause, SCENE_TOKENS) or _has_any(clause, ("云", "风", "风暴", "暴风雨", "海风", "cloud", "wind", "storm", "breeze")): expected.append("background")
        if _has_any(clause, ("黄昏", "正午", "日光", "阳光", "暮色", "阴影", "dusk", "noon", "sunlight", "shadow")): expected.append("lighting")
        if _has_any(clause, ("平静", "放空", "发呆", "出神", "微笑", "大笑", "面无表情", "calm", "smile", "expressionless", "gaze")): expected.append("expression")
        if _has_any(clause, ("特写", "全身", "半身", "正面", "背对", "俯视", "仰视", "远景", "全景", "侧身", "侧对", "环境为主", "以景", "close-up", "full body", "medium shot", "front view", "back view", "wide shot", "environmental", "side profile")): expected.append("framing")
        expected = list(dict.fromkeys(expected))
        mapped = [field for field in expected if available[field]]
        missing = [field for field in expected if not available[field]]
        # 字段非空不代表子句里的每个物体都被写入：逐个核对提到的物体组是否真的落进了 objects
        uncovered_objects = _missing_named_objects(spec, clause)
        if any(token in clause for token in contradictory_tokens):
            status, reason = "contradictory", "与描述中的另一项要求冲突"
        elif _has_any(clause, ABSTRACT_TOKENS) and not (_has_any(clause, POSE_TOKENS) or _has_any(clause, SCENE_TOKENS)):
            if expected and mapped and not missing:
                status, reason = "partial", "已转换为可见线索，但属于模型解释，请确认"
                score += 0.5
            else:
                status, reason = "ambiguous", "抽象/比喻含义缺少明确姿态或地点，请确认视觉解释"
        elif not expected:
            status, reason = "unmapped", "未识别到可落入 XML 的可见字段"
        elif missing and mapped:
            status, reason = "partial", f"未写入：{', '.join(missing)}"
            score += 0.5
        elif missing:
            status, reason = "unmapped", f"未写入：{', '.join(missing)}"
        elif uncovered_objects:
            status, reason = "partial", f"部分物体未写入 objects：{'、'.join(uncovered_objects)}"
            score += 0.5
        else:
            status, reason = "mapped", "已写入 " + ", ".join(mapped)
            score += 1
        row = {"text": original, "status": status, "fields": mapped, "reason": reason}
        if uncovered_objects:
            row.update(missing_objects=uncovered_objects, issue_kind="compiler_omission")
        rows.append(row)
    total = len(rows)
    return {
        "total": total,
        "grounded": sum(row["status"] == "mapped" for row in rows),
        "risky": sum(row["status"] != "mapped" for row in rows),
        "ratio": round(score / total, 2) if total else 0,
        "mode": mode,
        "semantic_verified": False,
        "notice": "这是有限关键词与字段检查，不是语义准确率或出图质量评分。",
        "clauses": rows,
    }


@dataclass(slots=True)
class CompileResult:
    spec: NewbieSemanticSpec
    compiler: str
    warnings: list[str]
    # 语义模型逐分句自报的落地去向；规则编译器兜底时为 None（覆盖检查降级到词表路径）
    clause_map: list[dict] | None = None


def compile_sentence(sentence: str, style_mode: NewbieStyleMode, settings: Settings) -> CompileResult:
    from .creation_policy import normalize_creation_identity
    fallback = normalize_creation_identity(fallback_semantic(sentence, style_mode), fallback=True)
    model_path = Path(settings.semantic_model_path)
    python = Path(settings.semantic_python)
    script = Path(__file__).resolve().parents[1] / "scripts" / "semantic_infer.py"
    if not model_path.exists() or not python.exists():
        return CompileResult(fallback, "deterministic_fallback", lint(fallback) + ["本地语义模型尚未安装；当前使用规则编译器，已知角色与常见动作可用。"])
    payload = {"sentence": sentence, "style_mode": style_mode.value,
               "fallback": fallback.model_dump(mode="json"), "clauses": split_clauses(sentence)}
    try:
        with SEMANTIC_INFERENCE_LOCK:
            completed = subprocess.run(
                [str(python), str(script), str(model_path)], input=json.dumps(payload, ensure_ascii=False),
                text=True, encoding="utf-8", capture_output=True, timeout=360, check=True,
            )
        raw = completed.stdout.strip().splitlines()[-1]
        envelope = json.loads(raw)
        parsed = NewbieSemanticSpec.model_validate(envelope["spec"])
        # 模型逐分句自报落地去向；协议缺失（旧脚本/空表）按未交代处理，覆盖检查会如实标出
        clause_map = envelope.get("clause_map") or None
        parsed.source_sentence = sentence
        parsed.style_mode = style_mode
        # Do not let poetic mood or missing information become an invented pose.
        if not _has_any(sentence.lower(), POSE_TOKENS):
            parsed.body_action = fallback.body_action if fallback.body_action != ["standing"] else ["unspecified neutral pose"]
        # A small language model tends to fill visual blanks with stock hand
        # gestures. That silently changes the request and often creates exactly
        # the anatomy/contact failures this pipeline is meant to avoid.
        if not _has_any(sentence.lower(), HAND_ACTION_TOKENS):
            parsed.left_hand = []
            parsed.right_hand = []
        # Relations are high-impact binding constraints, not a place to store
        # poetic paraphrases. Keep only concrete relations whose target exists
        # as a declared object; deterministic known-action relations are the
        # trusted fallback. This prevents `clouds like a storm prelude` and
        # `wind carrying lightness` from leaking into NewBie's <other> slot.
        declared = {
            value.lower().replace(" ", "_") for item in parsed.objects
            for value in (item.id, item.name) if value.strip()
        }
        concrete_predicates = ("hold", "grip", "press", "touch", "carry", "support", "sit on", "stand on", "beside", "behind", "in front", "look", "gaze", "stare")
        parsed.relations = [rel for rel in parsed.relations
                            if any(rel.object.lower().replace(" ", "_").startswith(value) for value in declared)
                            and any(token in rel.predicate.lower() for token in concrete_predicates)]
        if fallback.relations and not parsed.relations:
            parsed.relations = list(fallback.relations)
        # Explicit time, weather, staging and visible emotion from deterministic
        # rules must not be erased by a small model falling back to generic light
        # or portrait phrasing. Unknown scenes can still come entirely from Qwen.
        if fallback.background != ["simple coherent background"]:
            parsed.background = list(dict.fromkeys(fallback.background + [item for item in parsed.background if "simple coherent" not in item.lower()]))
        if fallback.lighting != ["soft directional lighting"]:
            parsed.lighting = list(dict.fromkeys(fallback.lighting + [item for item in parsed.lighting if "soft directional" not in item.lower()]))
        if fallback.framing != ["medium shot", "front view", "centered composition"]:
            overwide = ("wide environmental framing", "extreme wide", "very wide", "distant character", "small character")
            parsed.framing = list(dict.fromkeys(fallback.framing + [item for item in parsed.framing if not any(token in item.lower() for token in overwide)]))
        if fallback.expression != CHARACTERS.get(next((key for key, value in CHARACTERS.items() if value["tag"] == fallback.character_tag), ""), {}).get("expression", ["natural expression"]):
            parsed.expression = list(dict.fromkeys(fallback.expression + parsed.expression))
        if any(token in sentence.lower() for token in ("雕塑", "statue", "一动不动", "motionless")):
            parsed.body_action = list(dict.fromkeys(fallback.body_action + parsed.body_action))
        parsed.expression = _dedupe_limit(parsed.expression, 6)
        parsed.body_action = _dedupe_limit(parsed.body_action, 6)
        parsed.left_hand = _dedupe_limit(parsed.left_hand, 4)
        parsed.right_hand = _dedupe_limit(parsed.right_hand, 4)
        parsed.background = _dedupe_limit(parsed.background, 12)
        parsed.lighting = _dedupe_limit(parsed.lighting, 8)
        parsed.framing = _dedupe_limit(parsed.framing, 7)
        parsed = normalize_creation_identity(preserve_explicit_intent(parsed))
        compiler_name = "local_" + re.sub(r"[^a-z0-9]+", "_", model_path.name.lower()).strip("_")
        return CompileResult(parsed, compiler_name, lint(parsed) + list(envelope.get('patch_warnings', [])), clause_map=clause_map)
    except Exception as exc:
        warning = f"本地语义模型解析失败，已安全回退到规则编译器：{type(exc).__name__}"
        return CompileResult(fallback, "deterministic_fallback", lint(fallback) + [warning])


def response_for(result: CompileResult, *, contract: SceneContractV3 | None = None) -> dict:
    """Compile the current contract, or retain the legacy builder by default.

    The authoring path can supply its already-built, source-attributed contract.
    Rebuilding it here would lose strict entity references and attribution before
    compiling the native negatives. Revalidate a supplied contract without
    rewriting its references; a matching original does not certify semantics.
    """
    from .contract_builder import contract_from_semantic
    from .contract_compilers import PromptTarget, compile_contract
    from .model_conditioning import NEGATIVE_BASELINES, merge_negative
    from .scene_contract import SceneContractV3

    if contract is None:
        contract = contract_from_semantic(result.spec, compiler=result.compiler, clause_map=result.clause_map)
    else:
        if not isinstance(contract, SceneContractV3):
            raise TypeError('current contract must be a SceneContractV3')
        contract = SceneContractV3.model_validate(contract.model_dump(mode='python'))
        if contract.source_text != result.spec.source_sentence:
            raise ValueError('当前契约与语义规格原句不一致')
    native_prompts = {}
    negative_policy = {}
    for target in PromptTarget:
        compiled = compile_contract(contract, target)
        if target == PromptTarget.NEWBIE:
            # Keep the existing new-creation negative baseline while adding the
            # contract's text/forbidden constraints. Both public aliases below
            # now point to this exact prompt; never replace it with compile_xml.
            from .creation_policy import new_creation_negative
            baseline = build_negative(result.spec.model_copy(update={'exclusions': []}))
            compiled.negative, negative_policy = new_creation_negative(result.spec, baseline, compiled.negative,
                compiled.positive, contract_negative_terms=compiled.negative_terms)
            native_negative_policy = negative_policy
        else:
            # These channels are new compiler defaults, not a user editing a
            # native negative to ''. Resolve their own quality baseline here,
            # before routing/preview treat an explicit channel as authoritative.
            # Subsequent manual empty overrides still bypass this compiler.
            baseline_key = 'z_image_turbo' if target == PromptTarget.Z_IMAGE else target.value
            compiled.negative = merge_negative(baseline_key, NEGATIVE_BASELINES[baseline_key], compiled.negative)
            native_negative_policy = {
                # Keep this compile-time attribution distinct from the
                # executor's internal default marker: a later native edit to
                # '' must remain intentional even when old policy is retained.
                'scope': 'semantic_compile_model_defaults',
                'phase': 'semantic_compile_only',
                'baseline_model': baseline_key,
                'semantic_exclusions_preserved': list(dict.fromkeys(result.spec.exclusions)),
                'exclusion_authorship': 'not_inferred_from_field',
                'semantic_correctness_verified': False,
            }
        compiled.require_complete(contract)
        native_prompts[target.value] = {
            "positive": compiled.positive,
            "negative": compiled.negative,
            "trace": compiled.trace,
            "postprocess": compiled.postprocess,
            "trace_status": "compiled_payload_presence_only",
            "negative_policy": native_negative_policy,
        }
    coverage = analyze_coverage(result.spec, result.clause_map)
    omission_warnings = [f"解析遗漏：{row['reason']}" for row in coverage["clauses"]
                         if row.get("issue_kind") == "compiler_omission"]
    from .professional_review import build_professional_review
    return {
        "spec": result.spec.model_dump(mode="json"),
        "contract_v3": contract.model_dump(mode="json"),
        "native_prompts": native_prompts,
        "xml": native_prompts['newbie']['positive'],
        "negative_prompt": native_prompts['newbie']['negative'],
        "anima_prompt": native_prompts['anima']['positive'],
        "warnings": list(dict.fromkeys([*result.warnings, *omission_warnings])),
        "compiler": result.compiler,
        "parsed_character_tag": result.spec.character_tag,
        "coverage": coverage,
        "negative_policy": negative_policy,
        "professional_review": build_professional_review(result),
    }


def recompile_response(request) -> dict:
    """Rebuild edited semantics without silently losing conflict/selection checks.

    No language-model inference and no reapplication of knowledge over edits.
    Original intent is retained as context, not promoted to verified semantics.
    """
    result = CompileResult(request.spec, 'edited_semantic_spec', lint(request.spec))
    response = response_for(result)
    intent = request.intent
    # Preserve the original extractor's interpretation across repeated rebuilds.
    # It is conflict-check context, not independently verified identity evidence.
    if request.parsed_character_tag:
        response['parsed_character_tag'] = request.parsed_character_tag
    if intent and intent.knowledge_pack:
        from .knowledge import apply_knowledge
        from .professional_review import build_professional_review
        original_spec = fallback_semantic(intent.sentence, intent.style_mode)
        response['parsed_character_tag'] = request.parsed_character_tag or original_spec.character_tag
        selected = apply_knowledge(CompileResult(original_spec, 'selection_check', []),
            intent.knowledge_pack, intent.knowledge_operator, intent.knowledge_appearance)
        selection_review = build_professional_review(selected,
            parsed_character_tag=request.parsed_character_tag or original_spec.character_tag,
            knowledge_operator=intent.knowledge_operator)
        review = response['professional_review']
        review['questions'].extend(item for item in selection_review['questions']
                                   if item['id'] == 'knowledge_identity_conflict')
        if request.spec.character_tag != selected.spec.character_tag:
            review['questions'].append({
                'id': 'edited_knowledge_identity_conflict', 'kind': 'conflict',
                'question': '编辑后的角色与当前知识库选择不一致，应以哪一个为准？',
                'reason': f'规格为 {request.spec.character_tag}，所选资源为 {selected.spec.character_tag}。',
                'action': '请更改角色选择后重新解析，或修正规格中的角色身份。',
            })
        review['blocking'] = bool(review['questions'])
        response['warnings'].append('已按当前资源选择重新检查角色冲突；手工修改的外观仍是待验证内容，未重新覆盖为知识库默认值。')
    return response
