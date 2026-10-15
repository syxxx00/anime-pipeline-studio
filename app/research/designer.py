from __future__ import annotations

from copy import deepcopy
from xml.sax.saxutils import escape

from ..prompting import DEFAULT_NEGATIVE
from ..schemas import ExperimentCreateRequest, ExperimentDesignRequest


DEFAULT_CHARACTERS = [
    {
        "id": "w_arknights", "name": "W", "source": "Arknights", "name_tag": "w_(arknights)",
        "appearance": "shoulder-length silver-gray hair with darker underlayers, amber-orange eyes, two long black horns with vivid red edges",
        "clothing": "black cropped tactical jacket with vivid red lining and red sleeve stripes, fitted light-gray high-neck shirt, thick black scarf, black shorts with layered red fabric panels, patterned black tights, fingerless tactical gloves",
        "signature_action": "creating a controlled red explosion with an explosive device", "confirmed": False,
    },
    {
        "id": "specter_arknights", "name": "幽灵鲨 / Specter", "source": "Arknights", "name_tag": "specter_(arknights)",
        "appearance": "long silver-white hair with short jagged bangs, vivid red eyes, pale skin, calm unsettling expression",
        "clothing": "oversized black nun veil with a wide off-white inner border, dark gray cropped jacket with black buckle straps, white high-neck underlayer, layered black white and deep-blue skirt, long gold necklace",
        "signature_action": "holding her large circular saw weapon in a controlled combat stance", "confirmed": False,
    },
    {
        "id": "amiya_arknights", "name": "阿米娅 / Amiya", "source": "Arknights", "name_tag": "amiya_(arknights)",
        "appearance": "very long straight brown hair, large upright dark-brown rabbit ears, bright teal-blue eyes, youthful face",
        "clothing": "oversized black hooded coat with teal-blue lining, white ribbed blouse, short blue plaid pleated skirt, opaque black tights, wide teal shoulder sash carrying white Rhodes Island lettering",
        "signature_action": "casting restrained dark Originium Arts energy from one hand", "confirmed": False,
    },
    {
        "id": "kangel", "name": "超天酱 / OMGkawaiiAngel", "source": "Needy Streamer Overload", "name_tag": "omgkawaiiangel_(needy_girl_overdose)",
        "appearance": "long pale hair in twin tails with pastel pink and blue gradients, bright blue eyes, angelic streamer persona",
        "clothing": "canonical pastel blue and pink idol outfit, ribbon accessories, heart and cross motifs",
        "signature_action": "making her signature cheerful streamer pose with sparkling pastel effects", "confirmed": False,
    },
    {
        "id": "misaka_mikoto", "name": "御坂美琴 / Misaka Mikoto", "source": "A Certain Scientific Railgun", "name_tag": "misaka_mikoto",
        "appearance": "short chestnut-brown hair, brown eyes, small hair clip, confident expression",
        "clothing": "canonical Tokiwadai Middle School uniform, beige sweater vest, white shirt, gray pleated skirt, loose socks",
        "signature_action": "firing a railgun coin with blue-white electricity around her arm", "confirmed": False,
    },
]


DEFAULT_SCENES = [
    {"id": "identity_portrait", "name": "01 身份近景", "text": "Solo chest-up portrait, front view, looking at the viewer, neutral simple pose, both shoulders visible, plain light-gray background, even studio lighting, no objects and no text.", "confirmed": False},
    {"id": "full_body", "name": "02 全身服装", "text": "Solo full-body standing pose, entire body and footwear inside frame, arms relaxed beside the body, front three-quarter view, plain light background, even lighting, no text.", "confirmed": False},
    {"id": "hands", "name": "03 双手压力", "text": "Solo waist-up front view, both complete bare or gloved hands held clearly in front of the shoulders, palms facing the viewer, five separated fingers on each hand, anatomically connected wrists, simple symmetrical pose, plain background, no text.", "confirmed": False},
    {"id": "signature", "name": "04 标志性动作", "text": "Solo medium shot performing {signature_action}; keep the face, hairstyle, signature accessories and canonical clothing unobstructed; both hands visible; restrained effects; simple dark background; no text.", "confirmed": False},
    {"id": "dynamic", "name": "05 动态透视", "text": "Solo dynamic running and turning pose, full body inside frame, moderate foreshortening, one hand reaching toward the camera and the other visible, readable silhouette, simple outdoor ground and sky, daylight, no text.", "confirmed": False},
    {"id": "environment", "name": "06 复杂场景", "text": "Solo character in a detailed nighttime urban environment after rain, wet reflective pavement, layered foreground and background, cinematic rim lighting, both hands visible, character remains the clear focal point, no crowd, signage without readable letters, no text.", "confirmed": False},
]


def defaults() -> dict:
    return {
        "name": "模型对比：NewBie vs Anima Turbo（5角色×6场景）",
        "characters": deepcopy(DEFAULT_CHARACTERS),
        "scenes": deepcopy(DEFAULT_SCENES),
        "seeds": [42, 123456, 987654321],
        "width": 1024,
        "height": 1024,
    }


def _scene_text(character, scene) -> str:
    return scene.text.replace("{signature_action}", character.signature_action)


def same_english(character, scene) -> str:
    return (
        f"A clean high-quality anime game illustration of {character.name} from {character.source}. "
        f"The character has {character.appearance} and wears {character.clothing}. "
        f"{_scene_text(character, scene)} Preserve the exact canonical character identity, proportions, "
        "colors and signature accessories. Crisp line art, restrained cel shading, no watermark, no logo."
    )


def newbie_xml(character, scene) -> str:
    action = _scene_text(character, scene)
    return f"""<img>
<character_1>
<n>{escape(character.name_tag)}</n>
<gender>1girl, solo</gender>
<appearance>{escape(character.appearance)}</appearance>
<clothing>{escape(character.clothing)}</clothing>
<expression>canonical expression, looking at viewer when appropriate</expression>
<action>{escape(action)}</action>
<position>center, complete composition, no crop unless specified</position>
</character_1>
<general_tags>
<count>1girl, solo</count>
<style>anime_style, game_illustration, clean_lineart, restrained_cel_shading</style>
<background>as specified in action, coherent background, no floating fragments</background>
<atmosphere>controlled lighting, character remains focal point</atmosphere>
<quality>very_aesthetic, masterpiece, no_text</quality>
<resolution>max_high_resolution</resolution>
<artist></artist>
<objects>only objects explicitly specified</objects>
<other>canonical character design, accurate signature accessories</other>
</general_tags>
</img>"""


def anima_native(character, scene) -> str:
    return (
        f"masterpiece, best quality, score_7, safe, newest, 1girl, solo, {character.name_tag}, "
        f"{character.source}, {character.appearance}, {character.clothing}, {_scene_text(character, scene)}, "
        "canonical character design, accurate signature accessories, clean lineart, restrained cel shading, no text, no logo, no watermark"
    )


def preview(design: ExperimentDesignRequest) -> dict:
    rows = []
    for character in design.characters:
        for scene in design.scenes:
            rows.append({
                "character_id": character.id,
                "character": character.name,
                "scene_id": scene.id,
                "scene": scene.name,
                "same_english": same_english(character, scene),
                "newbie_native": newbie_xml(character, scene),
                "anima_native": anima_native(character, scene),
            })
    return {
        "rows": rows,
        "total_images": len(design.characters) * len(design.scenes) * len(design.seeds) * 2 * 2,
        "all_confirmed": all(item.confirmed for item in design.characters) and all(item.confirmed for item in design.scenes),
    }


def build_manifest(design: ExperimentDesignRequest) -> ExperimentCreateRequest:
    if not all(item.confirmed for item in design.characters):
        raise ValueError("仍有角色身份文本未确认")
    if not all(item.confirmed for item in design.scenes):
        raise ValueError("仍有场景文本未确认")
    items = []
    for character in design.characters:
        for scene in design.scenes:
            prompts = {
                "same_english": {"newbie": same_english(character, scene), "anima_turbo": same_english(character, scene)},
                "native_best": {"newbie": newbie_xml(character, scene), "anima_turbo": anima_native(character, scene)},
            }
            for seed in design.seeds:
                for prompt_form, model_prompts in prompts.items():
                    for model_id, prompt in model_prompts.items():
                        turbo = model_id == "anima_turbo"
                        items.append({
                            "axes": {
                                "model": model_id, "character": character.name, "character_id": character.id,
                                "scene": scene.name, "scene_id": scene.id, "prompt_form": prompt_form, "seed": seed,
                            },
                            "request": {
                                "model_id": model_id, "mode": "character", "prompt": prompt,
                                "negative_prompt": "" if turbo else DEFAULT_NEGATIVE,
                                "width": design.width, "height": design.height, "seed": seed,
                                "steps": 10 if turbo else 28, "cfg": 1 if turbo else 4.5,
                                "sampler": "euler" if turbo else "res_multistep",
                                "scheduler": "simple" if turbo else "linear_quadratic",
                                "shift": 3 if turbo else 6, "cfg_rescale": 0.9,
                                "lora_name": None, "lora_strength": 0.75, "batch_size": 1,
                                "output_group": f"experiments/newbie-vs-anima-5x6/{character.id}/{scene.id}",
                                "output_label": f"{model_id}__{character.id}__{scene.id}__{prompt_form}__seed-{seed}",
                            },
                        })
    return ExperimentCreateRequest.model_validate({
        "name": design.name,
        "notes": "由网页实验设计器确定性生成；相同英文文本与各模型原生最佳格式；未调用外部 LLM。",
        "external_llm_used": False,
        "items": items,
    })
