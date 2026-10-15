"""A complete single-frame description, without a second pose/object inventory.

This is an alternative authoring carrier, not an extraction of user intent or
a claim that an unparsed relationship graph contains the described relations.
The existing native compiler still owns identity, full recipes and negatives.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .art_direction import KNOWLEDGE_GUIDANCE_PROMPT
from .picture_event import PictureEvent


_Word = Annotated[str, Field(min_length=1, max_length=400)]
_CJK = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af\U00020000-\U0002fa1f]')


class ProsePictureChoice(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    count: str = Field(min_length=1, max_length=120,
        description='Subjects within this one frame, never candidate count.')
    scene_narrative: str = Field(min_length=80, max_length=3500,
        description='One complete English painting description, not alternatives or a list of prompts.')
    visual_logic: str = Field(min_length=1, max_length=1200)
    exclusions: list[_Word] = Field(max_length=12)
    uncertainties: list[_Word] = Field(max_length=6)


PROSE_PICTURE_SYSTEM = """Author ONE complete drawable frame in English prose.
All the original, current knowledge, frame, full style choices and raw feedback
remain supplied. This is a different description carrier, not a smaller task.
Choose the camera, body and attention direction, visible hands and their support,
object placement, inside/outside boundaries, environment and shared light/color
together. Commit to one physical moment. Each visible hand has one compatible
action; do not offer multiple gestures or cameras for the same moment.
If different candidate directions are requested, choose one now. Other directions
belong to later operations, not additional people or alternatives in this frame.
Deliberately requested composite/multi-panel images remain valid.
Write those choices ONCE in scene_narrative, in a natural integrated description.
Do not produce another tag/pose/object inventory, symbolism essay or schema explanation.
Count names subjects inside this frame; a nonnegative integer string or existing
English generation terms are both valid. A quantity does not specify gender or
species. Do not turn background biography into
current action. A selected name is an identity handle, not an animal/profession.
The host supplies exact identity tags, selected appearance and the entire style
recipe; don't invent canon, costume, source quotations or duplicate artist strings.
Keep the original's attention/action and emotional meaning. Ordinary unspecified
art choices are yours and replaceable; don't require expert input from the user.
Prior plans, fallible pixel observations and system objective/preserve strings
are unverified materials, not user words or immutable requirements. You may drop
an unrequested parent prop when it causes incompatible hands or distracts from
the original. Respect actual user corrections and explicit current conditions.
visual_logic explains this frame, not the schema; it may be Chinese. exclusions
are English unwanted visual features. uncertainties name actual unresolved matters,
not generic model/seed caveats. Empty lists are allowed, not targets to fill.
Do not claim approval or visual accuracy. Return only the complete supplied JSON.
""" + KNOWLEDGE_GUIDANCE_PROMPT


def prose_native_count(count: str) -> str:
    """Adapt quantity syntax without inventing gender, species or identity."""
    if re.fullmatch(r'[0-9]+', count.strip()):
        quantity = int(count.strip())
        return 'solo' if quantity == 1 else f'{quantity} subjects'
    return count


def prose_language_scope(choice: ProsePictureChoice) -> dict:
    """Declare field consumption, not verification of translated meaning."""
    return {'kind': 'host_applies_only_original_cjk_drawing_fields',
        'applied_choice_origin': 'host_copy_of_author_raw_with_selected_language_response_fields',
        'field_sources': {
            'count': 'language_response' if _CJK.search(choice.count) else 'author_original',
            'scene_narrative': 'language_response' if _CJK.search(choice.scene_narrative) else 'author_original',
            'visual_logic': 'author_original', 'uncertainties': 'author_original',
            'exclusions': ['language_response' if _CJK.search(item) else 'author_original'
                for item in choice.exclusions]},
        'exclusions_alignment': 'same_item_count_and_original_index_consumption',
        'semantic_correctness_verified': False}


def apply_prose_language(choice: ProsePictureChoice, language_return: dict) -> ProsePictureChoice:
    """Apply only requested language fields; the whole return stays evidence."""
    raw = choice.model_dump(mode='json')
    scope = prose_language_scope(choice)['field_sources']
    exclusions = language_return.get('exclusions')
    if not isinstance(exclusions, list) or len(exclusions) != len(choice.exclusions):
        raise ValueError('完整描述语言转换排除项数不一致；没有重排、补入或删去原索引。')
    value = deepcopy(raw)
    for field in ('count', 'scene_narrative'):
        if scope[field] == 'language_response':
            value[field] = deepcopy(language_return.get(field))
    for index, source in enumerate(scope['exclusions']):
        if source == 'language_response':
            value['exclusions'][index] = deepcopy(exclusions[index])
    applied = ProsePictureChoice.model_validate(value)
    drawing = [applied.count, applied.scene_narrative, *applied.exclusions]
    if any(_CJK.search(item) for item in drawing):
        raise ValueError('完整描述语言应用后绘图字段仍含未转换的中文／CJK；未派图或重试。')
    return applied


def prose_picture_event(choice: ProsePictureChoice) -> PictureEvent:
    """Use the authored description verbatim, never guess a graph from prose.

Empty structural fields mean not extracted by this carrier, not absent in the
picture. Native caption/prose consumers receive the entire authored narrative.
"""
    raw = ProsePictureChoice.model_validate(choice.model_dump(mode='json'))
    spec = {key: [] for key in ('expression', 'body_action', 'left_hand', 'right_hand',
        'objects', 'relations', 'framing', 'background', 'lighting', 'style')}
    spec.update(count=prose_native_count(raw.count), exclusions=deepcopy(raw.exclusions))
    return PictureEvent(scene_narrative=raw.scene_narrative, visual_logic=raw.visual_logic,
        positions={}, layout=[], uncertainties=deepcopy(raw.uncertainties), visual_spec=spec)


def prose_picture_receipt(choice: ProsePictureChoice, *, original_choice=None) -> dict:
    raw = ProsePictureChoice.model_validate(original_choice or choice.model_dump(mode='json'))
    native_count = prose_native_count(choice.count)
    raw_json = json.dumps(raw.model_dump(mode='json'), ensure_ascii=False,
        sort_keys=True, allow_nan=False, separators=(',', ':'))
    return {'method': 'prose', 'kind': 'host_prose_picture_event_transport',
        'model_authored_fields': list(ProsePictureChoice.model_fields),
        'raw_choice_sha256': hashlib.sha256(raw_json.encode('utf-8')).hexdigest(),
        'scene_narrative_origin': 'model_authored_complete_prose',
        'scene_narrative_consumption': 'verbatim_complete_description',
        'language_conversion_applied': raw.model_dump(mode='json') != choice.model_dump(mode='json'),
        'language_application': prose_language_scope(raw),
        'raw_count': raw.count, 'native_count': native_count,
        'count_format_conversion_applied': choice.count != native_count,
        'count_conversion_scope': 'quantity_format_only_no_gender_or_species_inference',
        'narrative_consumption': 'whole_authored_description_in_native_caption_or_prose',
        'structured_visual_fields': 'not_extracted_not_absence_or_relation_verification',
        'semantic_correctness_verified': False, 'visual_correctness_verified': False}
