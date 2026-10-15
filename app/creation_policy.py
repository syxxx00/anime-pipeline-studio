"""Bounded safeguards against our own defaults contradicting requested content.

This is not an anatomy judge or a general negation/scene parser. Historical
matrix formatting remains unchanged; new creation can audit automatic removals.
"""
import re
from xml.etree import ElementTree as ET

from .intent_guards import matches
from .schemas import NewbieSemanticSpec


FACE_DEFAULTS = {'delicate feminine face', 'soft tapered jawline', 'balanced feminine facial proportions'}

# These are ordinary framing/visibility choices, not image defects. The legacy
# formatter remains unchanged for historical research, but a new creation must
# not demand all hands in frame or prohibit a portrait merely because its scene
# mentions hands or an environment. Explicit semantic exclusions still follow.
FRAMING_DEFAULTS = {'hidden hands', 'cropped hands', 'close-up portrait'}


def affirmative_match(fields: list[str], tokens: tuple[str, ...]) -> bool:
    for value in fields:
        text = value.lower().replace('_', ' ')
        for token in tokens:
            pattern = re.escape(token)
            if re.search('[a-z]', token):
                pattern = r'(?<![a-z0-9_])' + pattern + r'(?![a-z0-9_])'
            for match in re.finditer(pattern, text):
                prefix = re.split(r'[,，;；。]|\bbut\b', text[:match.start()])[-1]
                if not re.search(r'(?:\b(?:no|not|without|avoid)\b(?:\W+\w+){0,2}\W*|(?:不要|避免|没有|不是|无)[^,，;；。]{0,6})$', prefix):
                    return True
    return False


def explicit_gender(sentence: str) -> str | None:
    # Quoted lettering is not a description of the protagonist. Only bounded,
    # affirmative single-subject cases are handled; mixed subjects stay unknown.
    text = re.sub(r'[“"].*?[”"]', '', sentence.lower())
    found = set()
    for clause in re.split(r'[,，;；。.!?]', text):
        if re.search(r'不是|不要|并非|不画|\b(?:not|without|no)\b', clause):
            continue
        if matches(clause, ('男性角色', '成年男子', '男性版', '男角色', '1boy', 'male character', 'a man', 'a boy')):
            found.add('1boy, solo')
        if matches(clause, ('女性角色', '成年女子', '女性版', '女角色', '1girl', 'female character', 'a woman', 'a girl')):
            found.add('1girl, solo')
    return next(iter(found)) if len(found) == 1 else None


def normalize_creation_identity(spec: NewbieSemanticSpec, *, fallback: bool = False) -> NewbieSemanticSpec:
    result = spec.model_copy(deep=True)
    gender = explicit_gender(spec.source_sentence)
    if gender:
        result.count = gender
    elif fallback and spec.character_tag == 'original_character':
        result.count = '1person, solo'
    if result.count != '1girl, solo':
        result.appearance = [value for value in result.appearance if value not in FACE_DEFAULTS]
    return result


def new_creation_negative(spec: NewbieSemanticSpec, automatic: str, contract_negative: str, positive_prompt: str = '',
                          *, contract_negative_terms: list[str] | None = None) -> tuple[str, dict]:
    """`automatic` must be built with spec.exclusions removed.

    Filter only generated defaults. Semantic/contract exclusions are appended intact.
    A semantic field does not by itself prove that the user authored it.
    Matching is against positive semantic fields, never raw negated user clauses.
    Known compiler entries are checked against their serialized original. Legacy
    strings without entries remain opaque; natural-language commas are not split.
    """
    if contract_negative_terms is None:
        contract_terms = [contract_negative] if contract_negative else []
    else:
        if (not isinstance(contract_negative_terms, list)
                or any(not isinstance(term, str) for term in contract_negative_terms)
                or ', '.join(contract_negative_terms) != contract_negative):
            raise ValueError('编译负面条目与完整负面原文不一致；未猜测拆分或合并')
        contract_terms = contract_negative_terms
    root = ET.fromstring(f'<root>{automatic}</root>')
    tag_node = root.find('danbooru_tags')
    if tag_node is None:
        raise ValueError('自动负面提示词缺少 danbooru_tags 槽')
    fields = [*spec.appearance, *spec.clothing, *spec.expression, *spec.body_action,
              *spec.left_hand, *spec.right_hand, *spec.framing, *spec.background, *spec.style]
    fields.extend(value for obj in spec.objects for value in [obj.name, *obj.appearance, *obj.position])
    if positive_prompt:
        # Text-intent surfaces/UI may be supplied by the contract compiler rather
        # than the legacy spec fields, but are still actual positive conditioning.
        fields.extend(ET.fromstring(positive_prompt).itertext())
    aliases = {
        'collage': ('collage', '拼贴'), 'comic panel': ('comic panel', 'manga panel', '漫画分镜'),
        'diorama': ('diorama', '微缩景观'), 'chibi': ('chibi', 'q版'),
        'super deformed': ('chibi', 'super deformed', 'q版'),
        'sticker': ('sticker', '贴纸'), 'mascot': ('mascot', '吉祥物'), 'icon': ('icon', '图标'),
        'abstract': ('abstract', '抽象'), 'abstract orb': ('orb', 'sphere', '球体', '圆球'),
        'flower-shaped object': ('flower', 'floral', '花朵', '花形'),
        'rose-shaped object': ('rose', '玫瑰'),
        'hidden hands': ('hidden hand', 'hidden hands', 'hands hidden', 'hand hidden', 'behind the back', 'out of frame', 'outside frame', '手藏', '隐藏手', '画外'),
        'cropped hands': ('cropped hand', 'cropped hands', 'out of frame', 'outside the frame', 'outside frame', '画外'),
        'close-up portrait': ('close-up', 'close up', 'portrait', '特写'),
        'simple background': ('simple background', 'simplified background', '简洁背景'),
        'plain background': ('plain background', 'solid background', '纯色背景'),
        'white background': ('white background', '白色背景'),
        'studio background': ('studio background', '摄影棚背景'),
        'empty background': ('empty background', '空白背景'),
        'floating fragments': ('floating fragments', 'shattered glass', 'glass fragments', '漂浮碎片', '玻璃碎片'),
        'black shards': ('black shards', '黑色碎片'),
        'furry': ('furry', '兽人'), 'long chin': ('long chin', '长下巴'),
        'square jaw': ('square jaw', '方下颌'), 'prominent nose': ('prominent nose', '高鼻梁'),
        'missing limbs': ('amputee', 'amputation', '截肢'),
    }
    removed = []
    kept = []
    for token in (part.strip() for part in (tag_node.text or '').split(',')):
        if not token:
            continue
        reason = ''
        if token in FRAMING_DEFAULTS:
            reason = '可见范围与景别由当前构图决定，不作为新创作的自动质量禁令'
        elif token == 'masculine face' and (spec.count != '1girl, solo' or affirmative_match(fields, ('masculine face', 'androgynous face', '中性面容'))):
            reason = '不能将既有女性角色场景的脸型默认用于此主体'
        elif token in aliases and affirmative_match(fields, aliases[token]):
            reason = '与正向语义中已指定的表现冲突'
        if reason:
            removed.append({'token': token, 'reason': reason})
        else:
            kept.append(token)
    # The contract already contains semantic exclusions, but preserve them even if a
    # future caller passes a contract with a narrower projection of the spec.
    retained_user = list(dict.fromkeys(spec.exclusions))
    tag_node.text = ', '.join(dict.fromkeys([*kept, *retained_user, *contract_terms]))
    result = '\n'.join(ET.tostring(child, encoding='unicode', short_empty_elements=False) for child in root)
    return result, {'scope': 'bounded_automatic_default_filter', 'phase': 'semantic_compile_only', 'removed_automatic_terms': removed,
                    'semantic_exclusions_preserved': retained_user,
                    'exclusion_authorship': 'not_inferred_from_field', 'semantic_correctness_verified': False}
