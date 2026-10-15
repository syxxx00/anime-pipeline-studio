"""A narrow, provenance-scoped check of two explicit spatial channels.

This is not a natural-language scene validator. It compares a small set of
literal, mutually exclusive axes only when the SOURCE value and its exact
quotation agree on the same frame, reference and axis. Unmentioned dimensions,
unknown prose and anatomical versus image-space directions remain distinct.
No text is rewritten, no evidence becomes verified, and acceptance does not
certify either the interpretation or the eventual image.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from .scene_contract import ConstraintRole, EvidenceKind, SceneContractV3


@dataclass(frozen=True)
class SpatialAxis:
    frame: str
    reference: str
    dimension: str
    value: str

    @property
    def key(self) -> tuple[str, str, str]:
        return self.frame, self.reference, self.dimension


_DEPTH = {'foreground': 'foreground', 'background': 'background',
          'middle ground': 'middle', 'midground': 'middle'}
_OPPOSITES = {frozenset(('left', 'right')), frozenset(('top', 'bottom')),
              frozenset(('foreground', 'background'))}
_NEGATED_OR_OPTIONAL = re.compile(
    r'\b(?:not|no|without|either|or|perhaps|maybe|might|could|possibly)\b|'
    r'不(?:在|是|要|一定)|没有|并非|或|可能|也许|任意|若|如果', re.I)
_UNOWNED_BODY_PARTS = r'hand|knee|shoulder|foot|eye|arm|leg'
_BODY_PARTS = _UNOWNED_BODY_PARTS + r'|side'
_BODY_CN = {'手': 'hand', '膝': 'knee', '肩': 'shoulder', '脚': 'foot',
            '眼': 'eye', '臂': 'arm', '腿': 'leg'}


def _normal(value: str) -> str:
    return ' '.join(value.lower().strip().split())


def _reference(value: str, aliases: dict[str, str]) -> str | None:
    normalized = _normal(value).removeprefix('the ')
    if normalized in {'frame', 'image', 'picture', 'screen', '画面', '镜头'}:
        return 'frame'
    return aliases.get(normalized)


def literal_spatial_axes(text: str, aliases: dict[str, str] | None = None) -> list[SpatialAxis]:
    """Recognize bounded literal clauses, not arbitrary direction-word presence.

    Bare ``left foreground`` and ``在左边`` use the conventional image frame.
    ``left of character`` uses that referenced entity; ``her left knee`` uses
    an anatomical landmark. Alternative/negated clauses and unknown relative
    anchors yield no assertion. This deliberately leaves many valid phrasings
    unparsed rather than inventing a reference system.
    """
    aliases = {_normal(key): value for key, value in (aliases or {}).items()}
    aliases.update({key: 'character' for key in ('character', 'her', 'him', '她', '他', '人物', '角色')})
    result: list[SpatialAxis] = []

    def add(frame: str, reference: str, dimension: str, value: str):
        axis = SpatialAxis(frame, reference, dimension, value)
        if axis not in result:
            result.append(axis)

    for raw in re.split(r'[,，;；。.!！?？\n]', text):
        clause = _normal(raw)
        if not clause or _NEGATED_OR_OPTIONAL.search(clause):
            continue
        # Remove parsed or unrecognized relative/anatomical phrases before
        # looking for conventional frame words. A body-left is never frame-left.
        masked = list(clause)

        def consume(match):
            masked[match.start():match.end()] = ' ' * (match.end() - match.start())

        for match in re.finditer(rf"(?:\b(?:near|by|beside|at|on|in)\s+|^)(?:her|his|their|character(?:'s)?)\s+"
                                 rf'(left|right)\s+({_BODY_PARTS})\b', clause):
            add('body', 'character.' + match[2], 'lateral', match[1])
            consume(match)
        for match in re.finditer(r'(?:在|靠|位于|放在|^)(?:她|他|人物|角色)(?:的)?(左|右)(手|膝|肩|脚|眼|臂|腿)(?:旁|边|侧)?', clause):
            add('body', 'character.' + _BODY_CN[match[2]], 'lateral', 'left' if match[1] == '左' else 'right')
            consume(match)
        for match in re.finditer(rf'\b(?:left|right)\s+(?:{_UNOWNED_BODY_PARTS})\b', clause):
            # An unspecified body's right hand is still not image-right.
            consume(match)
        for match in re.finditer(r'\b(upper|lower|top|bottom)\s+(?:half|part|side)\s+of\s+(.+?)(?=\s+(?:and|with)\b|$)', clause):
            reference = _reference(match[2], aliases)
            if reference == 'frame':
                add('screen', 'frame', 'vertical', 'top' if match[1] in {'upper', 'top'} else 'bottom')
            # Body/unknown-frame halves are not image-space coordinates.
            consume(match)
        for match in re.finditer(r'\b(?:to\s+|on\s+|at\s+)?(?:the\s+)?(left|right)\s+'
                                 r'(?:side\s+)?of\s+(.+?)(?=\s+(?:and|with|near|above|below)\b|$)', clause):
            reference = _reference(match[2], aliases)
            if reference is not None:
                add('screen' if reference == 'frame' else 'relative', reference, 'horizontal', match[1])
            consume(match)
        for match in re.finditer(r'(?:在|位于|放在)?(人物|角色|她|他|画面|镜头)(?:的)?(左|右)(?:边|侧)', clause):
            reference = _reference(match[1], aliases)
            if reference is not None:
                add('screen' if reference == 'frame' else 'relative', reference, 'horizontal',
                    'left' if match[2] == '左' else 'right')
            consume(match)
        for match in re.finditer(r'(?:在|位于|放在)(人物|角色|她|他)(?:的)?(?:上方|下方)', clause):
            # The literal source does not use image-top/image-bottom. Keep
            # relative vertical prose unknown until its anchor is supported.
            consume(match)
        for match in re.finditer(r"\b(?:screen[- ]|viewer(?:'s)?\s+)(left|right)\b", clause):
            add('screen', 'frame', 'horizontal', match[1])
            consume(match)
        rest = ''.join(masked)
        # Bare leading staging labels, or explicit locative constructions.
        # "left bank", "right hand" and "left alone" do not match.
        for match in re.finditer(r'\b(left|right)\s+(?:foreground|background|middle\s+ground|midground|'
                                 r'third|half|edge|corner|side)\b|'
                                 r'\b(?:on|at|in|towards?)\s+(?:the\s+)?(left|right)(?:\s+side)?\b', rest):
            add('screen', 'frame', 'horizontal', match[1] or match[2])
        if rest.strip() in {'left', 'right'}:
            add('screen', 'frame', 'horizontal', rest.strip())
        for match in re.finditer(r'(?:在|位于|放在|靠)(左|右)(?:边|侧)|^(左|右)(?:边|侧)$', rest):
            add('screen', 'frame', 'horizontal', 'left' if (match[1] or match[2]) == '左' else 'right')
        for match in re.finditer(r'(?:\b(?:in|at|on)\s+(?:the\s+)?(?:far\s+)?|'
                                 r'^(?:(?:left|right|far)\s+)?)'
                                 r'(foreground|background|middle\s+ground|midground)'
                                 r'(?=$|\s+(?:near|beside|with|behind|ahead|above|below|on|at|in|left|right|upper|lower|along|towards?)\b)', rest):
            add('screen', 'frame', 'depth', _DEPTH[match[1]])
        for match in re.finditer(r'(?:在|位于|放在)?(?:画面)?(前景|背景|中景)(?=$|[的内里处]|之中|中)', rest):
            add('screen', 'frame', 'depth', {'前景': 'foreground', '背景': 'background', '中景': 'middle'}[match[1]])
        for match in re.finditer(r'\b(?:upper|top|lower|bottom)\s+(?:half|third|edge|corner|part|side)\b|'
                                 r'\b(?:at|on|in)\s+(?:the\s+)?(top|bottom)\b', rest):
            value = match[1] or match[0].split()[0]
            add('screen', 'frame', 'vertical', 'top' if value in {'upper', 'top'} else 'bottom')
        for match in re.finditer(r'(?:画面|镜头)(?:的)?(上方|上部|上边|下方|下部|下边)|'
                                 r'(?:在|位于|放在)(上方|上部|上边|下方|下部|下边)|'
                                 r'^(上方|上部|上边|下方|下部|下边)$', rest):
            add('screen', 'frame', 'vertical', 'top' if '上' in match[0] else 'bottom')
    return result


def _scoped_axis_text(text: str, aliases: dict[str, str], owner: str, *, source: bool) -> str:
    """Keep staging clauses, not other subjects described in that field.

    A SOURCE's first exact quotation is already attributed by its record, but
    additional clauses must be recognizably staging or name this same entity.
    Author descriptions do not inherit that source attribution. Unknown prose
    is left unverified rather than assigning every direction to the field key.
    """
    positional = re.compile(r"^(?:the\s+)?(?:left|right|foreground|background|middle\s+ground|midground|"
        r"upper|lower|top|bottom|screen[- ]|viewer|at\b|on\b|in\b|near\b|by\b|beside\b|to\b|"
        r"towards?\b|placed\b|positioned\b|located\b|sitting\b|seated\b|standing\b|resting\b|"
        r"her\b|his\b|their\b)|^(?:在|位于|放在|靠|左边|右边|左侧|右侧|画面|镜头|前景|背景|中景|她的|他的)", re.I)
    clauses = []
    for index, raw in enumerate(re.split(r'[,，;；。.!！?？\n]', text)):
        clause = _normal(raw)
        if not clause:
            continue
        # Descriptive subordinate subjects such as "with the sea in the
        # background" are not the stone's depth. Preserve them in raw text,
        # just do not interpret them as coordinates of this entity.
        clause = re.split(r'\b(?:with|while|where|whereas)\b|而(?:人物|角色|她|他)|背景中', clause, maxsplit=1)[0].strip()
        if not clause:
            continue
        named = [reference for name, reference in aliases.items()
                 if name and re.match(re.escape(_normal(name)) + r'(?:\b|的|在|位于|放在)', clause)]
        if named and owner not in named:
            continue
        if owner in named or positional.search(clause) or (source and index == 0):
            clauses.append(clause)
    return '; '.join(clauses)


def validate_author_positions(contract: SceneContractV3, positions: dict[str, str],
                              references: dict[str, str]) -> dict:
    """Reject only quote-supported explicit SOURCE/ART axis oppositions.

    The returned audit keeps original values, quotes and ungrounded axes.
    SOURCE guessed depth is not elevated to an original requirement, and
    absence of a detected conflict is explicitly not a consistency proof.
    """
    aliases = {entity.id: entity.id for entity in contract.entities}
    unique_names: dict[str, list[str]] = {}
    for entity in contract.entities:
        unique_names.setdefault(_normal(entity.name), []).append(entity.id)
    aliases.update({name: refs[0] for name, refs in unique_names.items() if len(refs) == 1})
    aliases.update(references)
    character = next((entity for entity in contract.entities if entity.kind.value == 'character'), None)
    if character is not None:
        aliases.update({key: character.id for key in ('character', 'her', 'him', '她', '他', '人物', '角色')})
    # Normalize character aliases to one semantic referent in both languages.
    parser_aliases = {key: 'character' if character is not None and ref == character.id else ref
                      for key, ref in aliases.items()}
    records = []
    by_entity = {entity.id: entity for entity in contract.entities}
    for ref, authored_text in positions.items():
        entity = by_entity[references[ref]]
        owner = 'character' if character is not None and entity.id == character.id else entity.id
        author_axes = literal_spatial_axes(
            _scoped_axis_text(authored_text, parser_aliases, owner, source=False), parser_aliases)
        for constraint in entity.constraints:
            if constraint.name != 'position':
                continue
            user_evidence = [evidence for evidence in constraint.evidence if evidence.kind == EvidenceKind.USER]
            if not user_evidence:
                continue
            quote = constraint.source_span.text if constraint.source_span else ''
            if quote and quote not in contract.source_text:
                raise ValueError('SOURCE位置的精确引用不在当前原句中；未认证或修补')
            professional = any(evidence.note.startswith('professional_edit;') for evidence in user_evidence)
            reviewed = constraint.intent_role in {ConstraintRole.REQUIREMENT, ConstraintRole.SELECTED_DIRECTION} and any(
                evidence.verified for evidence in user_evidence)
            quote_axes = literal_spatial_axes(
                _scoped_axis_text(quote, parser_aliases, owner, source=True), parser_aliases) if quote else []
            for value in constraint.values:
                source_axes = literal_spatial_axes(
                    _scoped_axis_text(value, parser_aliases, owner, source=True), parser_aliases)
                grounded = [axis for axis in source_axes if axis in quote_axes] if quote else (
                    source_axes if professional or reviewed else [])
                # A quote offering alternatives or two opposing literal
                # directions does not supply one definite axis commitment.
                grounded = [axis for axis in grounded if not any(
                    other.key == axis.key and other.value != axis.value for other in quote_axes)]
                record = {'entity_ref': entity.id, 'condition_ref': constraint.id,
                    'source_value': value, 'source_quote': quote, 'author_ref': ref,
                    'author_position': authored_text, 'source_axes': [asdict(axis) for axis in source_axes],
                    'quote_supported_axes': [asdict(axis) for axis in grounded],
                    'unmatched_source_axes': [asdict(axis) for axis in source_axes if axis not in grounded],
                    'author_axes': [asdict(axis) for axis in author_axes],
                    'evidence': [evidence.model_dump(mode='json') for evidence in user_evidence],
                    'scope': 'explicit_literal_axes_only_not_translation_or_scene_verification'}
                records.append(record)
                for source_axis in grounded:
                    for author_axis in author_axes:
                        if (source_axis.key == author_axis.key and
                                frozenset((source_axis.value, author_axis.value)) in _OPPOSITES):
                            raise ValueError(f'SOURCE位置与作者位置冲突：{entity.id} / {constraint.id} / '
                                f'{source_axis.frame}:{source_axis.reference}:{source_axis.dimension}；'
                                f'来源 {value!r}（引用 {quote!r}）与作者 {authored_text!r} 相反。'
                                '未自动改词、认证来源理解或提交生成')
    return {'schema': 'spatial-realization-check/v1', 'records': records,
        'scope': 'explicit_literal_source_axis_oppositions_only',
        'semantic_correctness_verified': False, 'visual_accuracy_verified': False,
        'complete_spatial_consistency_verified': False,
        'unparsed_and_unquoted_dimensions_remain_unverified': True}
