"""Bounded literal-intent corrections; not a substitute for scene understanding.

Only unambiguous source spans handled here override generated defaults. Unknown
actions remain for the semantic model/review instead of being guessed.
"""
import re

from .schemas import NewbieObjectSpec, NewbieRelationSpec, NewbieSemanticSpec


def matches(text: str, tokens: tuple[str, ...]) -> bool:
    text = text.lower()
    for token in tokens:
        token = token.lower()
        if re.search(r'[a-z]', token):
            if re.search(r'(?<![a-z0-9_])' + re.escape(token) + r'(?![a-z0-9_])', text):
                return True
        elif token in text:
            return True
    return False


def preserve_explicit_intent(spec: NewbieSemanticSpec) -> NewbieSemanticSpec:
    source = spec.source_sentence.lower()
    assignments = {}
    gestures = {}
    # Stop at another hand mention or punctuation; do not attach a later hand's
    # phone/gesture to the earlier hand in the same sentence.
    pattern = r'(左手|右手|left hand|right hand)(.*?)(?=左手|右手|left hand|right hand|[，,；;。.!?]|$)'
    for hand, clause in re.findall(pattern, source):
        side = 'left' if hand in {'左手', 'left hand'} else 'right'
        # This guard only corrects affirmative, unambiguous commands. Leave
        # negation/contrast to semantic extraction instead of reversing it.
        if re.search(r'不|没|别|勿|\b(?:not|no|never|without)\b', clause):
            continue
        if matches(clause, ('手机', 'phone', 'smartphone')) and matches(clause, ('拿', '握', '持', 'hold', 'holds', 'holding')):
            assignments[side] = 'phone'
        if matches(clause, ('v字', 'v sign', 'v-sign', 'peace sign', 'victory sign')):
            gestures[side] = f'{side} hand making a V sign' + (' beside the face' if matches(clause, ('脸旁', '脸边', 'face')) else '')
        elif matches(clause, ('自然垂下', '垂在身侧', 'hanging naturally', 'rests at the side')):
            gestures[side] = f'{side} hand resting naturally at the side'
        elif '向后' in clause and '平衡' in clause:
            gestures[side] = f'{side} arm extended backward for balance'
        elif '向前张开' in clause:
            gestures[side] = f'{side} hand open toward the front with clearly separated fingers'
        elif '护在胸前' in clause:
            gestures[side] = f'{side} hand held protectively in front of the chest'
    phone_sides = [side for side, obj in assignments.items() if obj == 'phone']
    if phone_sides:
        binding = 'both hands' if len(phone_sides) == 2 else f'{phone_sides[0]} hand'
        phone = next((obj for obj in spec.objects if obj.id == 'phone' or matches(obj.name, ('phone', 'smartphone'))), None)
        if phone is None:
            phone = NewbieObjectSpec(id='phone', name='smartphone', appearance=['flat rectangular device'])
            spec.objects.append(phone)
        phone.position = [f'held in {binding}']
        # Remove conflicting phone bindings from body prose and relation slots,
        # not just the left/right fields, before adding the explicit assignment.
        spec.body_action = [value for value in spec.body_action if not matches(value, ('phone', 'smartphone'))]
        if not spec.body_action and matches(source, ('坐', 'sitting', 'seated')):
            spec.body_action = ['sitting naturally']
        spec.relations = [rel for rel in spec.relations
                          if rel.object != phone.id and not rel.object.startswith(phone.id + '.')
                          and not matches(rel.object.replace('_', ' '), ('phone', 'smartphone'))]
        for side in ('left', 'right'):
            field = f'{side}_hand'
            if side in phone_sides:
                setattr(spec, field, [f'{side} hand holding the smartphone'])
                spec.relations.append(NewbieRelationSpec(subject=f'{side} hand', predicate='holds', object=phone.id))
            else:
                setattr(spec, field, [value for value in getattr(spec, field) if not matches(value, ('phone', 'smartphone'))])
    for side, value in gestures.items():
        if side not in assignments:
            setattr(spec, f'{side}_hand', [value])
    closed_eyes = matches(source, ('闭着眼', '闭上眼', '闭眼', 'eyes closed', 'closed eyes'))
    negated_eyes = re.search(r'(?:不|没|别)(?:要|有|再|能|会|是)?\s*(?:闭着眼|闭上眼|闭眼)|\b(?:not|no|never|without)\b[^,;.，；。]{0,20}(?:eyes closed|closed eyes)', source)
    if closed_eyes and not negated_eyes:
        spec.expression = list(dict.fromkeys(['closed eyes', *[value for value in spec.expression
            if not matches(value, ('gaze', 'gazing', 'looking', 'open eyes', 'wide eyes'))
            and not re.search(r'eyes?\b.*\bopen\b', value.lower())]]))
    if re.search(r'触摸胸前的?十字架', source) and not re.search(r'(?:不|没|别)(?:要|有)?触摸胸前', source):
        ornament = next((obj for obj in spec.objects if matches(obj.name, ('cross',)) or matches(obj.id.replace('_', ' '), ('cross',))), None)
        if ornament is None:
            ornament = NewbieObjectSpec(id='cross_ornament', name='cross-shaped ornament', appearance=['recognizable cross shape'], position=['on the chest'])
            spec.objects.append(ornament)
        # The source does not choose left/right; keep that choice unspecified.
        relation = NewbieRelationSpec(subject='one hand', predicate='touches', object=ornament.id)
        if relation not in spec.relations:
            spec.relations.append(relation)
    return spec
