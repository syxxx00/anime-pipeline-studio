"""Read explicit scene expectations without interpreting or truncating the prompt."""
from .scene_contract import SceneContractV3, EntityKind, ConstraintStrength


def contract_check_spec(contract: SceneContractV3) -> dict:
    characters = [e for e in contract.entities if e.kind == EntityKind.CHARACTER]
    spec = {'source_sentence': contract.source_text, 'expects_character': bool(characters),
            'characters': [], 'objects': [], 'relations': [r.model_dump(mode='json') for r in contract.relations],
            'framing': contract.composition.framing, 'style': contract.style.description,
            'left_hand': [], 'right_hand': [], 'background': []}
    for entity in contract.entities:
        values = {}
        for c in entity.constraints:
            if c.strength != ConstraintStrength.FORBIDDEN:
                values.setdefault(c.name, []).extend(c.values)
        if entity.kind == EntityKind.CHARACTER:
            spec['characters'].append({'id': entity.id, 'name': entity.name, 'model_tags': entity.model_tags, **values})
            for hand in ('left_hand', 'right_hand'):
                spec[hand].extend(values.get(hand, []))
        elif entity.kind == EntityKind.OBJECT:
            spec['objects'].append({'id': entity.id, 'name': entity.name, **values})
        elif entity.kind == EntityKind.ENVIRONMENT:
            spec['background'].extend([entity.name, *[v for vs in values.values() for v in vs]])
    # The old tag detector only supports one expected identity. Do not silently
    # assign the first character's identity to a multi-character scene.
    if len(characters) == 1:
        tags = characters[0].model_tags.get('newbie', [])
        spec['character_tag'] = tags[0] if tags else ''
    return spec


def check_spec(request) -> dict:
    if request.scene_contract:
        return contract_check_spec(SceneContractV3.model_validate(request.scene_contract))
    return request.semantic_spec if isinstance(request.semantic_spec, dict) else {}
