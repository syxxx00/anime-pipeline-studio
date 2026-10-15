"""Pure merge shared with the isolated local language-model process."""
from copy import deepcopy


def merge_semantic_patch(fallback: dict, patch: dict) -> dict:
    result = deepcopy(fallback)
    warnings = []
    simple_fields = ('expression', 'body_action', 'left_hand', 'right_hand', 'framing', 'background', 'lighting')
    for key in simple_fields:
        value = patch.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            # Empty is an explicit extraction result, not "keep the old guess".
            result[key] = value
        else:
            warnings.append(f'语义模型未提供有效的 {key} 数组，此字段保留规则结果，尚未验证。')
    for key, required in (('objects', ('id', 'name')), ('relations', ('subject', 'predicate', 'object'))):
        value = patch.get(key)
        valid = isinstance(value, list) and all(
            isinstance(item, dict) and all(isinstance(item.get(name), str) for name in required)
            and (key != 'objects' or all(isinstance(item.get(name, []), list)
                 and all(isinstance(part, str) for part in item.get(name, [])) for name in ('appearance', 'position')))
            for item in value)
        if valid:
            result[key] = value
        else:
            warnings.append(f'语义模型未提供有效的 {key} 数组，此字段保留规则结果，尚未验证。')
    clause_map = patch.get('clause_map')
    if not isinstance(clause_map, list) or not all(isinstance(item, dict) for item in clause_map):
        clause_map = []
    return {'spec': result, 'clause_map': clause_map, 'patch_warnings': warnings}
