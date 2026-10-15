"""Human-reviewed route experiments, not an autonomous image judge.

Every prefix is one immutable image. Branches share that exact result; a later
hop consumes its immediate predecessor, never silently returns to the root.
"""
from itertools import permutations
from pydantic import BaseModel, Field, model_validator

from ..sampling_profiles import PROFILES


def route_preset(name='pairs'):
    models = list(PROFILES)
    routes = [[a, a, a, a] for a in models]
    for a, b in permutations(models, 2):
        routes.append([a, b, a, b] if name == 'pairs' else [a, b, b, a])
        if name == 'pairs':
            routes.append([a, b, b, b])
    if name == 'triples':
        routes.extend([a, b, c, a] for a, b, c in permutations(models, 3))
    elif name != 'pairs':
        raise ValueError('未知循环路线预设')
    return routes


class LoopDesign(BaseModel):
    model_config = {'extra': 'forbid'}
    routes: list[list[str]] = Field(default_factory=route_preset, min_length=1, max_length=100)
    # Candidate strengths, NOT calibrated equivalents or claimed optima.
    target_denoises: dict[str, float] = Field(default_factory=lambda: {
        'newbie': .15, 'anima': .30, 'illustrious': .30, 'z_image_turbo': .30})

    @model_validator(mode='after')
    def validate_routes(self):
        if set(self.target_denoises) != set(PROFILES) or any(not 0 < x <= 1 for x in self.target_denoises.values()):
            raise ValueError('循环需分别提供全部四个目标模型的重绘强度，范围 (0, 1]')
        if any(not 2 <= len(r) <= 4 or set(r) - set(PROFILES) for r in self.routes):
            raise ValueError('每条路线须含 2–4 个已支持模型（源图加最多三次交接）')
        if len({tuple(r) for r in self.routes}) != len(self.routes):
            raise ValueError('循环路线不得重复；共用前缀会自动复用')
        if {m for r in self.routes for m in r} != set(PROFILES):
            raise ValueError('本轮保留四个模型，请勿省略模型')
        if {r[0] for r in self.routes} != set(PROFILES):
            raise ValueError('本轮保留四种源图，各模型都需要至少一条从它出发的路线')
        return self


def prefixes(routes):
    return sorted({tuple(r[:i]) for r in routes for i in range(1, len(r)+1)}, key=lambda x: (len(x), x))


def append_loop_items(items, sources, design):
    """Topological, depth-first-in-time batches enable matched early stopping."""
    from hashlib import sha256
    roots = {s['axes']['model']: s for s in sources}
    indexed = {(model,): source for model, source in roots.items()}
    options = design.loop
    routes = options.routes
    for route in prefixes(routes):
        if len(route) == 1:
            continue
        parent = indexed[route[:-1]]
        target = roots[route[-1]]
        route_id = sha256('/'.join(route).encode()).hexdigest()[:16]
        label = ' → '.join(route)
        item = {'key': f"{target['axes']['scene']}_{target['axes']['seed']}_loop_{route_id}",
                'pixel_source': {'source_key': parent['key'], 'denoise': options.target_denoises[route[-1]]},
                'axes': target['axes'] | {'pipeline': '_to_'.join(route), 'recipe': label,
                    'route': '/'.join(route), 'round': len(route)-1, 'source_model': route[-2],
                    'denoise': options.target_denoises[route[-1]], 'pair_id': indexed[route[:1]]['key'],
                    'terminal': list(route) in routes},
                'request': target['request']}
        indexed[route] = item
        items.append(item)
    for source in sources:
        model = source['axes']['model']
        source['axes'].update(route=model, round=0, terminal=False)


def loop_summary(design):
    nodes = prefixes(design.loop.routes)
    return {'routes': design.loop.routes, 'terminal_paths': len(design.loop.routes),
            'unique_prefixes_per_case_seed': len(nodes),
            'target_denoises': design.loop.target_denoises,
            'seed_policy': '同一场景/Seed下每次交接固定使用该Seed；不宣称跨架构噪声等价。',
            'decision_policy': '人工比较全部中间版本；不自动选优、修复或宣称自主决策。'}
