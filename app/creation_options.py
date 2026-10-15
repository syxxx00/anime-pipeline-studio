"""Creation controls for both ordinary defaults and professional overrides.

This describes supported configuration, not comparative model capability.
Legacy automatic hand/text modification switches intentionally stay private.
"""
from .schemas import GenerateRequest, SourceStageParameters
from .sampling_profiles import PROFILES, SOURCE_PROFILES


def creation_options() -> dict:
    properties = GenerateRequest.model_json_schema()['properties']
    groups = [
        ('画幅与数量', [('width', '宽度', 16), ('height', '高度', 16), ('batch_size', '图片数量', 1)]),
        ('采样参数', [('steps', '采样步数', 1), ('cfg', '提示词引导 CFG', .1), ('shift', '采样偏移 Shift', .1), ('cfg_rescale', 'CFG 重缩放', .05)]),
        ('输出细化', [('highres_scale', '放大倍率', .1), ('highres_steps', '放大采样步数', 1), ('highres_denoise', '放大重绘强度', .01)]),
        ('NewBie 脸部细化', [('face_detail_enabled', '脸部细化（仅 NewBie）', None), ('face_detail_steps', '脸部细化步数', 1), ('face_detail_denoise', '脸部重绘强度', .01)]),
        ('协作交接', [('hybrid_denoise', '后续 NewBie 阶段重绘强度', .01)]),
        ('检查与资源', [('check_enabled', '生成后检查与报告', None), ('lora_strength', 'LoRA 强度', .05)]),
    ]
    defaults = {key: properties[key].get('default') for _, fields in groups for key, _, _ in fields}
    defaults.update(PROFILES['newbie'], parameter_policy='explicit', render_pipeline='adaptive', sampler='res_multistep', scheduler='linear_quadratic', seed=None, lora_name=None, composition_style_preset=None, character_style_preset=None)
    # Fresh ordinary/professional creations share opt-in pixel modification.
    # The daily engine does not load FaceDetailer's optional nodes; do not make
    # every adaptive start require them. Explicit and saved choices are intact.
    defaults['face_detail_enabled'] = False
    routes = [
        ('adaptive', '自动选择现有路线', 'newbie'), ('newbie', 'NewBie', 'newbie'),
        ('anima', 'Anima Turbo', 'anima'), ('illustrious', 'Illustrious', 'illustrious'),
        ('z_image_turbo', 'Z-Image Turbo', 'z_image_turbo'),
        ('anima_newbie', 'Anima → NewBie', 'newbie'),
        ('illustrious_newbie', 'Illustrious → NewBie', 'newbie'),
        ('z_newbie', 'Z-Image → NewBie', 'newbie'),
    ]
    return {
        'capabilities': {'creative_initial_selection': True, 'creative_model_neutral_highres': True,
                         'sampling_parameter_sources': True,
                         'creative_joint_picture_decision': True,
                         'creative_research_catalog': True,
                         'creative_prose_picture': True, 'creative_request_understanding': True,
                         'creative_candidate_directions': True},
        'defaults': defaults,
        'source_profiles': {route: PROFILES[profile] for route, profile in SOURCE_PROFILES.items()},
        'source_parameter_schema': SourceStageParameters.model_json_schema()['properties'],
        'routes': [dict(
            id=id, name=name, profile=PROFILES[profile], target={'anima': 'anima', 'z_image_turbo': 'z_image'}.get(profile, profile),
            samplers=['euler'] if profile in {'anima', 'z_image_turbo'} else (['euler_ancestral', 'dpmpp_2m_sde'] if profile == 'illustrious' else ['res_multistep', 'euler', 'euler_ancestral', 'dpmpp_2m_sde']),
            schedulers=['simple'] if profile in {'anima', 'z_image_turbo'} else (['normal', 'karras'] if profile == 'illustrious' else ['linear_quadratic']),
            minimum_steps=20 if profile == 'illustrious' else (10 if profile == 'newbie' else 1),
        ) for id, name, profile in routes],
        'groups': [dict(name=name, fields=[dict(key=key, label=label, step=step, schema=properties[key]) for key, label, step in fields]) for name, fields in groups],
        'samplers': ['res_multistep', 'euler', 'euler_ancestral', 'dpmpp_2m_sde'],
        'schedulers': ['linear_quadratic', 'simple', 'normal', 'karras'],
    }
