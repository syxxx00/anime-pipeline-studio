"""Choose a compatible initial model, without loading, routing chains or I/O writes.

The caller owns the one bounded author call and persists its originals. A file
catalogue is not proof of runtime support: optional live object_info comes from
the caller, never an implicit connection/start/download here. Later image-led
operations use their existing NextChoice; this is only an adaptive initial start.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import Settings
from .creation_options import creation_options
from .model_catalog import model_catalog
from .lora_catalog import describe_lora
from .sampling_profiles import PROFILES
from .schemas import GenerateRequest
from .workflow import build_workflow


SAMPLING_FIELDS = ('steps', 'cfg', 'sampler', 'scheduler', 'shift', 'cfg_rescale')
ParameterSource = Literal['model_default', 'user_override']


class StartingModelChoice(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    target: Literal['newbie', 'illustrious', 'anima', 'z_image_turbo']
    reason: str = Field(min_length=3, max_length=900)
    failure_risks: list[Annotated[str, Field(min_length=3, max_length=300)]] = Field(default_factory=list, max_length=3)

    @field_validator('reason', 'failure_risks')
    @classmethod
    def nonblank_words(cls, value):
        words = value if isinstance(value, list) else [value]
        if any(not word.strip() for word in words):
            raise ValueError('初始选择理由与风险不能只有空白，未代写或修剪原响应')
        return value


STARTING_CHOICE_SYSTEM = """Choose ONE available compatible INITIAL model for this whole
creative task, using the complete original, selected knowledge/style and supplied
historical evidence. Do not create the picture specification or select a fixed
multi-model chain. Models have no permanent roles. Historical findings depend on
their input, parameters and source; they do not certify this task's result.
Choose only a key in compatible_starts. Its configuration is frozen by the host:
do not propose changes to seed, frame, LoRA, style, explicit sampling or stages.
Explain a concrete task-based choice and up to three material failure risks, not
a universal ranking. The initial result still needs actual pixel review. Return
only the schema JSON; no claim of user approval or image quality verification."""


_MODEL_FILES = {
    'newbie': ('diffusion_model', 'vae', 'gemma_clip', 'jina_clip'),
    'anima': ('anima_diffusion_model', 'anima_clip', 'anima_vae'),
    'illustrious': ('illustrious_checkpoint',),
    'z_image_turbo': ('z_diffusion_model', 'z_clip', 'z_vae'),
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def _target(model):
    return {'anima_turbo': 'anima'}.get(str(model), str(model))


def _defaults(target):
    return {**{field: GenerateRequest.model_fields[field].default for field in SAMPLING_FIELDS},
            **{field: value for field, value in PROFILES[target].items() if field in SAMPLING_FIELDS}}


def _sources(config, declared):
    declared = {} if declared is None else declared
    if (not isinstance(declared, dict) or set(declared) - set(SAMPLING_FIELDS)
            or any(value not in {'model_default', 'user_override'} for value in declared.values())):
        raise ValueError('起点参数来源仅支持六个采样字段及model_default/user_override')
    current = _target(config.get('model_id', 'newbie'))
    if current not in PROFILES:
        raise ValueError('当前起点没有已安装的采样配置')
    defaults = _defaults(current)
    result = {}
    for field in SAMPLING_FIELDS:
        # Flat legacy/explicit requests are not classified by numerical equality.
        source = declared.get(field, 'user_override' if field in config else 'model_default')
        if source == 'model_default' and field in config and (
                isinstance(config[field], bool) or config[field] != defaults[field]):
            raise ValueError(f'{field}的model_default声明与当前模型实际默认值不一致，未覆盖')
        result[field] = source
    return result


def _candidate_config(config, target, sources):
    result = deepcopy(config)
    defaults = _defaults(target)
    for field, source in sources.items():
        if source == 'model_default':
            result[field] = defaults[field]
    result.update(model_id=PROFILES[target]['model_id'], render_pipeline=target, parameter_policy='explicit')
    return result


def _file_evidence(settings, target, catalog):
    model = next((item for item in catalog if item.get('id') == PROFILES[target]['model_id']), None)
    if model is None or model.get('ready') is not True:
        raise ValueError('现有本地模型目录没有就绪的完整权重声明')
    files = {}
    for name in _MODEL_FILES[target]:
        path = settings.path_map[name]
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f'当前配置的本地模型文件缺失：{name}')
        files[name] = {'path': str(path.resolve()), 'bytes': path.stat().st_size}
    return files


def _recipe_compatibility(config, target, recipes):
    # Single starts consume the chosen single-model style_preset. Do not invent
    # a different meaning for unused hybrid composition/character channels.
    identifier = config.get('style_preset')
    if not identifier:
        return
    recipe = recipes.get('style_preset')
    if not isinstance(recipe, dict) or recipe.get('id') != identifier:
        raise ValueError('所选完整画风配方没有与当前配置绑定')
    mode = recipe.get('mode')
    if mode not in {'newbie', 'anima', 'both'}:
        raise ValueError('所选画风配方的原生通道来源不明确')
    if mode != 'both' and target != mode:
        raise ValueError('所选完整画风配方未声明此起点的原生格式兼容，未移除或改写')


def _lora_compatibility(config, target, settings):
    name = config.get('lora_name')
    if not name:
        return
    root = settings.path_map['loras'].resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('已选LoRA原件缺失或路径无效，未移除')
    if PROFILES[target]['model_id'] not in describe_lora(path, root)['compatible_models']:
        raise ValueError('已选LoRA与该起点没有明确架构兼容证据，未移除')


def _node_evidence(workflow, object_info, *, deferred_image_files=()):
    classes = sorted({node['class_type'] for node in workflow.values()})
    if object_info is None:
        return {'status': 'not_inspected', 'required_classes': classes,
                'verified': False, 'scope': 'local_files_not_runtime_node_availability'}
    if not isinstance(object_info, dict):
        raise ValueError('调用者提供的节点能力快照不是object_info字典')
    relevant, deferred = {}, []
    for node in workflow.values():
        name = node['class_type']
        info = object_info.get(name)
        if not isinstance(info, dict):
            raise ValueError(f'当前节点能力快照缺少实际执行节点：{name}')
        relevant[name] = info
        fields = {**(info.get('input', {}).get('required', {})),
                  **(info.get('input', {}).get('optional', {}))}
        for key, value in node['inputs'].items():
            description = fields.get(key)
            if (name == 'LoadImage' and key == 'image' and isinstance(value, str)
                    and value in deferred_image_files):
                if not isinstance(description, (list, tuple)) or not description:
                    raise ValueError('局部消费者缺少实际图片输入声明：LoadImage.image')
                deferred.append(value)
                continue  # Capability-only files are materialized before actual job validation.
            if (isinstance(description, (list, tuple)) and description
                    and isinstance(description[0], (list, tuple))
                    and value not in description[0]):
                raise ValueError(f'当前节点未登记所选输入：{name}.{key}')
    result = {'status': 'inspected', 'required_classes': classes, 'verified': True,
              'object_info_sha256': _digest(relevant),
              'scope': 'provided_node_and_enum_snapshot_not_model_load_or_art_quality'}
    if deferred:
        result.update(deferred_image_files=sorted(set(deferred)), input_files_verified=False,
                      scope='static_nodes_and_enums_except_explicit_future_image_files_not_model_load_or_art_quality')
    return result


def _selection_view(target, original):
    """Show decision differences, not repeated whole generation payloads.

    Non-sampling fields are identical frozen selections, already present once
    in selected_config. Original candidate configs/files/nodes remain in the
    owning context; pointers/hashes bind that material, not pretend it was sent.
    """
    config, nodes = original['config'], original['nodes']
    return {'target': target, 'sampling': {field: config[field] for field in SAMPLING_FIELDS},
        'selected_stage_consumption': {
            'scope': 'compatible_single_initial_consumer_not_fixed_followup_chain',
            'highres_enabled': config.get('highres_enabled', False),
            'face_detail_enabled': config.get('face_detail_enabled', GenerateRequest.model_fields['face_detail_enabled'].default),
            'regional_conditioning': config.get('regional_conditioning') is not None},
        'node_snapshot': {key: deepcopy(nodes[key]) for key in ('status', 'verified', 'scope')},
        'originals': {name: {'json_pointer': f'/compatible_starts/{target}/{name}',
                            'sha256': _digest(original[name])}
                      for name in ('config', 'files', 'nodes')},
        'originals_scope': 'complete originals in owning starting-choice context; not repeated model inputs',
        'art_quality_verified': False}


def prepare_starting_choice(full_input: dict, settings: Settings, *, parameter_sources=None,
                            object_info=None, research_context=None) -> dict:
    """Freeze inputs and derive only executable compatible candidate configurations.

    No implicit model call, network/start/download, node import or weight load.
    Explicit starts return unchanged and do not request selection. An adaptive
    start needs a decision only with two compatible configurations; absent live
    nodes remain explicitly unverified, rather than described as running models.
    """
    data = deepcopy(full_input)
    config = deepcopy(data['selected_config'])
    original = data['original_text']
    if not isinstance(original, str) or not original.strip():
        raise ValueError('初始选路需要保留完整原文')
    original_sha = hashlib.sha256(original.encode('utf-8')).hexdigest()
    if data.get('original_sha256') not in {None, original_sha}:
        raise ValueError('初始选路原文与来源哈希不一致')
    context = {'schema_version': 1, 'config_snapshot': config,
               'original_sha256': original_sha, 'compatible_starts': {}, 'excluded_starts': {},
               'needs_choice': False, 'parameter_sources': {},
               'mode': 'adaptive_initial' if config.get('render_pipeline') == 'adaptive' else 'explicit_unchanged'}
    if context['mode'] == 'adaptive_initial':
        sources = _sources(config, parameter_sources)
        context['parameter_sources'] = sources
        catalog = model_catalog(settings)
        routes = {row['id']: row for row in creation_options()['routes']}
        recipes = data.get('full_style_recipes') or {}
        for target in PROFILES:
            try:
                candidate = _candidate_config(config, target, sources)
                if ('negative_prompt' in config and target != _target(config.get('model_id', 'newbie'))):
                    raise ValueError('明确的原生负面覆盖属于当前模型通道，未猜测跨模型格式或移除')
                face = candidate.get('face_detail_enabled', GenerateRequest.model_fields['face_detail_enabled'].default)
                if target != 'newbie' and face:
                    raise ValueError('该起点工作流不消费已选NewBie面部细化阶段，未静默关闭')
                if candidate.get('hand_repair_enabled') or candidate.get('text_clean_enabled'):
                    raise ValueError('已选自动修改阶段不是完整初始单模型消费者，未静默关闭')
                route = routes[target]
                if (candidate['sampler'] not in route['samplers']
                        or candidate['scheduler'] not in route['schedulers']
                        or candidate['steps'] < route['minimum_steps']):
                    raise ValueError('被冻结的采样字段不满足该起点支持范围')
                _recipe_compatibility(candidate, target, recipes)
                _lora_compatibility(candidate, target, settings)
                files = _file_evidence(settings, target, catalog)
                validated = GenerateRequest.model_validate({**candidate,
                    'prompt': 'initial capability inspection only'})
                workflow = build_workflow(validated, settings, 'initial-capability-inspection-only')
                nodes = _node_evidence(workflow, object_info)
                context['compatible_starts'][target] = {'config': candidate,
                    'files': files, 'nodes': nodes, 'art_quality_verified': False}
            except (ValueError, OSError) as exc:
                context['excluded_starts'][target] = str(exc)
        context['needs_choice'] = len(context['compatible_starts']) >= 2
    if research_context is None:
        from .creative_research_context import load_creative_research_context
        research_context = load_creative_research_context('initial_model_choice')
    context['data'] = {key: deepcopy(data[key]) for key in (
        'original_text', 'original_sha256', 'drawing_form', 'knowledge',
        'identity_binding', 'frame', 'full_style_recipes', 'selected_config',
        'feedback', 'current_pixel_observation', 'request_reading_material',
        'creation_intent', 'direction_assignment') if key in data}
    context['data'].update(compatible_starts={target: _selection_view(target, value)
                                            for target, value in context['compatible_starts'].items()},
        excluded_starts=deepcopy(context['excluded_starts']),
        parameter_sources=deepcopy(context['parameter_sources']),
        applicable_research=deepcopy(research_context),
        authority='only adaptive initial model choice; original, knowledge, full recipes and user overrides remain frozen')
    context['context_sha256'] = _digest(context)
    return context


def compatible_choice_material(context: dict) -> dict:
    """Send decision parameters, not host file/node checksum inventories.

    Full compatibility evidence remains in the owning frozen context and is
    consumed when applying the target. The selector cannot verify those hashes;
    its concern is the actual available model and preserved sampling/stages.
    """
    return {target: {key: deepcopy(candidate[key])
                     for key in ('target', 'sampling', 'selected_stage_consumption', 'node_snapshot')}
            for target, candidate in context['data']['compatible_starts'].items()}


def apply_starting_choice(config: dict, choice: StartingModelChoice | dict | None,
                          context: dict) -> tuple[dict, dict]:
    """Apply the recorded choice, not another round of defaults or heuristics."""
    frozen = deepcopy(context)
    signature = frozen.pop('context_sha256')
    if signature != _digest(frozen) or config != frozen['config_snapshot']:
        raise ValueError('初始选路输入或参数已变化，没有采用旧选择')
    if frozen['mode'] == 'explicit_unchanged':
        if choice is not None:
            raise ValueError('显式起点不能被模型选择替换')
        return deepcopy(config), {'origin': 'explicit_unchanged', 'context_sha256': signature,
                                  'quality_verified': False}
    starts = frozen['compatible_starts']
    if choice is None:
        if len(starts) != 1:
            raise ValueError('没有唯一兼容起点；需实际选择或说明当前不可执行原因')
        target = next(iter(starts))
        original = None
        origin = 'only_compatible_existing_start_not_model_judgement'
    else:
        selected = StartingModelChoice.model_validate(choice)
        target, original = selected.target, selected.model_dump(mode='json')
        if target not in starts:
            raise ValueError('初始选择不在当前实际兼容起点中，未替换或重新选择')
        if frozen['needs_choice']:
            origin = 'bounded_initial_model_choice_unverified'
        elif len(starts) == 1:
            # The host's conditions already determine this model. Keep the
            # returned reason as an unverified confirmation, not a new choice
            # or permission to relax any of those conditions.
            origin = 'only_compatible_existing_start_confirmed_not_model_selection'
        else:
            raise ValueError('初始选择不在当前实际兼容起点中，未替换或重新选择')
    actual = deepcopy(starts[target]['config'])
    changes = {field: {'previous': config.get(field), 'selected': actual.get(field),
                      'source': frozen['parameter_sources'][field]}
               for field in SAMPLING_FIELDS if config.get(field) != actual.get(field)}
    return actual, {'origin': origin, 'target': target, 'choice': original,
        'context_sha256': signature, 'original_sha256': frozen['original_sha256'],
        'parameter_changes': changes, 'availability': deepcopy(starts[target]['nodes']),
        'selected_config_sha256': _digest(actual), 'quality_verified': False}
