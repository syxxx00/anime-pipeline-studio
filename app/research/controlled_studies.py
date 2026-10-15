"""Controlled research with explicit model conditioning and a human-reviewed baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator, field_validator

from ..check_intent import contract_check_spec
from ..config import PROJECT_ROOT, load_settings
from ..contract_compilers import compile_contract, compile_contract_english
from ..model_conditioning import NEGATIVE_BASELINES, merge_negative
from ..sampling_profiles import PROFILES
from ..scene_contract import SceneContractV3
from ..schemas import ExperimentCreateRequest, GenerateRequest
from ..workflow import build_workflow
from .study_loops import LoopDesign, append_loop_items, loop_summary, route_preset

CATALOG = PROJECT_ROOT / 'config' / 'controlled_studies.json'
PARAMETERS = {'steps', 'cfg', 'sampler', 'scheduler', 'shift', 'cfg_rescale',
              'weight_dtype', 'lora_name', 'lora_strength'}


class PromptOverride(BaseModel):
    model_config = {'extra': 'forbid'}
    positive: str | None = Field(default=None, min_length=1, max_length=32000)
    negative: str | None = Field(default=None, max_length=16000)
    note: str = Field(default='', max_length=2000)


class ModelConditioning(BaseModel):
    model_config = {'extra': 'forbid'}
    parameters: dict
    negative: str = Field(max_length=16000)
    baseline_source: str = Field(min_length=1, max_length=2000)
    empty_negative_reason: str = Field(default='', max_length=2000)

    @model_validator(mode='after')
    def explicit_parameters(self):
        if set(self.parameters) != PARAMETERS:
            raise ValueError('模型参数必须完整列出，且不能混入隐藏设置：' + ', '.join(sorted(PARAMETERS)))
        if not self.negative.strip() and not self.empty_negative_reason.strip():
            raise ValueError('空负面词必须说明实验理由，不会自动补入或删除')
        return self


def default_models():
    return {model: {
        'parameters': {key: value for key, value in profile.items() if key != 'model_id'} | {
            'cfg_rescale': .9, 'weight_dtype': 'fp8_e5m2' if model == 'z_image_turbo' else ('default' if model != 'illustrious' else None),
            'lora_name': None, 'lora_strength': .75},
        'negative': NEGATIVE_BASELINES[model],
        'baseline_source': '项目已有本地采样设置与负面词；不是最优参数结论，需运行并人工检查基线',
        'empty_negative_reason': '',
    } for model, profile in PROFILES.items()}


class StudyCase(BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,60}$')
    name: str = Field(min_length=1, max_length=120)
    intent: str = Field(min_length=1, max_length=4000)
    short: SceneContractV3
    detailed: SceneContractV3
    inspect: list[str] = Field(min_length=1, max_length=40)
    # Changes are model/variant-specific, never silently propagated to another
    # model or to the other member of a pair.
    prompt_overrides: dict[str, dict[str, PromptOverride]] = Field(default_factory=dict)

    @field_validator('short', 'detailed', mode='before')
    @classmethod
    def no_silent_contract_fields(cls, value):
        return SceneContractV3.model_validate(value, extra='forbid')

    @model_validator(mode='after')
    def valid_overrides(self):
        if set(self.prompt_overrides) - {'english', 'structured', 'short', 'detailed', 'source'}:
            raise ValueError('未知提示词对照条件')
        for targets in self.prompt_overrides.values():
            if set(targets) - set(PROFILES):
                raise ValueError('提示词覆盖引用未知模型')
            if any(not o.note.strip() for o in targets.values()):
                raise ValueError('直接编辑原生文本需记录修改理由；不会声称改写后仍严格等义')
        return self


class StudyDesign(BaseModel):
    model_config = {'extra': 'forbid'}
    cases: list[StudyCase] = Field(min_length=1, max_length=40)
    models: dict[str, ModelConditioning]
    seeds: list[int] = Field(default_factory=lambda: [42, 3407, 123456], min_length=1, max_length=32)
    width: int = Field(default=1024, ge=512, le=1536)
    height: int = Field(default=1024, ge=512, le=1536)
    denoises: list[float] = Field(default_factory=lambda: [.15, .30, .50], min_length=1, max_length=16)
    phase: Literal['baseline', 'full'] = 'full'
    baseline_experiment_id: str = Field(default='', max_length=120)
    loop: LoopDesign = Field(default_factory=LoopDesign)

    @model_validator(mode='after')
    def consistent_axes(self):
        if set(self.models) != set(PROFILES):
            raise ValueError('本研究保留全部四个模型；必须分别提供完整模型配置')
        for model, options in self.models.items():
            validated = GenerateRequest.model_validate({'model_id': PROFILES[model]['model_id'],
                'render_pipeline': model, 'parameter_policy': 'explicit', 'prompt': 'parameter validation', **options.parameters})
            options.parameters = {key: validated.model_dump(mode='json')[key] for key in PARAMETERS}
        if len({c.id for c in self.cases}) != len(self.cases) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError('场景 ID 和 Seed 不得重复')
        if any(s < 0 or s > 2**53-1 for s in self.seeds) or self.width % 16 or self.height % 16:
            raise ValueError('Seed 超出网页精确整数范围，或画幅不是 16 的倍数')
        if len(set(self.denoises)) != len(self.denoises) or any(not 0 < d <= 1 for d in self.denoises):
            raise ValueError('重绘强度必须不重复且位于 (0, 1]')
        return self


def catalog(path=CATALOG):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def conditioning(case, contract, model, variant, design):
    channel = 'z_image' if model == 'z_image_turbo' else model
    compiled = compile_contract(contract, channel)
    if compiled.postprocess:
        raise ValueError('该研究执行器尚未接通精确文字排版操作；不能把未执行的后处理当作已完成，请使用独立文字编辑功能')
    positive = compile_contract_english(contract).positive if variant == 'english' else compiled.positive
    baseline = design.models[model]
    negative = merge_negative(model, baseline.negative, compiled.negative)
    override = case.prompt_overrides.get(variant, {}).get(model)
    if override:
        if override.positive is not None:
            positive = override.positive
        if override.negative is not None:
            negative = override.negative
    if variant != 'english' and model == 'newbie':
        try:
            root = ET.fromstring(positive)
        except ET.ParseError as exc:
            raise ValueError('NewBie 原生条件不是可解析 XML') from exc
        if root.tag != 'img':
            raise ValueError('NewBie 原生 XML 须以 img 为根节点')
    return {
        'positive': positive, 'negative': negative,
        'trace': {} if override or variant == 'english' else compiled.trace,
        'trace_status': 'manual_override_unverified' if override else 'compiled_payload_presence_only',
        'baseline_source': baseline.baseline_source,
        'override_note': override.note if override else '',
        'negative_effect': 'CFG=1 时常规采样不使用负面分支' if baseline.parameters['cfg'] == 1 else '参与 CFG；是否有效需实测',
    }


def make_request(case, contract, model, variant, seed, design):
    native = {('z_image' if m == 'z_image_turbo' else m): conditioning(case, contract, m, variant, design) for m in PROFILES}
    channel = native['z_image' if model == 'z_image_turbo' else model]
    return GenerateRequest.model_validate({
        'model_id': PROFILES[model]['model_id'], **design.models[model].parameters,
        'render_pipeline': model, 'parameter_policy': 'explicit', 'seed': seed,
        'width': design.width, 'height': design.height, 'batch_size': 1,
        'prompt': channel['positive'], 'negative_prompt': channel['negative'], 'use_default_negative_prompt': False,
        'native_prompts': native, 'semantic_spec': contract_check_spec(contract),
        'scene_contract': contract.model_dump(mode='json'),
        'highres_enabled': False, 'face_detail_enabled': False,
        'hand_repair_enabled': False, 'text_clean_enabled': False, 'check_enabled': True,
    })


def build_study(study_id, design: StudyDesign, definition: dict):
    if study_id not in {'format', 'detail', 'handoff', 'loops'}:
        raise ValueError('该研究尚无可执行入口；不会用固定路线冒充自主迭代')
    items = []
    seeds = design.seeds[:1] if design.phase == 'baseline' else design.seeds
    for case in design.cases:
        for seed in seeds:
            native_variant = 'structured' if study_id == 'format' else ('detailed' if study_id == 'detail' else 'source')
            variants = [(native_variant, case.detailed)]
            if design.phase == 'full':
                variants = [('english', case.detailed), ('structured', case.detailed)] if study_id == 'format' else (
                    [('short', case.short), ('detailed', case.detailed)] if study_id == 'detail' else variants)
            for variant, contract in variants:
                for model in PROFILES:
                    request = make_request(case, contract, model, variant, seed, design)
                    axes = {'character': case.name, 'scene': case.id, 'model': model, 'pipeline': model,
                            'seed': seed, 'prompt_form': variant, 'study': study_id, 'phase': design.phase,
                            'pair_id': f'{case.id}_{seed}_{model}', 'recipe': variant}
                    items.append({'key': f'{case.id}_{seed}_{variant}_{model}', 'axes': axes,
                                  'request': request.model_dump(mode='json')})
            if study_id == 'handoff' and design.phase == 'full':
                sources = items[-len(PROFILES):]
                for source in sources:
                    for target in sources:
                        for denoise in design.denoises:
                            source_model, target_model = source['axes']['model'], target['axes']['model']
                            route = f'{source_model}_to_{target_model}'
                            items.append({
                                'key': f'{case.id}_{seed}_{route}_{denoise}',
                                'pixel_source': {'source_key': source['key'], 'denoise': denoise},
                                'axes': target['axes'] | {'pipeline': route, 'prompt_form': 'source',
                                    'recipe': f'{route}_d{denoise}', 'source_model': source_model,
                                    'denoise': denoise, 'pair_id': source['key']},
                                'request': target['request'],
                            })
            if study_id == 'loops' and design.phase == 'full':
                append_loop_items(items, items[-len(PROFILES):], design)
    if study_id == 'loops':
        # Finish a whole comparison round across all scenes/seeds before the
        # next round. Pausing early must not preferentially expose one route.
        items.sort(key=lambda item: item['axes'].get('round', 0))
    name = definition['name'] + (' · 基线校验' if design.phase == 'baseline' else '')
    return ExperimentCreateRequest.model_validate({'name': name, 'items': items,
        'notes': definition['question'] + '\n' + definition['limits'] + '\n执行完成不代表质量合格；基线需人工查看。原始模型输出，不自动修脸/修手/擦字。',
        'external_llm_used': False})


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def workflow_signature(graph):
    graph = json.loads(json.dumps(graph))
    for node in graph.values():
        if node['class_type'] == 'KSampler':
            node['inputs']['seed'] = 0
        if node['class_type'] == 'SaveImage':
            node['inputs']['filename_prefix'] = 'research_preview'
    return _digest(graph)


def preview(study_id, design, data, settings=None):
    settings = settings or load_settings()
    definition = next((d for d in data['studies'] if d['id'] == study_id), None)
    if definition is None:
        raise ValueError('未知研究项目')
    manifest = build_study(study_id, design, definition)
    representative = {}
    calibration = []
    native_variant = 'structured' if study_id == 'format' else ('detailed' if study_id == 'detail' else 'source')
    for case in design.cases:
        for model in PROFILES:
            request = make_request(case, case.detailed, model, native_variant, 0, design)
            if request.lora_name:
                from ..lora_catalog import describe_lora
                root = settings.path_map['loras'].resolve()
                path = (root / request.lora_name).resolve()
                if not path.is_relative_to(root) or not path.is_file() or request.model_id.value not in describe_lora(path, root)['compatible_models']:
                    raise ValueError(f'{model} LoRA 不存在或缺少兼容架构证据')
            calibration.append({'case': case.id, 'model': model, 'contract': request.scene_contract,
                                'workflow': build_workflow(request, settings, 'research_preview')})
    for item in manifest.items:
        key = f"{item.axes['scene']}:{item.axes['model']}:{item.axes['prompt_form']}"
        if key not in representative and item.pixel_source is None:
            graph = build_workflow(item.request, settings, 'research_preview')
            representative[key] = {'workflow': graph, 'request': item.request.model_dump(mode='json')}
    # Model file metadata is a cheap change detector, not a cryptographic model
    # identity or a guarantee that a live worker has reloaded changed weights.
    files = {}
    for key, path in settings.path_map.items():
        if key in {'diffusion_model', 'vae', 'gemma_clip', 'jina_clip', 'anima_diffusion_model',
                   'anima_clip', 'anima_vae', 'z_diffusion_model', 'z_clip', 'z_vae', 'illustrious_checkpoint'}:
            stat = path.stat() if path.is_file() else None
            files[key] = {'path': str(path), 'size': stat.st_size if stat else None, 'mtime_ns': stat.st_mtime_ns if stat else None}
    for model, options in design.models.items():
        if options.parameters['lora_name']:
            path = settings.path_map['loras'] / options.parameters['lora_name']
            stat = path.stat()
            files[model + '_lora'] = {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    calibration_fingerprint = _digest({'calibration': calibration, 'model_files': files})
    snapshot = {'study_id': study_id, 'design': design.model_dump(mode='json'),
                'manifest': manifest.model_dump(mode='json'), 'execution': representative, 'model_files': files}
    warnings = ['配置来源是本地已有实现，未宣称最优；需先运行基线并人工确认。', '检查器不能可靠判定手指、接触关系、严格 OOC 或美感。']
    if any(v.parameters['cfg'] == 1 for v in design.models.values()):
        warnings.append('CFG=1 的模型通常不使用负面条件；不能将相同负面文本视为相同作用。')
    if any(c.prompt_overrides for c in design.cases):
        warnings.append('存在手工提示词覆盖，须核对是否改变内容；编译追踪不再作为等义证据。')
    if study_id == 'loops':
        snapshot['loop_summary'] = loop_summary(design)
        if design.phase == 'full':
            from ..image_handoff import build_handoff_workflow
            templates = {}
            for key, entry in representative.items():
                request = entry['request']
                target = request['render_pipeline']
                planned = GenerateRequest.model_validate(request | {'operation': 'image_handoff', 'parent_artwork_id': 'pending-parent',
                    'image_handoff': {'source_file': 'PARENT_OUTPUT_NOT_YET_GENERATED.png',
                        'source_sha256': '0'*64, 'source_artwork_id': 'pending-parent',
                        'source_model_id': 'bound-at-dispatch', 'denoise': design.loop.target_denoises[target]}})
                templates[key] = {'workflow': build_handoff_workflow(planned, settings,
                    'PARENT_OUTPUT_NOT_YET_GENERATED.png', 'research_preview'),
                    'binding': '预览模板：LoadImage 在派发时绑定直接父项的输出；真实来源和 SHA256 写入实际作业，不使用预览占位值。'}
            snapshot['handoff_templates'] = templates
        warnings += ['循环每一步使用同一场景要求、目标模型自己的完整原生正负文本；本轮不混入自动拓词或自动修复。',
                     '默认强度只是候选设置，不是等效强度或最优结论。NewBie 使用 0.15；0.5 在本地调度下接近完全重新加噪。',
                     '共享前缀只生成一次，不是独立重复样本。固定 Seed 回访可能收敛、振荡或退化，不能推广到其他强度/Seed/文本。']
    return {'digest': _digest(snapshot), 'total': len(manifest.items),
            'source_jobs': sum(x.pixel_source is None for x in manifest.items),
            'handoff_jobs': sum(x.pixel_source is not None for x in manifest.items),
            'calibration_fingerprint': calibration_fingerprint, 'warnings': warnings,
            'baseline_workflows': {entry['case'] + ':' + entry['model']: workflow_signature(entry['workflow']) for entry in calibration},
            'snapshot': snapshot, 'starts_automatically': False}


def require_baseline(experiments, design, result, acknowledged):
    if design.phase == 'baseline':
        return
    baseline = experiments.get(design.baseline_experiment_id)
    if not baseline or baseline.design.get('phase') != 'baseline':
        raise ValueError('完整矩阵需要选择本入口创建的基线实验')
    if baseline.design.get('calibration_fingerprint') != result['calibration_fingerprint']:
        raise ValueError('基线的场景、原生输入、模型文件或采样设置与当前设计不一致，请重新校验')
    if not baseline.items or any(i.status != 'completed' or len(i.files) != 1 or not Path(i.files[0]).is_file() for i in baseline.items):
        raise ValueError('基线尚未全部产出可查看的图片；不能用队列状态冒充校验完成')
    expected = result['baseline_workflows']
    seen = set()
    for item in baseline.items:
        key = str(item.axes.get('scene')) + ':' + str(item.axes.get('model'))
        source = Path(item.files[0])
        workflow_path = source.with_name(source.name.rsplit('__', 1)[0] + '__workflow.json')
        try:
            actual = workflow_signature(json.loads(workflow_path.read_text(encoding='utf-8')))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError('基线缺少可核验的实际执行工作流，不能只凭状态通过') from exc
        if key in seen or expected.get(key) != actual:
            raise ValueError('基线实际工作流与当前配置不一致，不能用计划参数代替执行证据')
        seen.add(key)
    if seen != set(expected):
        raise ValueError('基线未覆盖当前全部模型与场景')
    if acknowledged is not True:
        raise ValueError('请查看基线图片，确认配置能正常出图后再创建完整矩阵；机器不代替你认定质量')


def baseline_candidates(experiments, design, result):
    """Reuse across study types only when saved execution evidence matches."""
    candidates = []
    for run in experiments.experiments.values():
        if run.design.get('phase') != 'baseline':
            continue
        candidate = {'id': run.id, 'name': run.name, 'state': run.state, 'reusable': False}
        trial = design.model_copy(update={'phase': 'full', 'baseline_experiment_id': run.id})
        try:
            require_baseline(experiments, trial, result, True)
        except ValueError as exc:
            candidate['reason'] = str(exc)
        else:
            candidate['reusable'] = True
            candidate['reason'] = '配置与实际执行工作流匹配；仍需确认已查看图片'
        candidates.append(candidate)
    return candidates


def studies_router(experiments, *, data_path=CATALOG):
    router = APIRouter()

    @router.get('/api/research/studies')
    async def list_studies():
        data = catalog(data_path)
        data['loop_presets'] = {name: route_preset(name) for name in ('pairs', 'triples')}
        return data

    @router.post('/api/research/studies/{study_id}/preview')
    async def preview_study(study_id: str, payload: StudyDesign):
        try:
            result = preview(study_id, payload, catalog(data_path), experiments.settings)
            result['baseline_candidates'] = baseline_candidates(experiments, payload, result)
            return result
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post('/api/research/studies/{study_id}/create')
    async def create_study(study_id: str, payload: dict):
        if payload.get('confirmed') is not True:
            raise HTTPException(409, '请审阅实际文本与数量后确认；创建不会启动运行')
        try:
            design = StudyDesign.model_validate(payload.get('design'))
            result = preview(study_id, design, catalog(data_path), experiments.settings)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        if payload.get('digest') != result['digest']:
            raise HTTPException(409, '预览后设计或模型设置已改变，请重新预览并确认')
        token = str(payload.get('creation_token', ''))
        if not token or len(token) > 100:
            raise HTTPException(422, '缺少有效创建标识')
        for existing in experiments.experiments.values():
            if existing.design.get('creation_token') == token:
                if existing.design.get('digest') != result['digest']:
                    raise HTTPException(409, '创建标识已用于另一份设计')
                return experiments.serialize(existing)
        try:
            require_baseline(experiments, design, result, payload.get('baseline_reviewed'))
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        manifest = ExperimentCreateRequest.model_validate(result['snapshot']['manifest'])
        return experiments.create(manifest, design={'study_id': study_id, 'digest': result['digest'],
            'phase': design.phase, 'calibration_fingerprint': result['calibration_fingerprint'],
            'baseline_reviewed': payload.get('baseline_reviewed') is True,
            'execution': result['snapshot']['execution'], 'model_files': result['snapshot']['model_files'],
            'creation_token': token, 'input': result['snapshot']['design']})

    return router
