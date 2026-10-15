"""Explicit SOURCE correction with a closed dispatch gate, not a meaning judge.

The entire replacement interpretation and artistic proposal are reviewed data.
History remains available for research; it cannot certify the new task scope.
This control neither invokes a model nor resumes or submits a generation job.
"""
from __future__ import annotations

from copy import deepcopy

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .art_direction import AuthorDraft, compile_author_draft, fingerprint
from .authoring_contract import SourceExtraction
from .contract_builder import rebuild_current_contract
from .scene_contract import SceneContractV3
from .schemas import GenerateRequest, NewbieCompileRequest, NewbieSemanticSpec
from .source_continuity import (attribute_source_continuity, current_source,
    preserve_source_metadata, validate_source_continuity, validate_source_author_conditions)


class SourceRevisionConflict(ValueError):
    """Current context or a live operation changed; do not apply stale work."""


class SourceRevisionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expected_revision: int = Field(ge=0, strict=True)
    expected_decision_count: int = Field(ge=0, strict=True)
    expected_source_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    feedback_signature: str = Field(pattern=r'^[0-9a-f]{64}$')
    extraction: SourceExtraction
    changes: list[dict] = Field(min_length=1, max_length=128)
    author_proposal: AuthorDraft

    @field_validator('author_proposal', mode='before')
    @classmethod
    def explicitly_complete_draft(cls, value):
        if isinstance(value, AuthorDraft):
            if value.model_fields_set != set(AuthorDraft.model_fields):
                raise ValueError('来源修订须明确提供全部作者字段，不能把默认补齐冒充完整新稿')
            value = value.model_dump(mode='json')
        if not isinstance(value, dict) or set(value) != set(AuthorDraft.model_fields):
            raise ValueError('来源修订须明确提供全部作者字段且不能夹带来源或认证字段')
        return value


def source_scope(request: GenerateRequest) -> str:
    """Same original text does not imply the same current interpretation."""
    intent = request.creation_intent
    source = current_source(request.author_proposal,
        intent.original_text if intent else '', intent.original_sha256 if intent else '')
    return fingerprint({'source': source.snapshot if source else None,
                        'intent': intent.model_dump(mode='json') if intent else None})


def apply_source_revision(controller, run, payload):
    from .iteration_controller import target_request
    from .generation_request import prepare_generation_request
    from .source_revision import reinterpret_source

    item = SourceRevisionRequest.model_validate(payload)
    controller._verify_original(run)
    tick = controller._tick_tasks.get(run.id)
    lock = controller._locks.get(run.id)
    if (run.status != 'paused' or item.expected_revision != run.revision
            or item.expected_decision_count != len(run.decisions)):
        raise SourceRevisionConflict('请先暂停并读取最新解释与任务版本；没有应用过期修订')
    if (run.id in getattr(controller.planner, 'pending', {})
            or run.id in getattr(controller.planner, 'cancellations', {})
            or (tick is not None and not tick.done()) or (lock is not None and lock.locked())
            or any(step.status in {'reserved', 'submitted'} for step in run.steps)):
        raise SourceRevisionConflict('仍有在途推理、决策或图像；等待它结束后再修订，未取消或重发')
    referenced = {step.job_id for step in run.steps if step.job_id}
    owned = {job.id for job in controller.jobs.list()
             if job.request.get('output_group') == f'iteration_runs/{run.id}'}
    if owned - referenced or any(job.status not in {'completed', 'failed', 'cancelled'}
                                for job in controller.jobs.list() if job.id in owned | referenced):
        raise SourceRevisionConflict('图像提交尚未结束或归属未确认；没有覆盖其解释')
    if item.feedback_signature != controller.feedback_signature(run):
        raise SourceRevisionConflict('用户评价已经变化；请重新查看原话与修订依据')
    base = GenerateRequest.model_validate(run.base_request)
    previous_source = current_source(base.author_proposal, run.original_text, run.original_sha256)
    if previous_source is None or item.expected_source_sha256 != previous_source.snapshot_sha256:
        raise SourceRevisionConflict('当前SOURCE解释已变化或不存在；未修订历史快照')
    revised = reinterpret_source(base, item.extraction, item.changes,
                                 expected_source_sha256=item.expected_source_sha256)
    sources = revised.sources
    if sources.blocking:
        raise ValueError('修订后仍有来源冲突；先解决理解，不编译竞争条件')
    draft = item.author_proposal
    if [goal.model_dump(mode='json') for goal in draft.goals] != [goal.model_dump(mode='json') for goal in sources.goals]:
        raise ValueError('完整作者方案必须承接修订后的目标，不能保留旧目标或另造要求')
    if draft.conflicts:
        raise ValueError('作者方案仍有冲突；没有将修订标成可生成')
    spec = NewbieSemanticSpec.model_validate(draft.spec)
    if spec.source_sentence != run.original_text or spec.style_mode != sources.style_mode:
        raise ValueError('新方案必须保留完整原句和当前画面模式')
    proposal = deepcopy(base.author_proposal)
    proposal.update(source_conditions=sources.model_dump(mode='json'),
                    source_conditions_status='current_unverified_source_interpretation')
    new_source = current_source(proposal, run.original_text, run.original_sha256)
    validate_source_continuity(new_source, spec)
    validate_source_author_conditions(new_source, draft, spec)
    # Keep reviewed commitments; SOURCE correction does not authorize changing
    # unrelated user choices or turning declarations into verified canon.
    contract = rebuild_current_contract(SceneContractV3.model_validate(base.scene_contract), spec,
        compiler='explicit_source_reinterpretation_unverified', strict_entity_refs=True)
    contract, continuity = attribute_source_continuity(new_source, contract)
    knowledge = sources.knowledge
    compile_request = NewbieCompileRequest(sentence=run.original_text, style_mode=sources.style_mode,
        knowledge_pack=knowledge.pack if knowledge.selected else '',
        knowledge_operator=knowledge.operator if knowledge.selected else '',
        knowledge_appearance=knowledge.appearance if knowledge.selected else '')
    compiled = compile_author_draft(draft, compile_request,
        proposal_id=proposal.get('id', run.id), revision=int(proposal.get('revision', 0)) + 1,
        knowledge=knowledge.model_dump(mode='json'), recipes=proposal.get('recipe_snapshot'),
        config=base.model_dump(mode='json'), contract_override=contract,
        apply_selected_knowledge=False, preserve_style_conditions=True,
        runtime_evidence={'source_revision': revised.receipt, 'automatic_quality_certification': False})
    preserve_source_metadata(compiled['author_proposal'], proposal, new_source, continuity)
    compiled['author_proposal'].update(author='explicit_source_reinterpretation_and_art_proposal_unverified',
                                      source_revision_receipt=deepcopy(revised.receipt))
    # An explicit empty native negative remains an explicit user condition.
    for channel, old in (base.native_prompts or {}).items():
        policy = old.get('negative_policy') or {}
        if old.get('negative') == '' and policy.get('scope') != 'model_default_and_current_constraints':
            compiled['native_prompts'][channel]['negative'] = ''
            compiled['native_prompts'][channel]['negative_policy'] = {
                'scope': 'explicit_empty_preserved', 'semantic_correctness_verified': False}
    intent = deepcopy(compiled['creation_intent'])
    intent['research_refs'] = list(base.creation_intent.research_refs)
    intent['interpretation_method'] = 'explicit_source_reinterpretation_unverified'
    channel = {'anima': 'anima', 'illustrious': 'illustrious', 'z_image_turbo': 'z_image'}.get(base.model_id.value, 'newbie')
    native = compiled['native_prompts']
    updated = GenerateRequest.model_validate(base.model_dump(mode='json') | {
        'semantic_spec': compiled['spec'], 'scene_contract': compiled['contract_v3'],
        'native_prompts': native, 'author_proposal': compiled['author_proposal'],
        'creation_intent': intent, 'prompt': native[channel]['positive'],
        'negative_prompt': native[channel]['negative'], 'use_default_negative_prompt': False,
        'repair_context_prompt': '', 'repair_context_policy': 'legacy_local_context'})
    for target in run.allowed_models:
        candidate = target_request(updated, target)
        controller._validate_resources(candidate)
        prepare_generation_request(candidate, controller.jobs.settings)
    staged = run.model_copy(deep=True)
    staged.source_revision_history.append({
        'run_revision_before': run.revision, 'step_count': len(run.steps),
        'decision_count': len(run.decisions), 'scope': 'historical_interpretation_not_current_quality_or_handoff_authority',
        'receipt': deepcopy(revised.receipt), 'previous_base_request': deepcopy(run.base_request),
        'activation': {'new_interpretation_effective': True, 'art_rebinding_completed': True,
            'creation_intent_rebinding_completed': True, 'current_source_scope_sha256': source_scope(updated),
            'semantic_correctness_verified': False, 'visual_accuracy_verified': False,
            'dispatch_performed': False, 'resumed': False},
        'previous_candidate_reviews': [review.model_dump(mode='json') for review in run.candidate_reviews],
        'previous_exploration_branches': [branch.model_dump(mode='json') for branch in run.exploration_branches],
        'previous_exploration_artwork_ids': list(run.exploration_artwork_ids),
        'previous_incumbent_artwork_id': run.incumbent_artwork_id,
        'previous_selected_artwork_id': run.selected_artwork_id,
        'previous_selection_authority': run.selection_authority,
        'previous_user_selection_receipt': deepcopy(run.user_selection_receipt),
        'review_use': 'retained_as_history_and_research_not_current_goal_certification'})
    staged.base_request = updated.model_dump(mode='json')
    staged.candidate_reviews, staged.exploration_branches, staged.exploration_artwork_ids = [], [], []
    staged.incumbent_artwork_id = staged.selected_artwork_id = staged.selection_authority = ''
    staged.user_selection_receipt = None
    staged.consecutive_rejections = 0
    staged.revision += 1
    staged.events.append({'kind': 'explicit_source_reinterpretation', 'receipt': deepcopy(revised.receipt),
                          'scope': 'current_interpretation_only_not_resume_generation_or_quality_verification'})
    # Persist a detached replacement first. Failed writes never change the live
    # request, reviews or dispatch eligibility, nor delete any image/job history.
    controller.store.save(staged)
    for name in type(run).model_fields:
        setattr(run, name, deepcopy(getattr(staged, name)))
    controller.store.runs[run.id] = run
    return run
