from __future__ import annotations

from enum import StrEnum
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from .sampling_profiles import PROFILES, SOURCE_PROFILES
from .creation_intent import CreationIntent
from .regional_conditioning import RegionalTextCondition, validate_regional_request


class TestMode(StrEnum):
    BASELINE = "baseline"
    HANDS = "hands"
    CHARACTER = "character"
    CUSTOM = "custom"


class ModelProfile(StrEnum):
    NEWBIE = "newbie"
    Z_IMAGE_TURBO = "z_image_turbo"
    ANIMA_TURBO = "anima_turbo"
    ILLUSTRIOUS = "illustrious"


class RenderPipeline(StrEnum):
    ADAPTIVE = "adaptive"
    NEWBIE = "newbie"
    Z_IMAGE = "z_image_turbo"
    ANIMA = "anima"
    HYBRID = "anima_newbie"
    ILLUSTRIOUS_NEWBIE = "illustrious_newbie"
    Z_NEWBIE = "z_newbie"
    ILLUSTRIOUS = "illustrious"


PIXEL_BRIDGE_PIPELINES = {
    RenderPipeline.HYBRID,
    RenderPipeline.ILLUSTRIOUS_NEWBIE,
    RenderPipeline.Z_NEWBIE,
}


class SourceStageParameters(BaseModel):
    """Opt-in source override; an absent seed explicitly inherits the job seed."""
    model_config = {'extra': 'forbid'}
    model_id: Literal['anima_turbo', 'illustrious', 'z_image_turbo']
    steps: int = Field(ge=1, le=60)
    cfg: float = Field(ge=1, le=10)
    sampler: str
    scheduler: str
    shift: float = Field(ge=0, le=20)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    lora_name: str | None = None
    lora_strength: float = Field(default=0.75, ge=0, le=2)


class ImageHandoffInput(BaseModel):
    source_file: str = Field(max_length=1000)
    source_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    # Generated works keep their historical ancestry; programme-made material
    # is a separate source, never a fabricated completed job or artwork.
    source_kind: Literal['generated_artwork', 'visual_material'] = 'generated_artwork'
    source_artwork_id: str | None = Field(default=None, min_length=1, max_length=120)
    source_model_id: str = Field(max_length=100)
    source_material_id: str | None = Field(default=None, pattern=r'^[0-9a-f]{32}$')
    source_material_record_file: str | None = Field(default=None, max_length=1000)
    source_material_record_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    source_original_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    denoise: float = Field(gt=0, le=1)
    resize_applied: bool = False
    requested_style_preset: str = Field(default='', max_length=120)
    semantic_verified: bool = False

    @model_validator(mode='after')
    def validate_source_kind(self):
        material_fields = (self.source_material_id, self.source_material_record_file,
                           self.source_material_record_sha256, self.source_original_sha256)
        if self.source_kind == 'generated_artwork':
            if not self.source_artwork_id or any(value is not None for value in material_fields):
                raise ValueError('生成作品来源必须具有真实作品身份，不能混入程序材料来源')
        elif (self.source_artwork_id is not None or not all(material_fields)
              or self.source_model_id != 'programme_sketch' or self.semantic_verified):
            raise ValueError('程序视觉材料必须独立记录，不能伪造生成作品或已验证语义')
        return self


class GenerateRequest(BaseModel):
    creation_intent: CreationIntent | None = None
    author_proposal: dict[str, object] | None = None
    # Optional linkage for the first creation job. Internal research/iteration
    # submissions remain independent of the public preparation submit gate.
    preparation_id: str | None = Field(default=None, pattern=r'^[a-zA-Z0-9_-]{1,80}$')
    preparation_revision: int | None = Field(default=None, ge=1, strict=True)
    iteration_context: dict[str, object] | None = None
    pipeline_version: str = "adaptive"
    # Saved experiments omit this field and keep their historical execution
    # settings. New creation clients explicitly opt into editable parameters.
    parameter_policy: Literal["legacy", "explicit"] = "legacy"
    model_id: ModelProfile = ModelProfile.NEWBIE
    mode: TestMode = TestMode.BASELINE
    prompt: str = Field(min_length=1, max_length=32_000)
    negative_prompt: str = Field(default="", max_length=16_000)
    # Legacy jobs retain their fallback negatives; new explicit editors may
    # intentionally send an empty negative without the builder replacing it.
    use_default_negative_prompt: bool = True
    source_negative_prompt: str | None = Field(default=None, max_length=16_000)
    source_parameters: SourceStageParameters | None = None
    width: int = Field(default=1024, ge=512, le=1536)
    height: int = Field(default=1024, ge=512, le=1536)
    steps: int = Field(default=28, ge=1, le=60)
    cfg: float = Field(default=4.5, ge=1, le=10)
    seed: int = Field(default=42, ge=0, le=2**63 - 1)
    sampler: str = Field(default="res_multistep")
    scheduler: str = Field(default="linear_quadratic")
    shift: float = Field(default=6.0, ge=0, le=20)
    cfg_rescale: float = Field(default=0.9, ge=0, le=1)
    weight_dtype: Literal['default', 'fp8_e4m3fn', 'fp8_e5m2'] | None = None
    lora_name: str | None = None
    lora_strength: float = Field(default=0.75, ge=0, le=2)
    batch_size: int = Field(default=1, ge=1, le=4)
    output_group: str = Field(default="", max_length=500)
    output_label: str = Field(default="", max_length=300)
    semantic_spec: dict[str, object] | None = None
    render_pipeline: RenderPipeline = RenderPipeline.ADAPTIVE
    scene_contract: dict[str, object] | None = None
    native_prompts: dict[str, dict[str, object]] | None = None
    # Opt-in full-canvas masked text predictions; not a spatial/identity proof.
    regional_conditioning: list[RegionalTextCondition] | None = Field(default=None, min_length=1, max_length=4)
    style_preset: str = Field(default="", max_length=120)
    # Hybrid/adaptive graphs can condition composition and character stages
    # independently. None means "inherit style_preset" for saved v2 jobs;
    # an explicit empty string means "model free" for that stage.
    composition_style_preset: str | None = Field(default=None, max_length=120)
    character_style_preset: str | None = Field(default=None, max_length=120)
    anima_prompt: str = Field(default="", max_length=32_000)
    source_prompt: str = Field(default="", max_length=32_000)
    hybrid_denoise: float = Field(default=0.18, ge=0.08, le=0.4)
    highres_enabled: bool = False
    highres_scale: float = Field(default=1.5, ge=1.1, le=2.0)
    highres_steps: int = Field(default=14, ge=5, le=30)
    highres_denoise: float = Field(default=0.22, ge=0.1, le=0.6)
    face_detail_enabled: bool = True
    face_detail_steps: int = Field(default=12, ge=5, le=30)
    face_detail_denoise: float = Field(default=0.2, ge=0.1, le=0.5)
    hand_detail_steps: int = Field(default=16, ge=5, le=30)
    hand_detail_denoise: float = Field(default=0.45, ge=0.1, le=0.6)
    check_enabled: bool = True
    # A hand detector only proves that a hand exists. It is not a defect
    # classifier, so automatic redraw is opt-in until before/after QA exists.
    hand_repair_enabled: bool = False
    # 仅供旧请求兼容和 check 阈值覆盖。新创作由完整 style_preset 驱动生成；
    # check 默认从最终提示词自动识别，不再反向注入截断的“画风标签”。
    style_category: str = "auto"
    # Text removal can erase requested signs and canonical costume lettering.
    # Detection/reporting stays on; destructive cleanup is opt-in.
    text_clean_enabled: bool = False
    text_clean_steps: int = Field(default=14, ge=5, le=30)
    text_clean_denoise: float = Field(default=0.7, ge=0.2, le=0.9)
    parent_artwork_id: str | None = Field(default=None, max_length=120)
    operation: str = Field(default="generate", max_length=80)
    image_handoff: ImageHandoffInput | None = None
    mask_file: str | None = Field(default=None, max_length=1000)
    repair_instruction: str = Field(default="", max_length=4000)
    # Empty keeps historical standalone repairs unchanged. In a creation loop,
    # this is the full current scene used as context, not immutable edit-area constraints.
    repair_context_prompt: str = Field(default="", max_length=32000)
    repair_context_policy: Literal['legacy_local_context', 'scene_context_with_local_override'] = 'legacy_local_context'
    repair_source_file: str | None = Field(default=None, max_length=1000)
    repair_source_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    repair_mask_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    repair_notes: list[str] = Field(default_factory=list)
    region_repair_steps: int = Field(default=18, ge=5, le=40)
    region_repair_denoise: float = Field(default=0.42, ge=0.15, le=0.75)

    @field_validator("width", "height")
    @classmethod
    def multiple_of_16(cls, value: int) -> int:
        if value % 16:
            raise ValueError("必须是 16 的倍数")
        return value

    @field_validator("sampler")
    @classmethod
    def supported_sampler(cls, value: str) -> str:
        allowed = {"res_multistep", "euler_ancestral", "euler", "dpmpp_2m_sde"}
        if value not in allowed:
            raise ValueError(f"采样器仅支持: {', '.join(sorted(allowed))}")
        return value

    @field_validator("scheduler")
    @classmethod
    def supported_scheduler(cls, value: str) -> str:
        if value not in {"linear_quadratic", "simple", "normal", "karras"}:
            raise ValueError("调度器仅支持 linear_quadratic、simple、normal 或 karras")
        return value

    @field_validator("lora_name")
    @classmethod
    def safe_lora_name(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        normalized = value.replace("\\", "/")
        if normalized.startswith("/") or ":" in normalized or ".." in normalized.split("/"):
            raise ValueError("LoRA 必须是模型目录内的相对路径")
        return normalized

    @field_validator("output_group")
    @classmethod
    def safe_output_group(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/").strip("/")
        if not normalized:
            return ""
        if ":" in normalized or ".." in normalized.split("/"):
            raise ValueError("输出目录必须是项目内的相对路径")
        if any(not part.replace("-", "").replace("_", "").replace(".", "").isalnum() for part in normalized.split("/")):
            raise ValueError("输出目录只能使用字母、数字、中文、点、横线与下划线")
        return normalized

    @field_validator("output_label")
    @classmethod
    def safe_output_label(cls, value: str) -> str:
        normalized = value.strip()
        if any(char in normalized for char in '<>:"/\\|?*'):
            raise ValueError("输出名称包含 Windows 不允许的字符")
        return normalized

    @model_validator(mode="after")
    def model_specific_parameters(self) -> "GenerateRequest":
        validate_regional_request(self)
        if (self.preparation_id is None) != (self.preparation_revision is None):
            raise ValueError('准备 ID 与版本必须一起提供')
        if self.preparation_id is not None and not self.author_proposal:
            raise ValueError('关联准备的提交必须携带本版本作者提案')
        if self.author_proposal:
            if not self.creation_intent:
                raise ValueError('作者提案必须与完整原句一起传递')
            if self.author_proposal.get('original_sha256') != self.creation_intent.original_sha256:
                raise ValueError('作者提案与当前原句不一致')
            if self.semantic_spec and self.semantic_spec.get('source_sentence') != self.creation_intent.original_text:
                raise ValueError('当前语义规格不能替换原句')
            if self.scene_contract and self.scene_contract.get('source_text') != self.creation_intent.original_text:
                raise ValueError('当前场景契约不能替换原句')
            if self.author_proposal.get('condition_status') == 'historical_after_native_override':
                if self.semantic_spec is not None or self.scene_contract is not None:
                    raise ValueError('手改原生条件后旧提案只能保留为历史，不能继续声称规格/契约同步')
            else:
                if self.semantic_spec is None or not isinstance(self.author_proposal.get('spec'), dict):
                    raise ValueError('当前作者提案需要完整的当前语义规格')
                def digest(value):
                    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                        separators=(',', ':')).encode('utf-8')).hexdigest()
                expected = self.author_proposal.get('semantic_sha256')
                if expected != digest(self.semantic_spec) or expected != digest(self.author_proposal['spec']):
                    raise ValueError('作者提案与当前语义规格不一致，请一并重建，不能沿用旧caption')
        if self.creation_intent and self.iteration_context:
            if (self.iteration_context.get('original_text') != self.creation_intent.original_text
                    or self.iteration_context.get('original_sha256') != self.creation_intent.original_sha256):
                raise ValueError('下游阶段的原句与任务最初输入不一致')
        if self.image_handoff is not None:
            if (self.parameter_policy != 'explicit' or self.operation != 'image_handoff'
                    or self.render_pipeline.value not in PROFILES
                    or self.parent_artwork_id != self.image_handoff.source_artwork_id
                    or self.batch_size != 1
                    or self.hand_repair_enabled or self.text_clean_enabled or self.repair_source_file):
                raise ValueError('模型交接需要显式单模型和唯一原图，不能混入手部、文字或其他局部修改步骤')
            if self.face_detail_enabled and self.model_id != ModelProfile.NEWBIE:
                raise ValueError('当前目标的交接构建器不能执行已选NewBie面部阶段；未静默关闭设置。')
            if self.image_handoff.source_kind == 'visual_material':
                if (not self.creation_intent
                        or self.creation_intent.original_sha256 != self.image_handoff.source_original_sha256):
                    raise ValueError('程序视觉材料必须与本次完整原句一致')
        expected_model = {
            RenderPipeline.ADAPTIVE: ModelProfile.NEWBIE,
            RenderPipeline.NEWBIE: ModelProfile.NEWBIE,
            RenderPipeline.Z_IMAGE: ModelProfile.Z_IMAGE_TURBO,
            RenderPipeline.ANIMA: ModelProfile.ANIMA_TURBO,
            RenderPipeline.HYBRID: ModelProfile.NEWBIE,
            RenderPipeline.ILLUSTRIOUS_NEWBIE: ModelProfile.NEWBIE,
            RenderPipeline.Z_NEWBIE: ModelProfile.NEWBIE,
            RenderPipeline.ILLUSTRIOUS: ModelProfile.ILLUSTRIOUS,
        }[self.render_pipeline]
        if self.model_id != expected_model:
            # Backward compatibility for saved v1 requests, where model_id was
            # present but render_pipeline did not exist and therefore defaults
            # to NewBie. Normalize those records instead of making old studies
            # impossible to reopen.
            inferred = {
                ModelProfile.Z_IMAGE_TURBO: RenderPipeline.Z_IMAGE,
                ModelProfile.ANIMA_TURBO: RenderPipeline.ANIMA,
                ModelProfile.ILLUSTRIOUS: RenderPipeline.ILLUSTRIOUS,
            }.get(self.model_id)
            if self.render_pipeline in {RenderPipeline.NEWBIE, RenderPipeline.ADAPTIVE} and inferred:
                self.render_pipeline = inferred
            else:
                raise ValueError(f"管线 {self.render_pipeline.value} 必须使用模型标识 {expected_model.value}")
        if self.render_pipeline in PIXEL_BRIDGE_PIPELINES and not (self.source_prompt.strip() or self.anima_prompt.strip()):
            raise ValueError("协作管线需要源模型场景提示词")
        if self.model_id == ModelProfile.NEWBIE:
            if self.parameter_policy == 'legacy' and self.scheduler != "linear_quadratic":
                raise ValueError("NewBie 基线必须使用 linear_quadratic")
            if self.parameter_policy == 'legacy' and self.steps < 10:
                raise ValueError("NewBie 至少需要 10 步")
        elif self.model_id in {ModelProfile.Z_IMAGE_TURBO, ModelProfile.ANIMA_TURBO}:
            if self.parameter_policy == 'legacy' and (self.sampler != "euler" or self.scheduler != "simple"):
                raise ValueError("Turbo 基线必须使用 euler + simple")
            if self.lora_name:
                raise ValueError("当前 Turbo 对照基线尚未启用 LoRA")
        elif self.model_id == ModelProfile.ILLUSTRIOUS:
            if self.parameter_policy == 'legacy' and (self.sampler not in {"euler_ancestral", "dpmpp_2m_sde"} or self.scheduler not in {"normal", "karras"}):
                raise ValueError("Illustrious 基线必须使用 euler_ancestral/normal 或 dpmpp_2m_sde/karras")
            if self.parameter_policy == 'legacy' and self.steps < 20:
                raise ValueError("Illustrious 至少需要 20 步")
            if self.weight_dtype is not None:
                raise ValueError('当前 CheckpointLoader 不支持覆盖权重精度；请将 weight_dtype 设为 null，不能静默忽略')
        if self.source_parameters is not None:
            if self.parameter_policy != 'explicit' or self.render_pipeline not in PIXEL_BRIDGE_PIPELINES:
                raise ValueError('源阶段参数仅适用于明确选择的协作路线与 explicit 参数模式')
            profile = SOURCE_PROFILES[self.render_pipeline.value]
            source = self.source_parameters
            if source.model_id != PROFILES[profile]['model_id']:
                raise ValueError('源阶段模型与所选协作路线不一致')
            # Reuse actual single-model validation, including safe relative LoRA
            # paths and unsupported Turbo adapters. No recursive source override.
            validated = GenerateRequest(
                **source.model_dump(exclude={'seed'}), seed=source.seed if source.seed is not None else self.seed,
                parameter_policy='explicit', render_pipeline=profile, prompt=self.source_prompt or self.anima_prompt,
            )
            source.lora_name = validated.lora_name
        return self


class MatrixRequest(BaseModel):
    base: GenerateRequest
    seeds: list[int] = Field(default_factory=lambda: [42])
    variants: list[str] = Field(
        default_factory=lambda: ["identity", "hands", "scene", "style"]
    )

    @field_validator("seeds")
    @classmethod
    def seed_limit(cls, value: list[int]) -> list[int]:
        if not value or len(value) > 8:
            raise ValueError("种子数量必须为 1 到 8")
        return value


class DraftPromptRequest(BaseModel):
    model_id: ModelProfile = ModelProfile.NEWBIE
    description: str = Field(min_length=3, max_length=8000)
    character: str = Field(default="", max_length=500)
    style: str = Field(default="anime_style, game_illustration, clean_lineart", max_length=1000)
    visible_hands: bool = True


class NewbieStyleMode(StrEnum):
    STANDARD = "standard_anime"
    CHIBI = "chibi"
    POSTER = "poster"


class NewbieObjectSpec(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=160)
    appearance: list[str] = Field(default_factory=list, max_length=12)
    position: list[str] = Field(default_factory=list, max_length=8)


class NewbieRelationSpec(BaseModel):
    subject: str = Field(min_length=1, max_length=120)
    predicate: str = Field(min_length=1, max_length=180)
    object: str = Field(min_length=1, max_length=120)


class NewbieSemanticSpec(BaseModel):
    source_sentence: str = Field(min_length=1, max_length=4000)
    character_name: str = Field(default="original character", max_length=160)
    character_tag: str = Field(default="original_character", max_length=200)
    source: str = Field(default="", max_length=200)
    count: str = Field(default="1girl, solo", max_length=120)
    appearance: list[str] = Field(default_factory=list, max_length=30)
    clothing: list[str] = Field(default_factory=list, max_length=30)
    expression: list[str] = Field(default_factory=list, max_length=16)
    body_action: list[str] = Field(default_factory=list, max_length=20)
    left_hand: list[str] = Field(default_factory=list, max_length=16)
    right_hand: list[str] = Field(default_factory=list, max_length=16)
    objects: list[NewbieObjectSpec] = Field(default_factory=list, max_length=8)
    relations: list[NewbieRelationSpec] = Field(default_factory=list, max_length=12)
    framing: list[str] = Field(default_factory=list, max_length=12)
    background: list[str] = Field(default_factory=list, max_length=20)
    lighting: list[str] = Field(default_factory=list, max_length=12)
    style_mode: NewbieStyleMode = NewbieStyleMode.STANDARD
    style: list[str] = Field(default_factory=list, max_length=16)
    exclusions: list[str] = Field(default_factory=list, max_length=30)


class NewbieCompileRequest(BaseModel):
    sentence: str = Field(min_length=2, max_length=4000)
    style_mode: NewbieStyleMode = NewbieStyleMode.STANDARD
    knowledge_pack: str = Field(default="", max_length=80)
    knowledge_operator: str = Field(default="", max_length=80)
    knowledge_appearance: str = Field(default="", max_length=80)

    @model_validator(mode="after")
    def complete_knowledge_selection(self) -> "NewbieCompileRequest":
        values = (self.knowledge_pack, self.knowledge_operator, self.knowledge_appearance)
        if any(values) and not all(values):
            raise ValueError("知识库、角色和外观必须同时选择")
        return self


class NewbieRecompileRequest(BaseModel):
    spec: NewbieSemanticSpec
    intent: NewbieCompileRequest | None = None
    parsed_character_tag: str = Field(default='', max_length=500)


class JobResponse(BaseModel):
    id: str
    status: str
    created_at: str
    request: dict
    progress: float = 0
    stage: str = "queued"
    images: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    checks: list[dict] = Field(default_factory=list)
    error: str | None = None
    engine_dispatches: list[dict] = Field(default_factory=list)
    cancel_requested: bool = False
    cancel_error: str | None = None


class ExperimentPixelSource(BaseModel):
    model_config = {'extra': 'forbid'}
    source_key: str = Field(min_length=1, max_length=120)
    denoise: float = Field(gt=0, le=1)


class ExperimentItemCreate(BaseModel):
    key: str | None = Field(default=None, max_length=120)
    pixel_source: ExperimentPixelSource | None = None
    axes: dict[str, str | int | float | bool] = Field(default_factory=dict)
    request: GenerateRequest


class ExperimentCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    items: list[ExperimentItemCreate] = Field(min_length=1, max_length=1000)
    notes: str = Field(default="", max_length=4000)
    external_llm_used: bool = False

    @model_validator(mode='after')
    def ordered_pixel_dependencies(self):
        seen = set()
        for item in self.items:
            if item.key and item.key in seen:
                raise ValueError('实验项 key 不得重复')
            if item.pixel_source:
                if item.pixel_source.source_key not in seen:
                    raise ValueError('像素交接源必须是本实验中排在前面的具名项')
                request = item.request
                if request.render_pipeline.value not in {'newbie', 'anima', 'illustrious', 'z_image_turbo'} or request.parameter_policy != 'explicit':
                    raise ValueError('实验交接目标必须是显式单模型')
                if request.batch_size != 1 or request.highres_enabled or request.face_detail_enabled or request.hand_repair_enabled or request.text_clean_enabled or request.image_handoff or request.repair_source_file:
                    raise ValueError('实验交接一次只执行一个基础采样阶段')
            if item.key:
                seen.add(item.key)
        return self


class ExperimentReviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    image_index: int = Field(default=0, ge=0, le=3)


class ArtworkReviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    image_index: int = Field(default=0, ge=0, le=3)
    origin: Literal['user', 'assistant_validation'] = 'user'


class CharacterDesignSpec(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=120)
    source: str = Field(default="", max_length=200)
    name_tag: str = Field(min_length=1, max_length=200)
    appearance: str = Field(min_length=1, max_length=1000)
    clothing: str = Field(min_length=1, max_length=1000)
    signature_action: str = Field(min_length=1, max_length=1000)
    confirmed: bool = False


class SceneDesignSpec(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=3, max_length=2000)
    confirmed: bool = False


class ExperimentDesignRequest(BaseModel):
    name: str = Field(default="模型对比：NewBie vs Anima Turbo（5角色×6场景）", min_length=1, max_length=200)
    characters: list[CharacterDesignSpec] = Field(min_length=1, max_length=10)
    scenes: list[SceneDesignSpec] = Field(min_length=1, max_length=12)
    seeds: list[int] = Field(default_factory=lambda: [42, 123456, 987654321], min_length=1, max_length=8)
    width: int = Field(default=1024, ge=512, le=1536)
    height: int = Field(default=1024, ge=512, le=1536)

    @field_validator("width", "height")
    @classmethod
    def design_multiple_of_16(cls, value: int) -> int:
        if value % 16:
            raise ValueError("必须是 16 的倍数")
        return value

    @field_validator("seeds")
    @classmethod
    def unique_seeds(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("Seed 不得重复")
        if any(seed < 0 or seed > 2**63 - 1 for seed in value):
            raise ValueError("Seed 超出范围")
        return value
