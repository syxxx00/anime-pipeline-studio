from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .scene_contract import SceneContractV3, TextIntentMode


class NodeType(StrEnum):
    CONTRACT_GATE = "contract_gate"
    CHECKPOINT = "checkpoint"
    GENERATE = "generate"
    FUSE = "fuse"
    COMPARE = "compare"
    CHECK = "check"
    ROUTE = "route"
    LOCAL_REPAIR = "local_repair"
    TEXT_RENDER = "text_render"
    TEXT_VERIFY = "text_verify"
    HUMAN_GATE = "human_gate"
    ROLLBACK = "rollback"
    ACCEPT = "accept"


class ModelRole(StrEnum):
    NEWBIE = "newbie"
    ILLUSTRIOUS = "illustrious"
    ANIMA = "anima"
    Z_IMAGE = "z_image"
    DETERMINISTIC = "deterministic"
    HUMAN = "human"
    SYSTEM = "system"


class NodeScope(StrEnum):
    CONTRACT = "contract"
    FULL_IMAGE = "full_image"
    REGION = "region"
    TEXT_LAYER = "text_layer"
    METADATA = "metadata"


class GraphNode(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]*$", max_length=120)
    type: NodeType
    label: str = Field(min_length=1, max_length=200)
    role: ModelRole = ModelRole.SYSTEM
    scope: NodeScope = NodeScope.METADATA
    purpose: str = Field(default="", max_length=1000)
    inputs: list[str] = Field(default_factory=list, max_length=30)
    outputs: list[str] = Field(default_factory=list, max_length=30)
    preserve: list[str] = Field(default_factory=list, max_length=80)
    checks: list[str] = Field(default_factory=list, max_length=50)
    acceptance: list[str] = Field(default_factory=list, max_length=50)
    config: dict[str, Any] = Field(default_factory=dict)
    max_visits: int | None = Field(default=None, ge=1, le=20)


class GraphEdge(BaseModel):
    source: str
    target: str
    condition: str = Field(default="always", min_length=1, max_length=500)
    priority: int = Field(default=0, ge=-100, le=100)


class PipelineGraphV3(BaseModel):
    version: Literal["3.0"] = "3.0"
    id: str = "adaptive_anime_v3"
    entry_node: str
    nodes: list[GraphNode] = Field(min_length=1, max_length=100)
    edges: list[GraphEdge] = Field(default_factory=list, max_length=300)
    terminal_nodes: list[str] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_graph(self) -> "PipelineGraphV3":
        node_ids = [node.id for node in self.nodes]
        known = set(node_ids)
        if len(node_ids) != len(known):
            raise ValueError("pipeline graph node ids must be unique")
        if self.entry_node not in known:
            raise ValueError("entry node is missing")
        if not set(self.terminal_nodes) <= known:
            raise ValueError("terminal node is missing")
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(f"edge references unknown node: {edge.source} -> {edge.target}")

        adjacency: dict[str, list[str]] = {node_id: [] for node_id in known}
        for edge in self.edges:
            adjacency[edge.source].append(edge.target)
        cycle_nodes: set[str] = set()
        visiting: list[str] = []
        visited: set[str] = set()

        def walk(node_id: str) -> None:
            if node_id in visiting:
                cycle_nodes.update(visiting[visiting.index(node_id) :])
                return
            if node_id in visited:
                return
            visiting.append(node_id)
            for target in adjacency[node_id]:
                walk(target)
            visiting.pop()
            visited.add(node_id)

        walk(self.entry_node)
        limits = {node.id: node.max_visits for node in self.nodes}
        unbounded = sorted(node_id for node_id in cycle_nodes if limits[node_id] is None)
        if unbounded:
            raise ValueError("cycle nodes require max_visits: " + ", ".join(unbounded))
        return self


class ArtifactRef(BaseModel):
    id: str
    kind: str
    path: str = ""
    producer_node: str
    parent_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Defect(BaseModel):
    id: str
    category: str
    description: str
    severity: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    region: list[float] | None = Field(default=None, min_length=4, max_length=4)
    evidence: list[str] = Field(default_factory=list)
    suggested_role: ModelRole | None = None
    requires_human: bool = False
    resolved: bool = False


class CheckRecord(BaseModel):
    node_id: str
    artifact_id: str
    passed: bool
    scores: dict[str, float] = Field(default_factory=dict)
    defects: list[Defect] = Field(default_factory=list)


class StateCheckpoint(BaseModel):
    id: str
    node_id: str
    artifact_ids: list[str]
    locked_refs: list[str] = Field(default_factory=list)
    note: str = ""


class SceneState(BaseModel):
    version: Literal["3.0"] = "3.0"
    contract: SceneContractV3
    current_node: str
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    checks: list[CheckRecord] = Field(default_factory=list)
    open_defects: list[Defect] = Field(default_factory=list)
    locked_refs: list[str] = Field(default_factory=list)
    checkpoints: list[StateCheckpoint] = Field(default_factory=list)
    visit_counts: dict[str, int] = Field(default_factory=dict)
    history: list[str] = Field(default_factory=list)
    status: Literal["planned", "running", "needs_user", "accepted", "failed", "cancelled"] = "planned"


CHECK_DIMENSIONS = [
    "character_identity",
    "costume_and_skin",
    "anatomy_and_hands",
    "object_presence",
    "object_relations",
    "background_and_composition",
    "style_match",
    "technical_quality",
    "text_intent",
]


def repair_role_for(category: str) -> ModelRole:
    routing = {
        "character_identity": ModelRole.NEWBIE,
        "costume_and_skin": ModelRole.NEWBIE,
        "anatomy_and_hands": ModelRole.NEWBIE,
        "object_presence": ModelRole.ILLUSTRIOUS,
        "object_relations": ModelRole.ILLUSTRIOUS,
        "background_and_composition": ModelRole.ANIMA,
        "style_match": ModelRole.ILLUSTRIOUS,
        "technical_quality": ModelRole.NEWBIE,
        "text_intent": ModelRole.DETERMINISTIC,
    }
    return routing.get(category, ModelRole.HUMAN)


def build_adaptive_graph(contract: SceneContractV3) -> PipelineGraphV3:
    """Build a bounded macro graph. ComfyUI workflows are implementations of its nodes."""
    preserve = [
        constraint.id
        for entity in contract.entities
        for constraint in entity.constraints
        if constraint.preserve_across_stages
    ] + [relation.id for relation in contract.relations if relation.preserve_across_stages]

    nodes = [
        GraphNode(id="contract_gate", type=NodeType.CONTRACT_GATE, label="语义契约检查", scope=NodeScope.CONTRACT, checks=["generation_ready", "reference_integrity", "text_mode_valid"]),
        GraphNode(id="baseline", type=NodeType.CHECKPOINT, label="建立初始检查点", outputs=["checkpoint"], preserve=preserve),
        GraphNode(id="candidate_newbie", type=NodeType.GENERATE, label="NewBie 整图候选", role=ModelRole.NEWBIE, scope=NodeScope.FULL_IMAGE, purpose="使用 NewBie 原生条件生成完整画面，适用性由同条件对比判断", outputs=["candidate"]),
        GraphNode(id="candidate_anima", type=NodeType.GENERATE, label="Anima 整图候选", role=ModelRole.ANIMA, scope=NodeScope.FULL_IMAGE, purpose="使用 Anima 原生条件生成完整画面，不预设其只负责环境", outputs=["candidate"]),
        GraphNode(id="candidate_ill", type=NodeType.GENERATE, label="Illustrious 整图候选", role=ModelRole.ILLUSTRIOUS, scope=NodeScope.FULL_IMAGE, purpose="使用 Illustrious 原生条件生成完整画面，不预设其只负责道具或姿态", outputs=["candidate"]),
        GraphNode(id="fusion_ill_newbie", type=NodeType.FUSE, label="Illustrious → NewBie 混合候选", role=ModelRole.NEWBIE, scope=NodeScope.FULL_IMAGE, purpose="复用已有 Illustrious 候选像素，再由 NewBie 重采样；可能改善也可能损坏原有特点", inputs=["candidate_ill"], outputs=["candidate"], preserve=preserve, config={"bridge": "pixel", "denoise": "request.hybrid_denoise"}),
        GraphNode(id="compare", type=NodeType.COMPARE, label="候选对比与保留", inputs=["candidate_newbie", "candidate_anima", "candidate_ill", "fusion_ill_newbie"], outputs=["selected", "alternates"], checks=CHECK_DIMENSIONS, acceptance=["no_single_opaque_score", "keep_component_winners"]),
        GraphNode(id="selected_checkpoint", type=NodeType.CHECKPOINT, label="锁定候选的优秀部分", inputs=["selected"], outputs=["checkpoint"], preserve=preserve),
        GraphNode(id="component_check", type=NodeType.CHECK, label="分组件检查", inputs=["selected"], outputs=["check_report", "defects"], checks=CHECK_DIMENSIONS, acceptance=["required_relations_present", "no_high_severity_defect"], max_visits=4),
        GraphNode(id="defect_router", type=NodeType.ROUTE, label="检查缺陷与可用修改方法", inputs=["defects"], outputs=["repair_plan"], config={**{dimension: repair_role_for(dimension).value for dimension in CHECK_DIMENSIONS}, "routing_status":"unvalidated_heuristic", "implemented_inpaint":"newbie_only"}, max_visits=4),
        GraphNode(id="region_repair", type=NodeType.LOCAL_REPAIR, label="局部定向修复", scope=NodeScope.REGION, purpose="只修改有证据的缺陷区域，保护已锁定部分", inputs=["repair_plan", "selected"], outputs=["repaired"], preserve=preserve, acceptance=["target_defect_improved", "locked_regions_not_regressed"], config={"full_image_fallback": "human_approval_only", "rollback_on_regression": True}, max_visits=3),
        GraphNode(id="rollback", type=NodeType.ROLLBACK, label="退回上一检查点", inputs=["checkpoint"], outputs=["selected"], purpose="修复造成退化或达到循环上限时恢复", max_visits=3),
        GraphNode(id="human_gate", type=NodeType.HUMAN_GATE, label="人工确认不确定项", role=ModelRole.HUMAN, purpose="低置信检查、审美选择或整图重构前确认", inputs=["selected", "check_report"], outputs=["user_decision"], max_visits=2),
    ]
    edges = [
        GraphEdge(source="contract_gate", target="baseline", condition="contract.generation_ready"),
        GraphEdge(source="baseline", target="candidate_newbie"),
        GraphEdge(source="baseline", target="candidate_anima"),
        GraphEdge(source="baseline", target="candidate_ill"),
        GraphEdge(source="candidate_ill", target="fusion_ill_newbie"),
        GraphEdge(source="candidate_newbie", target="compare"),
        GraphEdge(source="candidate_anima", target="compare"),
        GraphEdge(source="candidate_ill", target="compare"),
        GraphEdge(source="fusion_ill_newbie", target="compare"),
        GraphEdge(source="compare", target="selected_checkpoint"),
        GraphEdge(source="selected_checkpoint", target="component_check"),
        GraphEdge(source="component_check", target="defect_router", condition="high_confidence_repairable_defect", priority=20),
        GraphEdge(source="component_check", target="human_gate", condition="low_confidence_or_aesthetic_decision", priority=10),
        GraphEdge(source="defect_router", target="region_repair"),
        GraphEdge(source="region_repair", target="component_check", condition="target_improved_and_visits_remaining", priority=20),
        GraphEdge(source="region_repair", target="rollback", condition="regression_or_no_improvement", priority=30),
        GraphEdge(source="rollback", target="human_gate", condition="repair_budget_exhausted"),
        GraphEdge(source="human_gate", target="defect_router", condition="user_requests_targeted_repair", priority=20),
    ]

    current_tail = "component_check"
    exact_text = any(intent.mode == TextIntentMode.EXACT for intent in contract.text_intents)
    if exact_text:
        nodes.extend([
            GraphNode(id="text_render", type=NodeType.TEXT_RENDER, label="确定性文字图层", role=ModelRole.DETERMINISTIC, scope=NodeScope.TEXT_LAYER, purpose="按契约内容排版，不让扩散模型拼写", outputs=["text_composite"], max_visits=3),
            GraphNode(id="text_verify", type=NodeType.TEXT_VERIFY, label="OCR 与像素一致性检查", role=ModelRole.DETERMINISTIC, scope=NodeScope.TEXT_LAYER, checks=["exact_content", "placement", "legibility"], acceptance=["exact_text_matches"], max_visits=3),
        ])
        edges.extend([
            GraphEdge(source="component_check", target="text_render", condition="visual_checks_pass", priority=30),
            GraphEdge(source="text_render", target="text_verify"),
            GraphEdge(source="text_verify", target="text_render", condition="text_mismatch_and_visits_remaining"),
        ])
        current_tail = "text_verify"

    nodes.append(GraphNode(id="accept", type=NodeType.ACCEPT, label="交付与可追溯记录", inputs=["selected", "check_report"], outputs=["final", "provenance"] ))
    edges.append(GraphEdge(source=current_tail, target="accept", condition="all_required_checks_pass", priority=40))
    edges.append(GraphEdge(source="human_gate", target="accept", condition="user_accepts", priority=40))
    return PipelineGraphV3(entry_node="contract_gate", nodes=nodes, edges=edges, terminal_nodes=["accept"])


def initial_scene_state(contract: SceneContractV3, graph: PipelineGraphV3 | None = None) -> SceneState:
    graph = graph or build_adaptive_graph(contract)
    return SceneState(contract=contract, current_node=graph.entry_node, history=["graph_planned"])
