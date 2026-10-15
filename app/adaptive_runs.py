from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .config import PROJECT_ROOT
from .pipeline_graph import ArtifactRef, CheckRecord, PipelineGraphV3, SceneState, build_adaptive_graph, initial_scene_state
from .pipeline_runtime import RepairPlan, build_repair_plan
from .scene_contract import SceneContractV3


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class CandidateSlot(BaseModel):
    node_id: str
    status: Literal["pending", "completed", "failed", "skipped"] = "pending"
    artifact: ArtifactRef | None = None
    error: str = ""


class AdaptiveRun(BaseModel):
    id: str
    created_at: str
    updated_at: str
    graph: PipelineGraphV3
    state: SceneState
    candidates: dict[str, CandidateSlot]
    pending_nodes: list[str] = Field(default_factory=list)
    selected_artifact_id: str = ""
    selected_checkpoint_artifact_id: str = ""
    repair_plan: RepairPlan | None = None
    repair_attempts: int = 0
    max_repair_attempts: int = 3
    base_request: dict = Field(default_factory=dict)
    candidate_job_ids: dict[str, str] = Field(default_factory=dict)
    repair_job_id: str = ""
    repair_action_id: str = ""
    repair_origin: Literal["", "automatic", "user_mask"] = ""
    repair_input_files: list[str] = Field(default_factory=list)
    fulfilled_text_intent_ids: list[str] = Field(default_factory=list)
    pause_reason: Literal["", "user", "service_restart"] = ""


CANDIDATE_NODES = ("candidate_newbie", "candidate_anima", "candidate_ill", "fusion_ill_newbie")


class AdaptiveRunStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or PROJECT_ROOT / "data" / "adaptive_runs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs: dict[str, AdaptiveRun] = {}
        self._load()

    def create(self, contract: SceneContractV3) -> AdaptiveRun:
        graph = build_adaptive_graph(contract)
        state = initial_scene_state(contract, graph)
        run_id = uuid.uuid4().hex[:12]
        if contract.generation_ready:
            state.status = "running"
            state.current_node = "baseline"
            pending = ["candidate_newbie", "candidate_anima", "candidate_ill"]
            state.history.extend(["contract_gate:pass", "baseline:checkpoint"])
        else:
            state.status = "needs_user"
            pending = ["contract_gate"]
            state.history.append("contract_gate:blocked")
        run = AdaptiveRun(
            id=run_id, created_at=now_iso(), updated_at=now_iso(), graph=graph, state=state,
            candidates={node_id: CandidateSlot(node_id=node_id, status="skipped" if node_id == "fusion_ill_newbie" else "pending") for node_id in CANDIDATE_NODES},
            pending_nodes=pending,
        )
        self.runs[run.id] = run
        self._save(run)
        return run

    def get(self, run_id: str) -> AdaptiveRun | None:
        return self.runs.get(run_id)

    def list(self) -> list[AdaptiveRun]:
        return sorted(self.runs.values(), key=lambda item: item.updated_at, reverse=True)

    def record_candidate(self, run: AdaptiveRun, node_id: str, artifact: ArtifactRef) -> AdaptiveRun:
        if run.state.status != "running" or run.pause_reason:
            raise ValueError("当前任务不可接收候选结果")
        if node_id not in CANDIDATE_NODES:
            raise ValueError("不是候选节点")
        slot = run.candidates[node_id]
        if slot.status not in {"pending", "skipped"}:
            raise ValueError("候选节点已经结束")
        slot.status = "completed"
        slot.artifact = artifact
        run.state.artifacts.append(artifact)
        run.state.history.append(f"{node_id}:completed:{artifact.id}")
        self._settle_candidate_node(run, node_id)
        return self._touch(run)

    def fail_candidate(self, run: AdaptiveRun, node_id: str, error: str) -> AdaptiveRun:
        if run.state.status != "running" or run.pause_reason:
            raise ValueError("当前任务不可更新候选状态")
        if node_id not in CANDIDATE_NODES:
            raise ValueError("不是候选节点")
        slot = run.candidates[node_id]
        slot.status = "failed"
        slot.error = error[:2000]
        run.state.history.append(f"{node_id}:failed")
        self._settle_candidate_node(run, node_id)
        return self._touch(run)

    def _settle_candidate_node(self, run: AdaptiveRun, node_id: str) -> None:
        run.pending_nodes = [item for item in run.pending_nodes if item != node_id]
        if node_id == "candidate_ill":
            fusion = run.candidates["fusion_ill_newbie"]
            if run.candidates[node_id].status == "completed":
                fusion.status = "pending"
                if "fusion_ill_newbie" not in run.pending_nodes:
                    run.pending_nodes.append("fusion_ill_newbie")
            else:
                fusion.status = "skipped"
                fusion.error = "Illustrious source candidate unavailable"
        terminal = {"completed", "failed", "skipped"}
        if all(slot.status in terminal for slot in run.candidates.values()) and not run.pending_nodes:
            if not any(slot.status == "completed" for slot in run.candidates.values()):
                run.state.status = "failed"
                run.state.current_node = "compare"
                run.state.history.append("compare:no_candidates")
            else:
                run.pending_nodes = ["compare"]
                run.state.current_node = "compare"
                run.state.status = "needs_user"
                run.state.history.append("compare:needs_selection")

    def select_candidate(self, run: AdaptiveRun, artifact_id: str, locked_refs: list[str]) -> AdaptiveRun:
        self._require_user_boundary(run)
        candidates = {slot.artifact.id: slot.artifact for slot in run.candidates.values() if slot.artifact is not None}
        if artifact_id not in candidates:
            raise ValueError("候选作品不存在")
        run.selected_artifact_id = artifact_id
        run.selected_checkpoint_artifact_id = artifact_id
        # Text rendered on a different derivative is not present in this candidate.
        run.fulfilled_text_intent_ids = []
        run.state.locked_refs = list(dict.fromkeys(locked_refs))
        run.state.current_node = "component_check"
        run.state.status = "running"
        run.pending_nodes = ["component_check"]
        run.state.history.extend([f"compare:selected:{artifact_id}", "selected_checkpoint:saved"])
        return self._touch(run)

    def record_check(self, run: AdaptiveRun, record: CheckRecord) -> AdaptiveRun:
        if run.state.status in {"accepted", "cancelled", "failed"} or run.pause_reason or run.repair_job_id:
            raise ValueError("当前任务不可更新检查；请先完成或继续正在进行的操作")
        if record.artifact_id != run.selected_artifact_id:
            raise ValueError("check 结果不属于当前选中作品")
        run.state.checks.append(record)
        run.state.open_defects = [defect for defect in record.defects if not defect.resolved]
        run.pending_nodes = []
        if record.passed and not run.state.open_defects:
            tail = "text_render" if self.missing_exact_text_intents(run) else "accept"
            run.pending_nodes = [tail]
            run.state.current_node = tail
            # Exact text currently needs explicit placement through the text
            # API; the worker does not dispatch a lettering stage by itself.
            run.state.status = "needs_user"
            run.state.history.append("component_check:pass")
        else:
            run.repair_plan = build_repair_plan(run.state.open_defects, locked_refs=run.state.locked_refs)
            auto = any(action.auto_runnable for action in run.repair_plan.actions)
            run.pending_nodes = ["region_repair"] if auto and run.repair_attempts < run.max_repair_attempts else ["human_gate"]
            run.state.current_node = run.pending_nodes[0]
            run.state.status = "running" if run.pending_nodes[0] == "region_repair" else "needs_user"
            run.state.history.append(f"component_check:defects:{len(run.state.open_defects)}")
        return self._touch(run)

    def record_repair(self, run: AdaptiveRun, artifact: ArtifactRef | None, *, improved: bool) -> AdaptiveRun:
        self._require_executing(run)
        run.repair_attempts += 1
        run.repair_job_id = ""
        run.repair_action_id = ""
        run.repair_origin = ""
        run.repair_input_files = []
        if improved and artifact is not None:
            run.state.artifacts.append(artifact)
            run.selected_artifact_id = artifact.id
            run.pending_nodes = ["component_check"]
            run.state.current_node = "component_check"
            run.state.status = "running"
            run.state.history.append(f"region_repair:improved:{artifact.id}")
        else:
            if artifact is not None:
                # Rejection changes the selected version, not the existence of
                # evidence. Keep failed derivatives available for comparison.
                run.state.artifacts.append(artifact)
                run.state.history.append(f'region_repair:rejected_candidate_retained:{artifact.id}')
            run.selected_artifact_id = run.selected_checkpoint_artifact_id
            run.pending_nodes = ["human_gate"]
            run.state.current_node = "human_gate"
            run.state.status = "needs_user"
            run.state.history.extend(["region_repair:regression", "rollback:checkpoint_restored"])
        return self._touch(run)

    def complete_repair(self, run: AdaptiveRun, artifact: ArtifactRef, record: CheckRecord, *, improved: bool) -> AdaptiveRun:
        """Finish one bounded iteration and immediately route its recheck."""
        self._require_executing(run)
        if record.artifact_id != artifact.id or artifact.parent_ids != [run.selected_artifact_id]:
            raise ValueError('修复候选与当前源图或检查记录不一致')
        if not improved:
            run.state.checks.append(record)
            return self.record_repair(run, artifact, improved=False)
        origin = run.repair_origin
        self.record_repair(run, artifact, improved=True)
        if origin == "automatic":
            run.selected_checkpoint_artifact_id = artifact.id
        return self.record_check(run, record)

    def record_user_repair_candidate(self, run: AdaptiveRun, artifact: ArtifactRef, record: CheckRecord) -> AdaptiveRun:
        """A completed user-mask edit is a candidate, not proof of improvement."""
        self._require_executing(run)
        if record.artifact_id != artifact.id or artifact.parent_ids != [run.selected_artifact_id]:
            raise ValueError('局部修改结果与当前作品或检查记录不一致')
        run.repair_attempts += 1
        run.repair_job_id = run.repair_action_id = run.repair_origin = ''
        run.repair_input_files = []
        run.state.artifacts.append(artifact)
        run.state.checks.append(record)
        run.state.open_defects = [item for item in record.defects if not item.resolved]
        run.selected_artifact_id = artifact.id
        # Keep the previous checkpoint available for explicit visual comparison.
        run.pending_nodes = ['human_gate']
        run.state.current_node = 'human_gate'
        run.state.status = 'needs_user'
        run.state.history.append(f'region_repair:awaiting_user_comparison:{artifact.id}')
        return self._touch(run)

    def block_repair(self, run: AdaptiveRun, reason: str) -> AdaptiveRun:
        self._require_executing(run)
        run.pending_nodes = ["human_gate"]
        run.state.current_node = "human_gate"
        run.state.status = "needs_user"
        run.state.history.append(f"region_repair:blocked:{reason[:500]}")
        return self._touch(run)

    def attach_repair_job(
        self, run: AdaptiveRun, job_id: str, *, action_id: str = "",
        origin: Literal["automatic", "user_mask"], input_files: list[str] | None = None,
    ) -> AdaptiveRun:
        if run.repair_job_id:
            raise ValueError("已有局部修改正在运行")
        if run.repair_attempts >= run.max_repair_attempts:
            raise ValueError("已达到局部修改次数上限")
        run.repair_job_id = job_id
        run.repair_action_id = action_id
        run.repair_origin = origin
        run.repair_input_files = list(input_files or [])
        run.pending_nodes = ["region_repair"]
        run.state.current_node = "region_repair"
        if run.state.status == "cancelled":
            # A submission may finish after cancellation. Retain ownership so
            # the orchestrator can cancel that job without resurrecting the run.
            run.pending_nodes = []
        elif not run.pause_reason:
            run.state.status = "running"
        run.state.history.append(f"region_repair:{origin}:job:{job_id}")
        return self._touch(run)

    def rollback(self, run: AdaptiveRun) -> AdaptiveRun:
        self._require_user_boundary(run)
        if not run.selected_checkpoint_artifact_id:
            raise ValueError("没有可回退的检查点")
        run.selected_artifact_id = run.selected_checkpoint_artifact_id
        run.fulfilled_text_intent_ids = []
        run.pending_nodes = ["human_gate"]
        run.state.current_node = "human_gate"
        run.state.status = "needs_user"
        run.state.history.append("rollback:checkpoint_restored_by_user")
        return self._touch(run)

    def record_review(self, run: AdaptiveRun, text: str, record: CheckRecord | None) -> AdaptiveRun:
        if record is not None:
            self._require_user_boundary(run)
            if record.artifact_id != run.selected_artifact_id:
                raise ValueError("评价对象不是当前选中版本")
        run.state.history.append(f"review:{text.strip()[:240]}")
        if record is None:
            return self._touch(run)
        return self.record_check(run, record)

    def accept(self, run: AdaptiveRun) -> AdaptiveRun:
        self._require_user_boundary(run)
        if not run.selected_artifact_id:
            raise ValueError("没有可接受的作品")
        if run.pending_nodes not in (["accept"], ["human_gate"]):
            raise ValueError("必须先完成候选检查；只有检查通过或人工门控状态可以接受")
        missing_text = self.missing_exact_text_intents(run)
        if missing_text:
            run.pending_nodes = ["text_render"]
            run.state.current_node = "text_render"
            run.state.status = "needs_user"
            run.state.history.append("human_gate:visual_accepted:text_pending")
            return self._touch(run)
        run.pending_nodes = []
        run.state.current_node = "accept"
        run.state.status = "accepted"
        run.state.history.append(f"accept:{run.selected_artifact_id}")
        return self._touch(run)

    @staticmethod
    def missing_exact_text_intents(run: AdaptiveRun):
        fulfilled = set(run.fulfilled_text_intent_ids)
        return [
            intent for intent in run.state.contract.text_intents
            if intent.mode.value == "exact" and intent.id not in fulfilled
        ]

    def record_text_artifact(
        self, run: AdaptiveRun, artifact: ArtifactRef, *, intent_id: str,
        content: str, verification: str,
    ) -> AdaptiveRun:
        if run.state.status in {"accepted", "cancelled", "failed"} or run.pause_reason or run.repair_job_id:
            raise ValueError("当前任务不可添加文字版本")
        missing = {intent.id: intent for intent in self.missing_exact_text_intents(run)}
        intent = missing.get(intent_id)
        if intent is None:
            raise ValueError("文字要求不存在或已经完成")
        if content != intent.content:
            raise ValueError("文字内容必须与语义契约逐字一致")
        if verification != "deterministic_source_and_pixel_invariants_verified":
            raise ValueError("文字图层没有通过像素边界校验")
        if artifact.parent_ids != [run.selected_artifact_id]:
            raise ValueError("文字版本不是从当前选中作品派生")
        run.state.artifacts.append(artifact)
        run.selected_artifact_id = artifact.id
        run.fulfilled_text_intent_ids.append(intent_id)
        run.state.checks.append(CheckRecord(
            node_id="text_verify", artifact_id=artifact.id, passed=True,
            scores={"exact_content": 1.0, "pixel_invariants": 1.0},
        ))
        if self.missing_exact_text_intents(run):
            run.pending_nodes = ["text_render"]
            run.state.current_node = "text_render"
            run.state.status = "needs_user"
        else:
            run.pending_nodes = ["accept"]
            run.state.current_node = "text_verify"
            run.state.status = "needs_user"
        run.state.history.append(f"text_verify:pass:{intent_id}:{artifact.id}")
        return self._touch(run)

    def cancel(self, run: AdaptiveRun) -> AdaptiveRun:
        if run.state.status == "accepted":
            raise ValueError("已接受的任务不能取消")
        run.pending_nodes = []
        run.pause_reason = ""
        run.state.status = "cancelled"
        run.state.history.append("run:cancelled")
        return self._touch(run)

    @staticmethod
    def _require_executing(run: AdaptiveRun) -> None:
        if run.state.status != "running" or run.pause_reason:
            raise ValueError("当前任务不在执行阶段，不能写入修复结果")

    @staticmethod
    def _require_user_boundary(run: AdaptiveRun) -> None:
        if (run.state.status != "needs_user" or run.pause_reason or run.repair_job_id
                or run.pending_nodes not in (["compare"], ["human_gate"], ["accept"], ["text_render"])):
            raise ValueError("请等待当前阶段完成；暂停不等于已经完成候选比较或局部修改")

    @staticmethod
    def _has_executable_work(run: AdaptiveRun) -> bool:
        return bool(set(run.pending_nodes) & {*CANDIDATE_NODES, "region_repair", "component_check"})

    def pause(self, run: AdaptiveRun) -> AdaptiveRun:
        if run.pause_reason and run.state.status == "needs_user":
            return run
        if run.state.status != "running" or not self._has_executable_work(run):
            raise ValueError("当前没有可暂停的执行阶段；等待人工操作无需暂停")
        run.pause_reason = "user"
        run.state.status = "needs_user"
        run.state.history.append("run:paused_by_user:submitted_jobs_continue")
        return self._touch(run)

    def resume(self, run: AdaptiveRun) -> AdaptiveRun:
        if run.state.status != "needs_user" or not run.pause_reason or not self._has_executable_work(run):
            raise ValueError("当前不是可恢复的暂停任务；请完成候选选择、评价或文字操作")
        run.pause_reason = ""
        run.state.status = "running"
        run.state.history.append("run:resumed")
        return self._touch(run)

    def attach_request(self, run: AdaptiveRun, request: dict) -> AdaptiveRun:
        run.base_request = request
        return self._touch(run)

    def attach_candidate_job(self, run: AdaptiveRun, node_id: str, job_id: str) -> AdaptiveRun:
        if (node_id not in CANDIDATE_NODES or run.candidates[node_id].status != "pending"
                or node_id in run.candidate_job_ids
                or (node_id not in run.pending_nodes and run.state.status != "cancelled")):
            raise ValueError("候选节点当前不可派发")
        run.candidate_job_ids[node_id] = job_id
        run.state.history.append(f"{node_id}:job:{job_id}")
        return self._touch(run)

    def _touch(self, run: AdaptiveRun) -> AdaptiveRun:
        run.updated_at = now_iso()
        self._save(run)
        return run

    def _save(self, run: AdaptiveRun) -> None:
        target = self.root / f"{run.id}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(target)

    def _load(self) -> None:
        for path in self.root.glob("*.json"):
            try:
                run = AdaptiveRun.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if run.state.status == "running":
                run.state.status = "needs_user"
                if self._has_executable_work(run):
                    run.pause_reason = "service_restart"
                    run.state.history.append("service_restart:paused_for_safety")
            self.runs[run.id] = run
