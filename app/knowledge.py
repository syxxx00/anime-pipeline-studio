from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path

from .config import PROJECT_ROOT
from .newbie_compiler import CompileResult, lint


KNOWLEDGE_ROOT = PROJECT_ROOT / "data" / "knowledge"

_PLACEHOLDER_PATTERNS = (
    re.compile(r"^canonical\b", re.I),
    re.compile(r"\bidentity$", re.I),
    re.compile(r"^[\w' -]+ outfit$", re.I),
)


def is_injectable_visual_fact(value: str) -> bool:
    """Reject catalog labels and promises of correctness that contain no visual fact."""
    normalized = " ".join(value.strip().split())
    return bool(normalized) and not any(pattern.search(normalized) for pattern in _PLACEHOLDER_PATTERNS)


VISUAL_FACT_CHECK_SCOPE = "declared_source_presence_only"


def visual_fact_record_sha256(record: dict) -> str:
    """Identify the stored declaration, not the remote artwork or its accuracy."""
    raw = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def declared_visual_fact_records(owner: dict, key: str, fallback_source: str = "") -> tuple[list[dict], list[dict]]:
    """Keep usable existing declarations together with their actual evidence scope.

    The historical ``verified`` status is preserved, not re-audited here. This
    loader checks only the declaration, source presence and placeholder shape;
    it neither fetches a URL nor reads artwork pixels or a review receipt.
    Legacy string arrays remain readable for migration/audit, but are not
    silently promoted to sourced character knowledge.
    """
    records: list[dict] = []
    rejected: list[dict] = []
    for item in owner.get(key, []):
        if isinstance(item, str):
            rejected.append({"value": item, "reason": "legacy_fact_without_evidence"})
            continue
        if not isinstance(item, dict):
            rejected.append({"value": str(item), "reason": "invalid_fact"})
            continue
        value = str(item.get("value", "")).strip()
        source = str(item.get("source_url") or fallback_source).strip()
        if item.get("status") != "verified":
            rejected.append({"value": value, "reason": "not_verified"})
        elif not source:
            rejected.append({"value": value, "reason": "missing_source"})
        elif not is_injectable_visual_fact(value):
            rejected.append({"value": value, "reason": "non_visual_placeholder"})
        else:
            digest = visual_fact_record_sha256(item)
            declared_id = str(item.get("id", "")).strip()
            records.append({
                "value": value,
                "record_id": declared_id or "fact_" + digest,
                "record_id_origin": "stored_id" if declared_id else "record_content_sha256",
                "record_sha256": digest,
                "source_binding_sha256": visual_fact_record_sha256({'record_sha256': digest, 'source_url': source}),
                "raw_record": deepcopy(item),
                "source_url": source,
                "record_source_url": str(item.get("source_url") or "").strip(),
                "source_url_origin": "record" if str(item.get("source_url") or "").strip() else "owner_fallback",
                "declared_status": item["status"],
                "evidence_type": str(item.get("evidence_type") or ""),
                "asset_key": str(item.get("asset_key") or ""),
                "evidence_check_scope": VISUAL_FACT_CHECK_SCOPE,
                "local_original_checked": False,
                "review_receipt_checked": False,
                "visual_accuracy_verified": False,
            })
    return records, rejected


def verified_visual_facts(owner: dict, key: str, fallback_source: str = "") -> tuple[list[str], list[dict]]:
    """Compatibility projection of existing sourced, declared-verified facts.

    The name is historical: source/status checks do not certify visual accuracy.
    New provenance-aware paths use ``declared_visual_fact_records`` instead.
    """
    records, rejected = declared_visual_fact_records(owner, key, fallback_source)
    return list(dict.fromkeys(item["value"] for item in records)), rejected


def audit_pack(pack: dict) -> dict:
    appearances = 0
    ready = 0
    placeholder_values = 0
    missing_source = 0
    entries: list[dict] = []
    for operator in pack.get("operators", []):
        operator_source = operator.get("source_url", "")
        fixed_ready, fixed_rejected = verified_visual_facts(operator, "fixed_visual_facts", operator_source)
        for appearance in operator.get("appearances", []):
            appearances += 1
            source_url = appearance.get("source_url") or operator_source
            injectable, rejected_records = verified_visual_facts(appearance, "visual_facts", source_url)
            legacy = list(appearance.get("clothing", []))
            rejected_records.extend({"value": value, "reason": "legacy_fact_without_evidence"} for value in legacy)
            rejected = [item["value"] for item in fixed_rejected + rejected_records]
            usable = bool(source_url and (fixed_ready or injectable) and injectable)
            if usable:
                ready += 1
            if not source_url:
                missing_source += 1
            placeholder_values += len(rejected)
            entries.append({
                "operator_id": operator.get("id", ""),
                "appearance_id": appearance.get("id", ""),
                "status": "injectable" if usable else "reference_only",
                "injectable_facts": len(fixed_ready) + len(injectable),
                "appearance_facts": injectable,
                "rejected_placeholders": rejected,
                "rejected_fact_records": fixed_rejected + rejected_records,
                "source_url": source_url,
            })
    return {
        "schema_version": "knowledge-audit-v3",
        "evidence_check_scope": VISUAL_FACT_CHECK_SCOPE,
        "local_original_checked": False,
        "review_receipt_checked": False,
        "visual_accuracy_verified": False,
        "appearance_count": appearances,
        "injectable_appearance_count": ready,
        "reference_only_count": appearances - ready,
        "placeholder_value_count": placeholder_values,
        "missing_source_count": missing_source,
        "entries": entries,
    }


def pack_with_audit(pack_id: str) -> dict:
    pack = load_pack(pack_id)
    audit = audit_pack(pack)
    enriched = {**pack, "audit": audit}
    entry_map = {(item["operator_id"], item["appearance_id"]): item for item in audit["entries"]}
    enriched["operators"] = [
        {
            **operator,
            "appearances": [
                {**appearance, "knowledge_status": entry_map[(operator["id"], appearance["id"])]["status"]}
                for appearance in operator.get("appearances", [])
            ],
        }
        for operator in pack.get("operators", [])
    ]
    return enriched


def load_pack(pack_id: str) -> dict:
    if not pack_id.replace("_", "").replace("-", "").isalnum():
        raise KeyError(pack_id)
    path = KNOWLEDGE_ROOT / f"{pack_id}.json"
    if not path.exists():
        raise KeyError(pack_id)
    return json.loads(path.read_text(encoding="utf-8"))


def list_packs() -> list[dict]:
    results: list[dict] = []
    for path in sorted(KNOWLEDGE_ROOT.glob("*.json")):
        try:
            pack = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        audit = audit_pack(pack)
        results.append({
            "id": pack["id"],
            "name": pack["name"],
            "status": pack.get("status", "unknown"),
            "description": pack.get("description", ""),
            "operator_count": len(pack.get("operators", [])),
            "audit": {key: value for key, value in audit.items() if key != "entries"},
        })
    return results


def apply_knowledge(result: CompileResult, pack_id: str, operator_id: str, appearance_id: str) -> CompileResult:
    pack = load_pack(pack_id)
    operator = next((item for item in pack.get("operators", []) if item.get("id") == operator_id), None)
    if operator is None:
        raise ValueError("知识库中不存在所选角色")
    appearance = next((item for item in operator.get("appearances", []) if item.get("id") == appearance_id), None)
    if appearance is None:
        raise ValueError("所选角色没有这个外观")
    spec = result.spec.model_copy(deep=True)
    spec.character_name = operator["name"]
    spec.character_tag = operator["tag"]
    spec.source = pack_id
    source_url = appearance.get("source_url") or operator.get("source_url", "")
    spec.appearance, rejected_appearance = verified_visual_facts(operator, "fixed_visual_facts", operator.get("source_url", ""))
    spec.clothing, rejected_clothing = verified_visual_facts(appearance, "visual_facts", source_url)
    rejected = len(rejected_appearance) + len(rejected_clothing) + len(operator.get("fixed_appearance", [])) + len(appearance.get("clothing", []))
    spec.source = source_url or f"{pack_id}:unverified"
    warnings = lint(spec) + [f"已选择角色索引：{operator['name']} / {appearance['name']}。保留已有有来源的具体视觉声明作引导；本次只检查声明与来源存在，未重新审读官方像素或确认生成身份。"]
    if rejected:
        warnings.append(f"知识库安全门已忽略 {rejected} 条占位描述（如 canonical/outfit 标签），它们不能证明具体外观。")
    if not spec.clothing:
        warnings.append("该外观目前只有参考索引，没有足够的可注入服装事实；不会假装已锁定服装。")
    return CompileResult(spec, result.compiler + "+knowledge_audited", warnings, clause_map=result.clause_map)
