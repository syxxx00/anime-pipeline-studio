from __future__ import annotations

from .newbie_compiler import CompileResult, analyze_coverage


def build_professional_review(
    result: CompileResult,
    *,
    parsed_character_tag: str = "",
    knowledge_operator: str = "",
) -> dict:
    """Return only questions that materially change the requested picture.

    Ordinary omissions are handled as explicit assumptions. This deliberately
    avoids asking a casual user to specify camera or rendering terminology.
    """
    coverage = analyze_coverage(result.spec, result.clause_map)
    questions: list[dict] = []
    compiler_issues: list[dict] = []

    final_tag = result.spec.character_tag
    if (
        knowledge_operator
        and parsed_character_tag
        and parsed_character_tag != "original_character"
        and parsed_character_tag != final_tag
    ):
        questions.append({
            "id": "knowledge_identity_conflict",
            "kind": "conflict",
            "question": "原描述中的角色与知识库选择不一致。最终应以哪一个角色为准？",
            "reason": f"描述解析为 {parsed_character_tag}，知识库选择为 {final_tag}。",
            "action": "请修改原描述，或重新选择知识库角色。",
        })

    contradictory = [row for row in coverage["clauses"] if row["status"] == "contradictory"]
    if contradictory:
        examples = "；".join(row["text"] for row in contradictory[:2])
        questions.append({
            "id": "contradictory_description",
            "kind": "conflict",
            "question": "描述中有无法同时成立的画面要求，哪一项应保留？",
            "reason": examples,
            "action": "请在原描述中删掉不需要的版本，再重新解析。",
        })

    missing_objects = [
        row for row in coverage["clauses"]
        if row["status"] in {"partial", "unmapped"}
        and (row.get("missing_objects") or (
            "objects" in row["reason"] and any(token in row["reason"] for token in ("为空", "未写入"))
        ))
    ]
    if missing_objects:
        compiler_issues.append({
            "id": "missing_key_object",
            "kind": "compiler_omission",
            "title": "解析器遗漏了已经指定的内容",
            "reason": "；".join(row["reason"] for row in missing_objects[:2]),
            "action": "原文无需重复补充。可重新解析，或在专业模式修正规格；继续生成可能遗漏这些要求。",
        })

    ambiguous = [row for row in coverage["clauses"] if row["status"] == "ambiguous"]
    if ambiguous:
        questions.append({
            "id": "ambiguous_visual_meaning",
            "kind": "severe_missing",
            "question": "这段抽象描述希望具体变成什么可见画面？",
            "reason": "；".join(row["text"] for row in ambiguous[:2]),
            "action": "只需补充角色在做什么或身处何处，风格细节可继续交给系统。",
        })

    assumptions: list[str] = []
    source = result.spec.source_sentence.lower()
    if "unspecified neutral pose" in result.spec.body_action:
        assumptions.append("未指定明确动作：使用自然中性姿态。")
    if result.spec.background == ["simple coherent background"]:
        assumptions.append("未指定环境：使用与主体协调的简洁背景。")
    if not any(token in source for token in ("特写", "全身", "半身", "远景", "俯视", "仰视", "close-up", "full body", "wide shot")):
        assumptions.append("未指定镜头：由管线按角色动作和场景自动构图。")

    return {
        "blocking": bool(questions),
        "questions": questions[:3],
        "compiler_issues": compiler_issues,
        "assumptions": assumptions,
    }
