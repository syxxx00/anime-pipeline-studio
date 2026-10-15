from __future__ import annotations

import re


_FACETS = {
    "identity": ("角色", "身份", "ooc", "辨识度", "像不像", "串角色", "异格", "皮肤", "服装", "知识库", "类miku"),
    "face": ("脸", "面部", "五官", "眼睛", "眼神", "女相", "表情", "鼻子", "嘴"),
    "hands": ("手", "手指", "指头", "手腕", "掌", "握", "抓", "按下", "接触"),
    "anatomy": ("肢体", "人体", "解剖", "胳膊", "手臂", "腿", "脚", "身体", "姿势", "比例"),
    "object": ("道具", "物体", "武器", "起爆器", "按钮", "圆锯", "十字架"),
    "background": ("背景", "环境", "长廊", "礁石", "海底", "天空", "建筑", "场景"),
    "composition": ("构图", "景别", "特写", "全身", "半身", "占画面", "视角", "镜头"),
    "style": ("风格", "画风", "2d", "3d", "平涂", "厚涂", "线条", "色彩", "配色", "光影"),
    "quality": ("细节", "画质", "清晰", "模糊", "碎裂", "噪点", "精细", "完成度"),
    "semantics": ("提示词", "描述", "语义", "没生成", "缺失", "不符合", "没有出现", "图文", "不匹配"),
    "text": ("文字", "字符", "水印", "签名"),
}

_POSITIVE = ("好看", "惊艳", "喜欢", "可爱", "正确", "自然", "漂亮", "优秀", "不错", "很棒", "干净", "准确", "有优势")
_NEGATIVE = ("问题", "缺少", "缺失", "没有", "没生成", "不对", "错误", "奇怪", "崩", "模糊", "碎裂", "不符合", "糟糕", "多余", "少了", "ooc", "违和", "不匹配", "失衡", "单一", "不喜欢", "不好看", "不正确", "不自然", "不准确")
_CLAUSE_SPLIT = re.compile(r"(?<=[，。！？!?；;\n])|(?:但是|不过|然而|其次|而且)")
_NEGATED_PROBLEM = re.compile(r"(?:没有|没|无|不存在|不再)(?:明显的?|什么)?(?:问题|错误|违和感?|ooc|崩坏|崩)", re.I)
_VAGUE_REASSURANCE = re.compile(r"(?:其他)?问题(?:都)?(?:还好|不大|不明显)")


def analyze_review(text: str) -> dict:
    """Create a replaceable search index while keeping user prose authoritative."""
    normalized = " ".join(text.strip().split())
    lowered = normalized.lower()
    facets = [name for name, hints in _FACETS.items() if any(hint in lowered for hint in hints)]
    positives: list[str] = []
    problems: list[str] = []
    unclassified: list[str] = []
    for raw in _CLAUSE_SPLIT.split(normalized):
        clause = raw.strip(" ，。；;\n")
        if not clause:
            continue
        # Negated/contrastive attribution cannot be reduced to keyword counts.
        # Keep the complete original clause for semantic/human interpretation.
        if '不是' in clause or '并非' in clause:
            unclassified.append(clause)
            continue
        negated_problem = bool(_NEGATED_PROBLEM.search(clause))
        remaining = _VAGUE_REASSURANCE.sub("", _NEGATED_PROBLEM.sub("", clause.lower()))
        has_positive = negated_problem or any(token in remaining for token in _POSITIVE)
        has_negative = any(token in remaining for token in _NEGATIVE) or any(token in remaining for token in ('没按要求', '未按要求', '不按要求'))
        if has_positive and not has_negative:
            positives.append(clause)
        elif has_negative:
            problems.append(clause)
        else:
            unclassified.append(clause)
    return {
        "version": "rules-zh-v2",
        "method": "keyword_index_not_a_quality_verdict",
        "requires_semantic_review": bool(unclassified),
        "facets": facets,
        "positive_excerpts": positives,
        "problem_excerpts": problems,
        "unclassified_excerpts": unclassified,
    }
