"""check 系统：生成后全面质量检查（动漫专用），规则按风格类别区分。

职责：
- 风格解析：从请求（semantic_spec / 编译后 XML 的 <style> 段）判定风格类别
- 规则引擎：data/check_style_rules.json 决定每类风格启用哪些检查、用什么阈值
- 本地检测编排：scripts/check_image.py 子进程（动漫脸/手/WD14/美学评分/风格年代一次跑完）
- 报告构建：把原始信号按风格规则翻译成 CheckEntry 列表

判定原则：只报告证据，不因检测到手或文字自动修改图片；缺证据不等于通过。
所有阈值都是待用户跨样本校准的占位默认值（不要把特殊当一般）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import PROJECT_ROOT, Settings
from .comfy_client import local_engine_environment
from .newbie_compiler import BASE_STYLE
from .schemas import GenerateRequest
from .text_detect import DETECT_LOCK
from .check_intent import check_spec
from .check_evidence import object_evidence, words

logger = logging.getLogger(__name__)

RULES_PATH = PROJECT_ROOT / "data" / "check_style_rules.json"
STYLE_CATEGORIES = ("chibi", "poster", "flat", "painterly", "cel", "realistic", "standard")

# Use the same existing D-drive cache/offline policy as the image engine.
# A missing detector is a report limitation, never permission to download.


def capability_report(settings: Settings) -> dict:
    hand_model = PROJECT_ROOT / "runtime" / "ComfyUI" / "models" / "ultralytics" / "bbox" / "hand_yolov9c.pt"
    vlm_path = PROJECT_ROOT / "runtime" / "models" / "Qwen3-VL-2B-Instruct"
    return {
        "hand_localization": {
            "status": "available" if hand_model.is_file() else "missing",
            "model": "hand_yolov9c", "can_prove": ["hand_region_present"],
            "cannot_prove": ["finger_count_correct", "joints_correct", "contact_relation_correct"],
        },
        "hand_anatomy_judgement": {
            "status": "research_candidate" if vlm_path.is_dir() else "unavailable",
            "candidate": "Qwen3-VL-2B-Instruct",
            "gate": "must_be_calibrated_on_user_rated_anime_failures_before_routing_or_repair",
        },
        "identity": {"status": "evidence_only", "can_prove": ["character_tag_signal"], "cannot_prove": ["strict_non_ooc"]},
        "scene_semantics": {"status": "human_required", "cannot_prove": ["object_shape_correct", "spatial_relation_correct"]},
        "exact_text": {
            "status": "deterministic_render_available",
            "ocr": "available" if shutil.which("tesseract") else "optional_engine_missing",
            "pixel_invariant_check": True,
        },
        "automatic_destructive_cleanup": {"status": "disabled"},
        "iteration_change": {
            "status": "evidence_only", "can_prove": ["pixel_change_from_parent_and_root"],
            "cannot_prove": ["noise_rate", "aesthetic_improvement", "semantic_correction"],
            "automatic_accept_or_repaint": False,
        },
    }


@dataclass(slots=True)
class CheckEntry:
    """单项检查结果。status: pass / warning / unknown / repaired / skipped / error"""

    name: str
    status: str
    summary: str
    details: dict = field(default_factory=dict)


@dataclass(slots=True)
class CheckContext:
    """一次 check 的全部中间产物：风格类别、规则、原始检测信号、最终报告。"""

    style_category: str
    rules: dict
    detection: dict | None = None
    entries: list[CheckEntry] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "style_category": self.style_category,
            "checks": [asdict(entry) for entry in self.entries],
        }


def load_style_rules() -> dict:
    try:
        return json.loads(RULES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("check_style_rules.json 读取失败，全部检查按 standard 默认规则执行")
        return {}


def _style_text_blob(request: GenerateRequest) -> str:
    """汇集最终模型原生提示词中的风格信息，供 check 选择阈值。"""
    parts: list[str] = []
    spec = check_spec(request)
    if isinstance(spec, dict):
        style_mode = spec.get("style_mode")
        if style_mode:
            parts.append(str(style_mode))
        style_tags = spec.get("style")
        if isinstance(style_tags, list):
            parts.extend(str(tag) for tag in style_tags)
    # An explicit contract is the common evaluation target across model/form
    # variants. Model-specific syntax must not change the judging rules.
    if not request.scene_contract:
        for match in re.finditer(r"<style>(.*?)</style>", request.prompt, re.DOTALL | re.IGNORECASE):
            parts.append(match.group(1))
        if '<style>' not in request.prompt.lower():
            parts.append(request.prompt)
        parts.extend((request.source_prompt, request.anima_prompt))
    blob = " ".join(parts).lower()
    # BASE_STYLE 是 standard 模式的模板文案，其中 "soft painterly cel shading" 带 painterly 字样，
    # 不剔除则所有标准动漫都被误判成 painterly（5f8bb60c 实测）。模板文案不代表风格意图。
    for item in BASE_STYLE:
        blob = blob.replace(item.lower(), " ")
    return blob


def resolve_style_category(request: GenerateRequest, rules: dict | None = None) -> str:
    """显式选择（请求 style_category 非 auto）优先；否则按规则文件里的判定线索归类；
    判定不出 → standard（全项开启、默认阈值）。"""
    rules = rules if rules is not None else load_style_rules()
    explicit = getattr(request, "style_category", "auto") or "auto"
    if explicit != "auto" and explicit in STYLE_CATEGORIES:
        return explicit
    clues = rules.get("_风格判定线索", {})
    priority = rules.get("_判定优先级", [])
    blob = _style_text_blob(request)
    for category in priority:
        if any(clue.lower() in blob for clue in clues.get(category, [])):
            return category
    return "standard"


def rules_for(rules: dict, category: str) -> dict:
    """取某风格类别的检查规则；缺类别或缺检查项时回退 standard。"""
    standard = rules.get("standard", {}).get("checks", {})
    specific = rules.get(category, {}).get("checks", {})
    merged = {}
    for name in ("text", "hands", "faces", "breakdown", "style_era", "background", "aesthetic"):
        merged[name] = {**standard.get(name, {"enabled": True}), **specific.get(name, {})}
    return merged


def _run_check_detect(settings: Settings, image_path: Path, mask_path: Path, hand_conf: float, max_hands: int) -> dict:
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_image.py"
    payload = {
        "image": str(image_path),
        "mask_out": str(mask_path),
        "hand_conf": hand_conf,
        "max_hands": max_hands,
    }
    env = local_engine_environment()
    env['CUDA_VISIBLE_DEVICES'] = ''  # CPU checks never compete for image VRAM.
    try:
        completed = subprocess.run(
            [settings.semantic_python, str(script)],
            input=json.dumps(payload, ensure_ascii=False),
            text=True, encoding="utf-8", capture_output=True, timeout=300, check=True, env=env,
        )
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('check 本地检测超时；没有自动重试或联网下载') from exc
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or '')[-500:]
        missing = any(code in (exc.stderr or '') for code in ('LocalEntryNotFoundError', 'OfflineModeIsEnabled'))
        if missing:
            raise RuntimeError(f'check 缺少本地缓存，未联网下载；检查未完成：{tail}') from exc
        raise RuntimeError(f'check 检测子进程退出码 {exc.returncode}：{tail}') from exc


async def run_detection(settings: Settings, image_path: Path, mask_path: Path,
                        hand_conf: float = 0.5, max_hands: int = 4) -> dict:
    """本地全量检测（子进程全走 CPU + 离线缓存，与引擎 GPU 完全隔离）。串行锁沿用，防御多任务并发起子进程。"""
    async with DETECT_LOCK:
        return await asyncio.to_thread(_run_check_detect, settings, image_path, mask_path, hand_conf, max_hands)


# WD14/Danbooru 中指向"画面含文字"的标签。
# 漫画 DBNet 文字检测器有盲区：衣物印花/竖排白字（wide_snowfield 样本实测漏检，
# text_threshold 降到 0.15、detect_size 2048 仍 0 检出——模型覆盖问题，非阈值问题），
# 但 WD14 的 clothes_writing 等标签能捕捉到信号。无定位框 → 只报告不重绘。
TEXT_HINT_TAGS = ("clothes_writing", "text_focus", "english_text", "sign", "logo",
                  "speech_bubble", "sound_effects")


def text_tag_hints(detection: dict, threshold: float = 0.5) -> list[tuple[str, float]]:
    """从检测结果的 WD14 标签里挑疑似文字信号，按概率降序。"""
    tags = (detection.get("tags") or {}).get("top_general") or {}
    hints = [(tag, float(tags[tag])) for tag in TEXT_HINT_TAGS if float(tags.get(tag, 0.0)) >= threshold]
    return sorted(hints, key=lambda item: -item[1])


def _expects_character(request: GenerateRequest) -> bool:
    spec = check_spec(request)
    if 'expects_character' in spec:
        return bool(spec['expects_character'])
    return bool(spec.get('character_tag') or spec.get('character_name') or spec.get('count'))


def build_report(detection: dict | None, rules: dict, request: GenerateRequest) -> list[CheckEntry]:
    """把原始检测信号按风格规则翻译成报告。detection 为 None 表示本地检测被整体跳过。"""
    entries: list[CheckEntry] = []
    detection = detection or {}

    spec = check_spec(request)
    if spec.get('relations'):
        entries.append(CheckEntry('scene_relations', 'unknown',
                                  '已保留全部关系要求；当前检测器不能核验左右、接触、支撑与物体形状，请人工检查',
                                  {'expected_relations': spec['relations']}))
    if detection and spec:
        expected = str(spec.get("character_tag", "")).lower().replace(" ", "_")
        characters = (detection.get("tags") or {}).get("top_character") or {}
        normalized = {str(tag).lower().replace(" ", "_"): float(score) for tag, score in characters.items()}
        match = next(((tag, score) for tag, score in normalized.items()
                      if expected and (expected == tag or expected in tag or tag in expected)), None)
        if expected and match:
            entries.append(CheckEntry("identity", "pass", f"角色身份标签命中：{match[0]}（{match[1]:.2f}）；不代表服装、脸型或道具无 OOC",
                                      {"expected": expected, "top_character": characters}))
        elif expected and normalized:
            top = max(normalized.items(), key=lambda item: item[1])
            entries.append(CheckEntry("identity", "warning",
                                      f"未命中目标角色 {expected}；最高角色信号为 {top[0]}（{top[1]:.2f}），需人工核对 OOC",
                                      {"expected": expected, "top_character": characters}))
        elif expected:
            entries.append(CheckEntry("identity", "warning", "角色分类器没有给出可核验身份；不能据此宣称不 OOC",
                                      {"expected": expected, "top_character": characters}))

        persons = detection.get("persons") or {}
        image = detection.get("image") or {}
        area = float(image.get("width", 0)) * float(image.get("height", 0))
        boxes = persons.get("boxes") or []
        ratios = [max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1])) / area
                  for b in boxes if area > 0 and len(b) == 4]
        framing = " ".join(str(item).lower() for item in spec.get("framing") or [])
        expects_wide = any(token in framing for token in ("wide", "environment", "distant", "全景", "远景"))
        largest = max(ratios, default=0.0)
        if not boxes or not framing or area <= 0:
            entries.append(CheckEntry('composition', 'unknown', '缺少可用人物框、景别要求或图片尺寸，不能判定构图通过',
                                      {'framing': framing, 'person_area_ratios': ratios}))
        elif expects_wide and largest > 0.62:
            entries.append(CheckEntry("composition", "warning",
                                      f"语义要求环境构图，但人物占框约 {largest:.0%}，疑似退化为肖像",
                                      {"framing": framing, "person_area_ratios": ratios}))
        else:
            # A person box cannot prove that the head/feet or requested objects
            # are inside frame. Passing the wide-shot heuristic is not success.
            entries.append(CheckEntry("composition", "unknown", f"人物最大占框约 {largest:.0%}；人物框不足以核验完整入框及构图要求",
                                      {"framing": framing, "person_area_ratios": ratios,
                                       "evidence_type": "person_bbox_only"}))

        objects = spec.get("objects") or []
        expected_objects = " ".join(
            " ".join([str(item.get("name", "")), *[str(v) for v in item.get("appearance") or []]])
            for item in objects if isinstance(item, dict)
        ).lower().replace("_", " ")
        general = (detection.get("tags") or {}).get("top_general") or {}
        object_signals = {str(tag): float(score) for tag, score in general.items() if float(score) >= 0.35}
        evidence = object_evidence([item for item in objects if isinstance(item, dict)], general)
        suspicious_names = ("food", "fruit", "flower", "animal", "cat", "dog", "sword", "gun", "weapon", "ball", "book")
        unexpected = {tag: score for tag, score in object_signals.items()
                      if (words(tag) & set(suspicious_names)) - words(expected_objects)}
        if objects and evidence['missing_noun_signals']:
            entries.append(CheckEntry("object_grounding", "warning",
                                      "请求包含明确道具，但标签检测没有确认该道具；请人工检查形状与接触关系",
                                      {"expected": expected_objects, "top_general": object_signals, "unexpected": unexpected, **evidence}))
        elif objects and evidence['unsupported_names']:
            entries.append(CheckEntry('object_grounding', 'unknown', '道具名缺少可核验的名词信号，属性命中不能代替物体确认', evidence))
        elif unexpected:
            entries.append(CheckEntry("object_grounding", "warning", "发现可能未请求的物体意象，请人工检查",
                                      {"expected": expected_objects, "unexpected": unexpected}))
        elif objects:
            entries.append(CheckEntry("object_grounding", "pass", "道具标签有对应信号（空间/接触关系仍需人工检查）",
                                      {"expected": expected_objects, **evidence}))

    faces_cfg = rules.get("faces", {})
    if not faces_cfg.get("enabled", True):
        entries.append(CheckEntry("faces", "skipped", "该风格类别不检查面部"))
    elif detection:
        faces = detection.get("faces", {})
        count = int(faces.get("count", 0))
        min_conf = float(faces_cfg.get("min_conf", 0.4))
        low_conf = [s for s in faces.get("scores", []) if s < min_conf]
        if 'count' not in faces:
            entries.append(CheckEntry('faces', 'unknown', '缺少人脸检测结果，不能判断脸部情况'))
        elif count == 0 and _expects_character(request):
            entries.append(CheckEntry("faces", "warning", "未检测到动漫人脸（语义规格要求有角色）",
                                      {"count": count, "scores": faces.get("scores", [])}))
        elif count == 0:
            entries.append(CheckEntry('faces', 'unknown', '未检测到人脸，且没有明确角色要求；不判定为脸部通过'))
        elif low_conf:
            entries.append(CheckEntry("faces", "warning", f"{len(low_conf)} 张脸置信度低于 {min_conf}",
                                      {"count": count, "scores": faces.get("scores", [])}))
        else:
            entries.append(CheckEntry("faces", "pass", f"检测到 {count} 张脸；仅证明脸存在，不证明五官、性别感或 OOC 正确",
                                      {"count": count, "scores": faces.get("scores", [])}))

    bg_cfg = rules.get("background", {})
    if not bg_cfg.get("enabled", True):
        entries.append(CheckEntry("background", "skipped", "该风格类别下简单背景是正常表现"))
    elif detection:
        bg = detection.get("tags", {}).get("background", {})
        threshold = float(bg_cfg.get("simple_bg_threshold", 0.5))
        worst = max(bg.items(), key=lambda item: item[1], default=None)
        if worst and worst[1] >= threshold:
            entries.append(CheckEntry("background", "warning",
                                      f"背景偏简单（{worst[0]}={worst[1]:.2f} ≥ {threshold}）", dict(bg)))
        elif bg:
            entries.append(CheckEntry("background", "pass",
                                      (f"背景非简单背景（最强简单背景信号 {worst[1]:.2f}）；未核验语义、形状与空间关系"
                                       if worst else "无背景标签信号；未核验语义、形状与空间关系"),
                                      dict(bg)))
        else:
            entries.append(CheckEntry('background', 'unknown', '无背景标签信号，不能据此判断背景正常'))

    aes_cfg = rules.get("aesthetic", {})
    if not aes_cfg.get("enabled", True):
        entries.append(CheckEntry("aesthetic", "skipped", "该风格类别不评美学分"))
    elif detection:
        aes = detection.get("aesthetic") or {}
        score = float(aes.get("percentile", 0.0))
        label = str(aes.get("label", ""))
        floor = float(aes_cfg.get("warn_below", 0.4))
        status = 'unknown' if 'percentile' not in aes else ("warning" if score < floor else "pass")
        entries.append(CheckEntry("aesthetic", status, f"数据集美学分 {label}（百分位 {score:.2f}，下限 {floor}）；不能替代解剖与语义检查",
                                  {"label": label, "percentile": score}))

    era_cfg = rules.get("style_era", {})
    if not era_cfg.get("enabled", True):
        entries.append(CheckEntry("style_era", "skipped", "该风格类别不判定风格年代"))
    elif detection:
        era = detection.get("style_era", {})
        if era:
            top = max(era.items(), key=lambda item: item[1])
            entries.append(CheckEntry("style_era", "pass", f"风格年代：{top[0]}（{top[1]:.2f}）", dict(era)))

    brk_cfg = rules.get("breakdown", {})
    if not brk_cfg.get("enabled", True):
        entries.append(CheckEntry("breakdown", "skipped", "该风格类别不查画面崩坏"))
    elif detection:
        faces = detection.get("faces", {})
        aes = detection.get("aesthetic") or {}
        score = float(aes.get("percentile", 0.0))
        floor = float(brk_cfg.get("aesthetic_floor", 0.3))
        no_face = int(faces.get("count", 0)) == 0 and _expects_character(request)
        if 'count' not in faces or 'percentile' not in aes:
            entries.append(CheckEntry('breakdown', 'unknown', '缺少检测信号，无法执行崩坏风险启发式'))
        elif no_face and score < floor:
            entries.append(CheckEntry("breakdown", "warning",
                                      f"疑似画面崩坏（无脸 + 美学百分位 {score:.2f} < {floor}）",
                                      {"face_count": 0, "aesthetic": score}))
        elif no_face or score < floor:
            reason = "未检测到脸" if no_face else f"美学百分位 {score:.2f} < {floor}"
            entries.append(CheckEntry("breakdown", "warning", f"崩坏风险信号：{reason}",
                                      {"face_count": faces.get("count", 0), "aesthetic": score}))
        else:
            entries.append(CheckEntry("breakdown", "unknown", "未触发“无脸/低美学分”启发式；尚未核验彩噪、碎裂及解剖，不能判定画面正常",
                                      {"face_count": faces.get("count", 0), "aesthetic": score}))

    # 场景文字第二检测器（报告型）：漫画 DBNet 的盲区（衣物印花/竖排字）由它补覆盖。
    # 实测其信号噪声大（真字 0.23 vs 领口褶皱误报 0.224），阈值过滤后仍只报告、绝不进修复。
    text_cfg = rules.get("text", {})
    if not text_cfg.get("enabled", True):
        pass  # poster 等允许文字的风格：text 条目在 jobs.py 记 skipped，这里同样不报
    elif detection:
        scene = detection.get("text_scene") or {}
        conf = float(text_cfg.get("scene_text_conf", 0.2))
        hits = [(b, s) for b, s in zip(scene.get("boxes", []), scene.get("scores", [])) if s >= conf]
        if scene.get("error"):
            entries.append(CheckEntry("text_scene", "error", f"场景文字检测失败：{scene['error']}"))
        elif hits:
            entries.append(CheckEntry(
                "text_scene", "warning",
                f"场景文字检测器发现 {len(hits)} 处疑似文字（低置信信号，可能误报，请目检）",
                {"boxes": [b for b, _s in hits], "scores": [s for _b, s in hits], "conf": conf}))
        elif 'count' not in scene:
            entries.append(CheckEntry('text_scene', 'unknown', '缺少场景文字检测结果，不能据此宣称无文字'))
        else:
            entries.append(CheckEntry("text_scene", "pass", "场景文字检测：未见疑似文字"))

    return entries
