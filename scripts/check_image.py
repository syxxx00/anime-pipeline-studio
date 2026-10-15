"""生成后质量检查子进程：动漫专用检测/评分（deepghs imgutils 生态 + ultralytics 手部检测）。

用法（引擎 venv 的 python，内部含 torch/onnxruntime）：
    python scripts/check_image.py
stdin（单张）: {"image": "输入图路径", "mask_out": "手部遮罩输出路径或null",
         "hand_conf": 0.5, "max_hands": 4, "face_conf": 0.25}
stdin（批量）: {"images": [{同上单张参数}, ...]}  —— 只加载一次模型，校准/扫目录用
stdout: 单张 {"faces": {...}, "hands": {...}, "tags": {...}, "aesthetic": {"label": str, "percentile": float}, "style_era": {...}}
        批量 {"results": [同上单张结果, ...]}

职责边界：这里只产出原始检测信号（脸/手/标签/美学分/风格年代），不做合格判定；
风格区分的规则判定在 app/checks.py（子进程不知道请求的风格类别）。

手部门控（旧 FaceDetailer 方法证伪的根因修复）：
- hand_yolov9c（Bingsu/adetailer，含 AnHDet 动漫手数据）conf>=0.5（文档默认，旧值 0.3 会把崩坏的脸误检成手）
- 与动漫脸框 IoU>0.1 或手框中心落在脸框内 → 丢弃
- 框面积 > 15% 图面积 → 丢弃（躯干误检）
- 最多保留 4 框（按置信度）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 注意：cv2/numpy 不在顶层导入 —— gate_hand_boxes 是纯函数，要在项目 venv（无 cv2）里被单测直接导入。
# cv2/numpy 的导入在 check_one 内部进行（该函数只会在引擎 venv 子进程里执行）。

# 手部检测模型在 ComfyUI 的 ultralytics 目录（与引擎内节点共用同一份权重）
HAND_MODEL = Path(__file__).resolve().parent.parent / "runtime" / "ComfyUI" / "models" / "ultralytics" / "bbox" / "hand_yolov9c.pt"

# 背景信号标签（WD14/Danbooru 体系）：checks.py 的背景丰富度检查读这些键
BACKGROUND_TAGS = [
    "simple_background", "white_background", "grey_background", "gray_background",
    "gradient_background", "abstract_background", "black_background", "blue_background",
]

_hand_detector = None


def _load_hand_detector():
    global _hand_detector
    if _hand_detector is None:
        from ultralytics import YOLO
        _hand_detector = YOLO(str(HAND_MODEL))
    return _hand_detector


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def gate_hand_boxes(hand_boxes: list[list[float]], hand_scores: list[float],
                    face_boxes: list[list[float]], img_area: float,
                    max_hands: int) -> tuple[list[list[float]], list[float], list[dict]]:
    """手部门控：脸重叠/超大框/低分框丢弃。纯函数，便于单测。"""
    kept, kept_scores, rejected = [], [], []
    for box, score in sorted(zip(hand_boxes, hand_scores), key=lambda item: -item[1]):
        area_ratio = (box[2] - box[0]) * (box[3] - box[1]) / img_area
        reason = None
        if area_ratio > 0.15:
            reason = "面积过大（疑似躯干误检）"
        else:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            for face in face_boxes:
                if _iou(box, face) > 0.1 or (face[0] <= cx <= face[2] and face[1] <= cy <= face[3]):
                    reason = "与脸部区域重叠（疑似崩坏脸部误检）"
                    break
        if reason:
            rejected.append({"box": [round(v) for v in box], "score": round(score, 3), "reason": reason})
            continue
        kept.append(box)
        kept_scores.append(score)
        if len(kept) >= max_hands:
            break
    return kept, kept_scores, rejected


def check_one(image_path: Path, mask_out: Path | None, opts: dict) -> dict:
    import cv2
    import numpy as np
    from imgutils.detect import detect_faces
    # anime_dbaesthetic：Danbooru 系美学评分，返回 (label, percentile)；
    # 旧 get_aesthetic_score(CAQE) 已被官方标注 deprecated + low effectiveness，不用
    from imgutils.metrics import anime_dbaesthetic
    from imgutils.tagging import get_wd14_tags
    from imgutils.validate import anime_style_age_score

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"无法读取图片：{image_path}")
    height, width = img.shape[:2]
    img_area = float(width * height)

    face_conf = float(opts.get("face_conf", 0.25))
    faces_raw = detect_faces(image_path, conf_threshold=face_conf)
    face_boxes = [[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b, _label, _s in faces_raw]
    face_scores = [float(s) for _b, _label, s in faces_raw]

    hand_conf = float(opts.get("hand_conf", 0.5))
    max_hands = int(opts.get("max_hands", 4))
    # 强制 CPU：GPU 属于引擎。子进程碰 CUDA 会与 ComfyUI 争抢导致初始化挂死（已实测复现）；
    # 检测模型都很小，CPU 足够快（手部 yolo 冷启动实测 ~14s）
    hand_result = _load_hand_detector().predict(str(image_path), conf=hand_conf, device="cpu", verbose=False)[0]
    hand_boxes = [box.tolist() for box in hand_result.boxes.xyxy]
    hand_scores = [float(score) for score in hand_result.boxes.conf]
    kept, kept_scores, rejected = gate_hand_boxes(hand_boxes, hand_scores, face_boxes, img_area, max_hands)

    mask_written = None
    if kept and mask_out is not None:
        mask = np.zeros((height, width), dtype=np.uint8)
        for x1, y1, x2, y2 in kept:
            mask[int(y1):int(y2), int(x1):int(x2)] = 255
        kernel = np.ones((11, 11), dtype=np.uint8)  # 10px 膨胀，盖住边缘便于接缝
        mask = cv2.dilate(mask, kernel)
        cv2.imwrite(str(mask_out), mask)
        mask_written = str(mask_out)

    _rating, general, character = get_wd14_tags(image_path, general_threshold=0.1, character_threshold=0.1)
    top_general = dict(sorted(general.items(), key=lambda item: -item[1])[:40])
    top_character = dict(sorted(character.items(), key=lambda item: -item[1])[:15])
    background = {tag: float(general.get(tag, 0.0)) for tag in BACKGROUND_TAGS}

    # 场景文字第二检测器（deepghs/text_detection，ICDAR2015 训练）：漫画 DBNet 对衣物印花/
    # 竖排白字有盲区（wide_snowfield 样本实测漏检），它补覆盖。注意其输出噪声大
    # （会把领口褶皱当文字，实测真字 0.23 与误报 0.224 同档）——只能报告，不能喂自动修复。
    text_scene: dict = {"count": 0, "boxes": [], "scores": []}
    try:
        from imgutils.detect import detect_text as scene_detect_text
        scene_hits = scene_detect_text(image_path)
        text_scene = {
            "count": len(scene_hits),
            "boxes": [[round(float(v)) for v in box] for box, _label, _s in scene_hits],
            "scores": [round(float(s), 3) for _box, _label, s in scene_hits],
        }
    except Exception as exc:  # 模型缺失/加载失败不拖垮主检测
        text_scene["error"] = f"{type(exc).__name__}: {exc}"

    # 人物检测（deepghs/anime_person_detection）：给文字清理当"角色区域"依据——
    # 落在人物身上的文字是角色设定内容（如阿米娅外套印花），不能自动擦除，只能报告
    persons: dict = {"count": 0, "boxes": [], "scores": []}
    try:
        from imgutils.detect import detect_person
        persons_raw = detect_person(image_path)
        persons = {
            "count": len(persons_raw),
            "boxes": [[round(float(v)) for v in box] for box, _label, _s in persons_raw],
            "scores": [round(float(s), 3) for _box, _label, s in persons_raw],
        }
    except Exception as exc:  # 模型缺失不拖垮主检测；无人物框 = 文字清理退回无保护旧行为
        persons["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "image": {"width": width, "height": height},
        "faces": {
            "count": len(face_boxes),
            "boxes": [[round(v) for v in box] for box in face_boxes],
            "scores": [round(score, 3) for score in face_scores],
        },
        "hands": {
            "count": len(kept),
            "boxes": [[round(v) for v in box] for box in kept],
            "scores": [round(score, 3) for score in kept_scores],
            "rejected": rejected,
            "mask": mask_written,
        },
        "tags": {"background": background,
                 "top_general": {k: round(float(v), 3) for k, v in top_general.items()},
                 "top_character": {k: round(float(v), 3) for k, v in top_character.items()}},
        "text_scene": text_scene,
        "persons": persons,
        "aesthetic": (lambda pair: {"label": str(pair[0]), "percentile": round(float(pair[1]), 3)})(anime_dbaesthetic(image_path)),
        "style_era": {k: round(float(v), 3) for k, v in anime_style_age_score(image_path).items()},
    }


def main() -> None:
    # utf-8-sig：兼容外部管道（如 PowerShell）写入时带 BOM 的情况
    payload = json.loads(sys.stdin.buffer.read().decode("utf-8-sig"))
    if "images" in payload:
        results = []
        for item in payload["images"]:
            try:
                results.append(check_one(Path(item["image"]),
                                         Path(item["mask_out"]) if item.get("mask_out") else None, item))
            except Exception as exc:  # 批量模式下一张失败不拖垮整批
                results.append({"error": f"{type(exc).__name__}: {exc}", "image": item.get("image")})
        print(json.dumps({"results": results}, ensure_ascii=False))
    else:
        result = check_one(Path(payload["image"]),
                           Path(payload["mask_out"]) if payload.get("mask_out") else None, payload)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
