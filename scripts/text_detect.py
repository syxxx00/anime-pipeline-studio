"""动漫/AI 伪文字检测子进程（漫画文字检测器，manga-image-translator 同源）。

用法（引擎 venv 的 python，内部含 torch）：
    python scripts/text_detect.py
stdin（单张）: {"image": "输入图路径", "mask_out": "遮罩输出路径",
         "text_threshold": 0.5, "box_threshold": 0.7, "unclip_ratio": 2.3,
         "mask_threshold": 128, "dilate_px": 6, "device": "cuda"}
stdin（批量）: {"images": [{同上单张参数}, ...]}  —— 只加载一次模型，校准/扫目录用
stdout: 单张 {"count": N, "boxes": [[x1,y1,x2,y2],...], "scores": [...], "mask": "遮罩路径或null"}
        批量 {"results": [同上单张结果, ...]}
约定：检测不到文字时 count=0 且不落遮罩文件，调用方据此跳过文字清理工序。
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# 注意：cv2/numpy/vendor 检测器不在顶层导入 —— split_character_text 是纯函数，
# 要在项目 venv（无 cv2/torch）里被单测直接导入；重依赖的导入都在 detect_one/run 内部。
# 裁剪版漫画文字检测器（只保留 default/DBNet-ResNet34，见该目录内说明）
VENDOR = Path(__file__).resolve().parent.parent / "runtime" / "vendor" / "image-text-detection"


def split_character_text(boxes: list, scores: list, exclude_boxes: list) -> tuple[list, list]:
    """按排除区（人物框）把文字框分为"保留清理"与"角色身上"。纯函数，便于单测。

    判定 = 文字框中心落在任一人物框内：人物框是整个身体，衣物印花必然在内；
    背景招牌/海报文字不与人物框重叠，不受影响。角色身上文字是角色设定内容
    （如阿米娅外套印花），自动擦除 = 破坏设定，只能报告由人定夺。
    已知局限：人物框是矩形，当角色几乎占满画面时（storm 样本人物框 [260,2,1519,1536]
    接近全图），角色头顶后方的背景招牌也会被误保护——宁可少擦不可错擦，
    误保护只多一条报告，误擦则破坏设定。待跨样本观察后再决定是否收紧。
    """
    kept_idx, excluded_idx = [], []
    for i, box in enumerate(boxes):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        on_character = any(eb[0] <= cx <= eb[2] and eb[1] <= cy <= eb[3] for eb in exclude_boxes)
        (excluded_idx if on_character else kept_idx).append(i)
    kept = ([boxes[i] for i in kept_idx], [scores[i] for i in kept_idx])
    excluded = [{"box": boxes[i], "score": scores[i]} for i in excluded_idx]
    return kept, excluded


async def detect_one(image_path: Path, mask_out: Path | None, opts: dict, device: str) -> dict:
    import cv2
    import numpy as np

    sys.path.insert(0, str(VENDOR))
    from detection import dispatch
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"无法读取图片：{image_path}")
    textlines, raw_mask, _ = await dispatch(
        "default", img,
        int(opts.get("detect_size", 1536)),
        # 默认 0.3/0.5 是对 AI 伪文字校准过的：仓库默认 0.5/0.7 为漫画真文字调，
        # 伪文字笔画弱，0.5/0.7 下 storm 样本实测 0 检出，0.3/0.5 正常命中
        float(opts.get("text_threshold", 0.3)),
        float(opts.get("box_threshold", 0.5)),
        float(opts.get("unclip_ratio", 2.3)),
        False, False, False, False, device, False,
    )
    boxes = [[int(v) for v in line.xyxy] for line in textlines]
    scores = [round(float(line.prob), 3) for line in textlines]

    # 角色区域保护：落在人物身上的文字不进入清理（角色服装文字是设定内容，擦除=破坏设定），
    # 只随结果返回供调用方报告。人物框由调用方从 check 检测结果传入（exclude_boxes）。
    excluded: list[dict] = []
    exclude_boxes = opts.get("exclude_boxes") or []
    if exclude_boxes and boxes:
        (boxes, scores), excluded = split_character_text(boxes, scores, exclude_boxes)
        if excluded and raw_mask is not None:
            # 排除区像素级清零：否则框被滤掉、框内弱笔画仍会经两级阈值逻辑进入遮罩
            pad_e = 12
            for item in excluded:
                x1, y1, x2, y2 = item["box"]
                raw_mask[max(0, y1 - pad_e):min(raw_mask.shape[0], y2 + pad_e),
                         max(0, x1 - pad_e):min(raw_mask.shape[1], x2 + pad_e)] = 0
    # 每个框补充像素级统计：peak=框内最高文字概率，fill=框内高概率像素占比。
    # 框级 score 对长条小字会趋近 0，单看它无法区分真字与误报，要靠像素统计兜底。
    stats = []
    if raw_mask is not None:
        prob = raw_mask.astype(np.float32) / 255.0
        h, w = prob.shape[:2]
        for x1, y1, x2, y2 in boxes:
            region = prob[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if region.size:
                stats.append({"peak": round(float(region.max()), 3), "fill": round(float((region >= 0.5).mean()), 4)})
            else:
                stats.append({"peak": 0.0, "fill": 0.0})
    if not boxes or raw_mask is None or mask_out is None:
        return {"image": str(image_path), "count": len(boxes), "boxes": boxes, "scores": scores, "box_stats": stats, "excluded": excluded, "mask": None}

    # 遮罩 = 像素级文字概率图阈值化 + 轻度膨胀。
    # 不要填充整个检测框：框经 unclip 放大后远大于真实笔画，误报框会直接毁掉大片画面。
    # 但全局统一阈值会漏掉框内的弱笔画（storm 样本招牌上半段 raw<128 没被罩住，
    # 重绘后原字残留），所以分两级：框外只认强笔画(>=mask_threshold)，
    # 框内（检测器已断言"这里有字"）放宽到 box_mask_threshold 把弱笔画补齐。
    mask_threshold = int(opts.get("mask_threshold", 128))
    box_mask_threshold = int(opts.get("box_mask_threshold", 64))
    strong = raw_mask >= mask_threshold
    h, w = raw_mask.shape[:2]
    inbox = np.zeros((h, w), dtype=bool)
    pad = 8
    for x1, y1, x2, y2 in boxes:
        inbox[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)] = True
    mask = (strong | (inbox & (raw_mask >= box_mask_threshold))).astype(np.uint8) * 255
    # 小框整体填充：检测器存在"框回归看到了字、像素图漏掉笔画"的盲区
    # （storm 样本招牌上半段白字 raw 概率≈0，反色/加大 detect_size 均无效）。
    # 实测 21 框数据：小框(<=8%画面)填充损伤有界；早期误报灾难来自 25% 大框填充，
    # 所以大框仍只用像素级笔画。
    img_area = h * w
    small_box_fill = float(opts.get("small_box_fill", 0.08))
    for x1, y1, x2, y2 in boxes:
        if (x2 - x1) * (y2 - y1) / img_area <= small_box_fill:
            mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255
    dilate_px = int(opts.get("dilate_px", 6))
    if dilate_px > 0:
        kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, kernel)
    area_ratio = float((mask > 0).mean())
    if area_ratio > 0.15:
        # 防御：框内放宽仍不该罩住画面 15% 以上（误报框+噪声底的组合），退回纯强笔画
        mask = strong.astype(np.uint8) * 255
        if dilate_px > 0:
            mask = cv2.dilate(mask, kernel)
        area_ratio = float((mask > 0).mean())
    if area_ratio <= 0:
        # 框级有弱信号但像素级毫无把握：视为无文字，避免空遮罩触发无意义重绘
        return {"image": str(image_path), "count": 0, "boxes": [], "scores": [], "box_stats": [], "excluded": excluded, "mask": None}
    mask_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_out), mask)
    return {"image": str(image_path), "count": len(boxes), "boxes": boxes, "scores": scores,
            "box_stats": stats, "excluded": excluded, "mask_area_ratio": round(area_ratio, 4), "mask": str(mask_out)}


async def run(payload: dict) -> dict:
    try:
        import torch
        device = payload.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = "cpu"
    batch = payload.get("images")
    if batch:
        results = [await detect_one(Path(item["image"]), Path(item["mask_out"]) if item.get("mask_out") else None, item, device) for item in batch]
        return {"results": results}
    return await detect_one(Path(payload["image"]), Path(payload["mask_out"]) if payload.get("mask_out") else None, payload, device)


if __name__ == "__main__":
    result = asyncio.run(run(json.loads(sys.stdin.read())))
    print(json.dumps(result, ensure_ascii=False))
