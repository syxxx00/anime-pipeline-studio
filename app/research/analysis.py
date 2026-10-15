from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean

from ..reviews import analyze_review


AXES = ("model", "pipeline", "recipe", "character", "scene", "prompt_form", "seed")


def _axis(item, key: str) -> str:
    axes = item.axes or {}
    request = item.request or {}
    if key == 'model' and request.get('model_id'):
        return str(request['model_id'])
    if key == 'pipeline' and request.get('render_pipeline'):
        return str(request['render_pipeline'])
    if key == 'recipe':
        return str(axes.get('recipe') or axes.get('model') or request.get('output_label', '').split('__')[0] or request.get('render_pipeline') or 'unknown')
    aliases = {
        "model": ("model", "model_id"),
        "pipeline": ("pipeline", "render_pipeline"),
        "character": ("character", "character_id"),
        "scene": ("scene", "scene_id"),
        "prompt_form": ("prompt_form",),
        "seed": ("seed",),
    }
    for name in aliases[key]:
        value = axes.get(name, request.get(name))
        if value not in (None, ""):
            return str(value)
    return "unknown"


def _item_reviews(item, artwork_store) -> list[dict]:
    reviews: list[dict] = []
    seen: set[str] = set()
    for index, _image in enumerate(item.images):
        artwork = artwork_store.get(artwork_store.id_for(item.job_id, index)) if item.job_id else None
        for review in artwork.reviews if artwork else []:
            review_id = str(review.get("id") or (index, review.get("created_at"), review.get("text")))
            if review_id not in seen:
                reviews.append({**review, "image_index": index, "analysis": analyze_review(review.get("text", ""))})
                seen.add(review_id)
    for review in item.reviews:
        review_id = str(review.get("id") or (review.get("image_index", 0), review.get("created_at"), review.get("text")))
        if review_id not in seen:
            reviews.append({**review, "analysis": analyze_review(review.get("text", ""))})
            seen.add(review_id)
    return reviews


def _is_user_review(review: dict) -> bool:
    """Use declared attribution, not wording, for user-review statistics.

    Missing origin keeps the established legacy meaning. An explicit unknown,
    empty or null origin is not guessed to be a user. This does not authenticate
    the author; all original words and source declarations remain in evidence.
    """
    return 'origin' not in review or review['origin'] == 'user'


def _group(rows: list[dict], axis: str) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["axes"][axis]].append(row)
    result = []
    for value, values in sorted(buckets.items()):
        statuses = Counter(row["status"] for row in values)
        all_reviews = [review for row in values for review in row["reviews"]]
        reviews = [review for review in all_reviews if _is_user_review(review)]
        facets = Counter(facet for review in reviews for facet in (review.get("analysis") or {}).get("facets", []))
        positives = [text for review in reviews for text in (review.get("analysis") or {}).get("positive_excerpts", [])]
        problems = [text for review in reviews for text in (review.get("analysis") or {}).get("problem_excerpts", [])]
        warnings = Counter(
            str(check.get("name", "unknown")) for row in values for check in row["checks"]
            if str(check.get("status", "")).lower() in {"warning", "error", "failed"}
        )
        timings = [row["seconds"] for row in values if row["seconds"] is not None]
        completed = statuses.get("completed", 0)
        reviewed_images = sum(row['reviewed_images'] for row in values if row["status"] == "completed")
        image_count = sum(row['image_count'] for row in values if row['status'] == 'completed')
        result.append({
            "value": value, "count": len(values), "statuses": dict(statuses),
            "completion_rate": completed / len(values) if values else 0,
            "reviewed_images": reviewed_images,
            "image_count": image_count,
            "review_coverage": reviewed_images / image_count if image_count else 0,
            "review_count": len(reviews), "non_user_review_count": len(all_reviews) - len(reviews),
            "positive_mentions": len(positives),
            "problem_mentions": len(problems), "facets": dict(facets),
            "check_warnings": dict(warnings),
            "average_seconds": round(mean(timings), 2) if timings else None,
            "positive_examples": positives[:5], "problem_examples": problems[:5],
        })
    return result


def analyze_experiment(experiment, artwork_store) -> dict:
    rows = []
    for item in experiment.items:
        reviews = _item_reviews(item, artwork_store)
        user_reviews = [review for review in reviews if _is_user_review(review)]
        rows.append({
            "id": item.id, "job_id": item.job_id, "status": item.status,
            "axes": {axis: _axis(item, axis) for axis in AXES},
            "seconds": item.timings.get("total"), "checks": list(item.checks),
            "reviews": reviews, "image_count": len(item.images),
            "reviewed_images": len({r.get('image_index', 0) for r in user_reviews if isinstance(r.get('image_index', 0), int) and 0 <= r.get('image_index', 0) < len(item.images)}),
        })
    completed = sum(row["status"] == "completed" for row in rows)
    reviewed = sum(row['reviewed_images'] for row in rows if row["status"] == "completed")
    images = sum(row['image_count'] for row in rows if row['status'] == 'completed')
    all_reviews = [review for row in rows for review in row['reviews']]
    user_review_count = sum(_is_user_review(review) for review in all_reviews)
    return {
        "schema_version": "research-analysis-2",
        "experiment_id": experiment.id,
        "summary": {
            "total": len(rows), "completed": completed,
            "completed_images": images,
            "reviewed_images": reviewed,
            "review_coverage": reviewed / images if images else 0,
            "review_count": user_review_count,
            "non_user_review_count": len(all_reviews) - user_review_count,
            # Descriptive coverage cannot establish a winner or its scope.
            "conclusion_ready": False,
            "warning": "聚合只描述现有样本；评价覆盖率与关键词数量不能证明模型优劣或生产路线适用性。",
        },
        "groups": {axis: _group(rows, axis) for axis in AXES},
        "review_evidence": [
            {"item_id": row['id'], "axes": row['axes'], "review": review,
             "artwork_id": artwork_store.id_for(row['job_id'], review.get('image_index', 0)) if row['job_id'] and isinstance(review.get('image_index', 0), int) else None}
            for row in rows for review in row['reviews']
        ],
    }
