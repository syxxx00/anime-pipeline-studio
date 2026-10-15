"""Draft goals from raw input without replacing it with the model's summary."""
import json
import subprocess
import hashlib

from .config import PROJECT_ROOT
from .creation_intent import CreationIntent


def normalize_goal_draft(original_text, draft):
    goals = []
    for index, row in enumerate(draft.get('goals', []), 1):
        # IDs belong to the application, not to the language model's wording.
        goals.append({**row, 'id': f'goal_{index}'})
    intent = CreationIntent(original_text=original_text, goals=goals,
        interpretation_notes=draft.get('interpretation_notes', []),
        interpretation_method='local_qwen3_1_7b_unverified_draft')
    if not intent.goals or any(not g.source_quote for g in intent.goals if g.priority == 'required'):
        raise ValueError('目标提取缺少目标或原文引用，需要审阅，不能视为完整理解')
    return intent


def interpret_task(original_text, settings):
    result = subprocess.run([settings.semantic_python, str(PROJECT_ROOT / 'scripts' / 'intent_infer.py'),
        settings.semantic_model_path], input=json.dumps({'original_text': original_text}, ensure_ascii=False),
        text=True, encoding='utf-8', capture_output=True, check=True, timeout=360)
    draft = json.loads(result.stdout.strip().splitlines()[-1])
    digest = hashlib.sha256(original_text.encode('utf-8')).hexdigest()
    folder = PROJECT_ROOT / 'data' / 'task_intents'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{digest}.json').write_text(json.dumps({'original_text': original_text,
        'original_sha256': digest, 'unverified_model_draft': draft}, ensure_ascii=False, indent=2), encoding='utf-8')
    # The model never supplies the authoritative source or its hash.
    return normalize_goal_draft(original_text, draft)
