from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.semantic_patch import merge_semantic_patch

from transformers import AutoModelForCausalLM, AutoTokenizer


model_path = sys.argv[1]
payload = json.loads(sys.stdin.read())
fallback = payload["fallback"]
clauses = payload.get("clauses") or []
system = """Extract the visible scene from a Chinese or English anime image request.
Return JSON only with exactly these keys, all arrays of short concrete English phrases:
expression, body_action, left_hand, right_hand, objects, relations, framing, background, lighting, clause_map.
Objects use {id,name,appearance,position}; relations use {subject,predicate,object}.
Capture every explicit location, time of day, light source, pose, gaze and composition. Explicit framing takes priority over scene defaults: scenery does not automatically mean a wide shot.
Translate emotions into visible facial or body cues. Translate metaphors into visible behavior: for example 'like a statue' means completely motionless and composed, never stone skin or a literal sculpture.
Never add lore, symbolism, text, floating fragments or unexplained objects. Keep explicitly named objects (up to the schema limit of eight); do not silently omit them to satisfy an arbitrary two-object rule.
Preserve left/right ownership and hand gestures exactly. If one hand holds a phone while the other makes a V sign, never describe either both hands holding it or the other hand holding it. A request to omit something is not a request to add it. Do not translate closed eyes into a distant gaze.
Never invent a hand gesture, visible hand, held object, action or interaction. If the request does not mention hands, leave left_hand and right_hand empty. Still capture every explicitly named object, including objects in the background that nobody touches. Use an empty relations array when there is no explicit interaction. Missing information stays unspecified; do not complete it with a typical anime pose.
Relations are only for literal physical/spatial binding or gaze directed at an object declared in objects (holding, pressing, touching, sitting on, looking at). Never encode time, mood, metaphor, weather, or simile as a relation.
clause_map reports grounding: exactly one entry per input clause, same order, each {"clause": <verbatim clause text>, "status": "mapped"|"partial"|"unmapped", "fields": [keys of this JSON the clause informed], "note": "short reason"}.
status "unmapped" means the clause carries no visible renderable content you could write into the JSON. Never omit a clause."""
user = json.dumps({"sentence": payload["sentence"], "clauses": clauses}, ensure_ascii=False)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype="auto")
text = tokenizer.apply_chat_template(
    [{"role": "system", "content": system}, {"role": "user", "content": user}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tokenizer([text], return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=2000, do_sample=False)
answer = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
start, end = answer.find("{"), answer.rfind("}")
if start < 0 or end < start:
    raise RuntimeError("model did not return JSON")
patch = json.loads(answer[start:end + 1])
print(json.dumps(merge_semantic_patch(fallback, patch), ensure_ascii=False))
