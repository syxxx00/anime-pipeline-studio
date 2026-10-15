"""CPU-only goal extraction. The parent process retains the verbatim input."""
import json
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    original = json.load(sys.stdin)['original_text']
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(sys.argv[1], local_files_only=True, torch_dtype='auto')
    system = '''Read the user's image request. Return JSON only: {"goals": [...], "interpretation_notes": [...]}.
Each goal: {"id": "short_ascii_id", "description": "Chinese", "priority": "required|preferred|context|proposal", "source_quote": "verbatim substring of the user input", "observable": "Chinese visual observation"}.
Keep the named subject, important objects, actions and relationships as separate required goals. Emotions are goals, not mandatory palettes. Reasons and character history are context, not visible objects or actions. Questions about composition and style allow creative proposals. Never invent lore, tools, objects, weather, clothing or poses. Quote the user exactly; do not paraphrase source_quote. Do not choose models or pipelines. This is an unverified draft, not proof of complete understanding.'''
    prompt = tokenizer.apply_chat_template([{'role':'system','content':system},
        {'role':'user','content':original}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer([prompt], return_tensors='pt')
    output = model.generate(**inputs, max_new_tokens=1400, do_sample=False)
    answer = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    start, end = answer.find('{'), answer.rfind('}')
    if start < 0 or end < start:
        raise ValueError('本地模型未返回任务目标 JSON')
    print(json.dumps(json.loads(answer[start:end+1]), ensure_ascii=False))


if __name__ == '__main__':
    main()
