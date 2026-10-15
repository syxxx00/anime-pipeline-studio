/* The compilation snapshot and user overrides form one request. No DOM access. */
(function (root) {
  function stable(value) {
    if (Array.isArray(value)) return value.map(stable);
    if (value && typeof value === 'object') return Object.fromEntries(
      Object.keys(value).sort().map(key => [key, stable(value[key])])
    );
    return value;
  }

  function resolveCompilation(compiled, editor) {
    if (!compiled) throw new Error('请先解析描述');
    const spec = editor ? JSON.parse(editor.spec) : compiled.spec;
    if (JSON.stringify(stable(spec)) !== JSON.stringify(stable(compiled.spec))) {
      throw new Error('语义规格已修改，请先点击“用规格重建模型格式”，使契约和所有模型提示词同步。');
    }
    const xml = editor ? editor.xml.trim() : compiled.xml;
    const negative = editor ? editor.negative.trim() : compiled.negative_prompt;
    if (!xml) throw new Error('正向提示词不能为空');
    // Also update the model-native channel read by routing and candidate creation.
    // Empty negative prompts are intentional and must not restore compiled text.
    const native = structuredClone(compiled.native_prompts || {});
    function applyOverride(target,positive,negative,manual=true){
      const previous=native[target]||{},changed=positive!==(previous.positive||'')||negative!==(previous.negative||'');
      native[target]={...previous,positive,negative};
      if(changed){
        // A compiler trace belongs to the compiled payload, not arbitrary edits.
        // Keep it as historical evidence, but do not present it as validation of
        // the new words or silently rewrite the semantic/text postprocess plan.
        native[target].compiled_trace=previous.compiled_trace||previous.trace||{};
        native[target].trace={};native[target].trace_status=manual?'unverified_manual_override':'unverified_legacy_alias_override';
        native[target].postprocess_status='compiled_plan_not_revalidated_after_prompt_edit';
        native[target].negative_policy_status='compiled_policy_not_reapplied_after_prompt_edit';
      }
    }
    for(const [target,override] of Object.entries(editor?.native||{})){
      if(!(target in native))throw new Error(`不存在模型提示词通道：${target}`);
      if(!override.positive.trim())throw new Error(`${target} 的正向提示词不能为空`);
      applyOverride(target,override.positive.trim(),override.negative.trim());
    }
    applyOverride('newbie',xml,negative,Boolean(editor));
    return {spec, xml, negative, native};
  }
  root.StudioRequest = {resolveCompilation};
  if (typeof module !== 'undefined') module.exports = root.StudioRequest;
})(globalThis);
