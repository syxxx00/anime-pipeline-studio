/* Presentation of retained interpretations, not a semantic correctness judge. */
(function(root){
  const labels={count:'数量',appearance:'外观',clothing:'服装',expression:'情绪／表情',body_action:'动作',left_hand:'左手',right_hand:'右手',framing:'景别',background:'环境',lighting:'光线',style:'画法',exclusions:'排除项'};
  const collections={conditions:'条件',objects:'对象',relations:'关系',goals:'目标',context:'背景缘由',uncertainties:'尚未确定',conflicts:'冲突报告'};
  function describe(value){
    if(value==null)return '—';
    if(Array.isArray(value))return value.map(describe).join('\n')||'（无）';
    if(typeof value!=='object')return String(value);
    if(value.predicate)return `${value.subject} → ${value.predicate} → ${value.object}`;
    if(value.value)return `${labels[value.field]||value.field||''}${value.field?'：':''}${value.value}`;
    if(value.description)return value.description;
    if(value.name)return [value.name,...(value.appearance||[]).map(describe),...(value.position||[]).map(describe)].join('；');
    return JSON.stringify(value,null,2); // Unknown records remain visible, never silently omitted.
  }
  function model(record){
    const revision=record?.source_reconsideration,preview=revision?.preview;
    if(!preview?.proposed_source_snapshot||!preview.receipt?.preview_sha256)return null;
    const annotations=revision.revision_art_raw?.change_annotations||[];
    return {candidate_sha256:preview.receipt.preview_sha256,original:record.input_snapshot.sentence,
      canAdopt:!(preview.proposed_source_snapshot.conflicts||[]).length,
      changes:(preview.actual_changes||[]).map((item,index)=>({...item,
        label:collections[item.collection]||item.collection,beforeText:describe(item.before),afterText:describe(item.after),
        annotation:annotations.find(note=>note.change_index===index)})),
      columns:[{title:'原先的解释',snapshot:preview.previous_snapshot},{title:'待采用的解释',snapshot:preview.proposed_source_snapshot}],
      collections};
  }
  function render(panel,record){
    const view=model(record);panel.hidden=!view;if(!view)return false;
    const doc=panel.ownerDocument,slot=panel.querySelector('[data-source-comparison]');slot.replaceChildren();
    const add=(parent,tag,text)=>{const el=doc.createElement(tag);el.textContent=text;parent.append(el);return el;};
    const original=add(slot,'details','');add(original,'summary','完整原句（保持不变）');add(original,'blockquote',view.original);
    if(!view.changes.length)add(slot,'p','本次候选没有改变解释；这不代表理解已核定。');
    for(const change of view.changes){
      const card=add(slot,'article','');card.className='source-change';add(card,'h4',change.label);
      const pair=add(card,'div','');pair.className='source-pair';
      for(const [title,text] of [['之前',change.beforeText],['候选',change.afterText]]){const column=add(pair,'div','');add(column,'strong',title);add(column,'p',text);}
      if(change.annotation){add(card,'blockquote',`原文：${change.annotation.source_quote}`);add(card,'p',`系统的理由（待核对）：${change.annotation.reason}`);}
    }
    const all=add(slot,'details','');add(all,'summary','查看完整解释：包括未改变的关系和背景');
    const pair=add(all,'div','');pair.className='source-pair';
    for(const column of view.columns){const section=add(pair,'section','');add(section,'h4',column.title);
      for(const [key,title] of Object.entries(collections)){add(section,'strong',title);add(section,'p',describe(column.snapshot?.[key]||[]));}}
    panel.querySelector('[data-source-adopt]').disabled=!view.canAdopt||['queued','running','cancelled'].includes(record.status);
    panel.querySelector('[data-source-clarify]').disabled=['queued','running','cancelled'].includes(record.status);
    panel.dataset.candidateSha256=view.candidate_sha256;
    return true;
  }
  root.SourceReview={describe,model,render};
  if(typeof module!=='undefined')module.exports=root.SourceReview;
})(globalThis);
