/* Explicit cooperative execution; existing research manifests are untouched. */
(function(root){
  const key='pipeline-studio.current-cooperative-run';
  const labels={candidate_newbie:'NewBie',candidate_anima:'Anima',candidate_ill:'Illustrious',fusion_ill_newbie:'Illustrious → NewBie'};
  function describe(run){
    const pending=run.pending_nodes||[],status=run.state.status,paused=Boolean(run.pause_reason);
    const terminal=['accepted','cancelled','failed'].includes(status);
    const executable=pending.some(x=>x in labels||['region_repair','component_check'].includes(x));
    const idle=status==='needs_user'&&!paused&&!run.repair_job_id&&['compare','human_gate','accept','text_render'].includes(pending[0]);
    return {terminal,paused,idle,canPause:status==='running'&&executable,
      canResume:status==='needs_user'&&paused&&executable,canSelect:idle,
      canAccept:idle&&['human_gate','accept'].includes(pending[0]),
      canRollback:idle&&Boolean(run.selected_checkpoint_artifact_id)&&run.selected_checkpoint_artifact_id!==run.selected_artifact_id,
      title:paused?'已暂停后续派发':({running:'执行中',needs_user:'等待你的操作',accepted:'已接受',cancelled:'已取消',failed:'没有可用候选'})[status]||status};
  }
  root.CooperativeState={describe,key};
  if(typeof module!=='undefined')module.exports=root.CooperativeState;
  if(typeof document==='undefined')return;
  let current=null,busy=false,renderKey='',revision=0,trackedId='';
  const panel=document.createElement('section');panel.className='cooperative-panel hidden';panel.id='cooperativePanel';
  panel.innerHTML=`<header class="cooperative-head"><h2>候选与协作任务</h2><button id="closeCooperative">返回单图视图</button></header>
    <label>已有任务 <select id="cooperativeRunList"><option value="">选择任务</option></select></label> <button id="reloadCooperativeList">刷新列表</button>
    <p id="cooperativeStatus" class="cooperative-status" role="status"></p><p id="cooperativeConnection" role="status"></p><p id="cooperativeOperation" role="status"></p>
    <div class="cooperative-actions"><button data-coop-action="pause">暂停后续派发</button><button data-coop-action="resume">继续原任务</button><button data-coop-action="cancel">取消任务</button><button id="refreshCooperative">重新读取</button><button id="cooperativeGraph">查看任务图</button></div>
    <div id="cooperativeCandidates" class="cooperative-candidates"></div><section id="cooperativeSelection"></section>
    <details><summary>任务记录与检查边界</summary><p>检查报告不等于已验证角色、手部或审美正确。局部修改仍需比较确认；当前不是任意模型反复迭代。</p><pre id="cooperativeHistory"></pre></details>`;
  document.querySelector('.canvas-stage').after(panel);
  const open=document.createElement('button');open.id='openCooperative';open.textContent='协作任务';document.querySelector('.workspace-actions').prepend(open);
  const hideSingle=hidden=>{document.querySelector('.canvas-stage').classList.toggle('hidden',hidden);document.querySelector('.creative-area').classList.toggle('cooperative-active',hidden);};
  const note=(text,bad=false)=>{ $('cooperativeConnection').textContent=text;$('cooperativeConnection').classList.toggle('bad',bad); };
  function show(){jobWatcher.stop();panel.classList.remove('hidden');hideSingle(true);}
  function single({forget=false}={}){revision++;watcher.stop();panel.classList.add('hidden');hideSingle(false);if(forget)try{localStorage.removeItem(key);}catch{}}
  function selected(id){if(!current||current.id!==id)throw new Error('协作任务已切换，请重新打开对应任务。');return current;}
  function card(art,title,extra=''){
    return `<article class="cooperative-card ${art.id===current?.selected_artifact_id?'selected':''}"><h3>${esc(title)}</h3>${art.metadata?.image_url?`<img src="${esc(art.metadata.image_url)}" alt="${esc(title)}">`:''}<small>${esc(art.id)}<br>${esc(art.path||'')}</small><button data-coop-art="${esc(art.id)}">查看 / 评价 / 修改</button>${extra}</article>`;
  }
  function render(run,jobs={}){
    current=run;const flags=describe(run);
    const list=$('cooperativeRunList');let option=[...list.options].find(x=>x.value===run.id);
    if(!option){option=new Option('',run.id);list.append(option);}option.textContent=`${run.id} · ${flags.title} · ${run.state.contract.source_text}`;list.value=run.id;
    $('cooperativeStatus').textContent=`${run.id} · ${flags.title} · ${run.pending_nodes.join(' / ')||'没有待执行节点'}\n${flags.paused?'已提交的任务可能仍在采样；继续后读取原任务，不重复生成。':'候选固定为三个单模型与一个 Illustrious → NewBie 组合，失败候选不会阻止其他候选结束。'}`;
    for(const button of panel.querySelectorAll('[data-coop-action]')){
      const action=button.dataset.coopAction;button.disabled=busy||!({pause:flags.canPause,resume:flags.canResume,cancel:!flags.terminal,accept:flags.canAccept,rollback:flags.canRollback})[action];
    }
    $('cooperativeGraph').disabled=false;
    const signature=JSON.stringify([run,jobs,busy]);if(signature===renderKey)return;renderKey=signature;
    $('cooperativeCandidates').innerHTML=Object.entries(run.candidates).map(([node,slot])=>{
      const job=jobs[run.candidate_job_ids[node]],detail=job?.status?`${job.status} · ${job.stage||''}${Number.isFinite(job.progress)?` · ${Math.round(job.progress*100)}%`:''}`:run.candidate_job_ids[node]?'已提交，等待任务状态':slot.status;
      if(slot.artifact)return card(slot.artifact,labels[node]||node,`<button data-coop-select="${esc(slot.artifact.id)}" ${busy||!flags.canSelect?'disabled':''}>选择此版本</button>`);
      return `<article class="cooperative-card"><h3>${esc(labels[node]||node)}</h3><p>${esc(detail)}</p><small>${esc(run.candidate_job_ids[node]||'尚未提交')}</small><p>${esc(slot.error||job?.observation_error||'')}</p></article>`;
    }).join('');
    const artifacts=run.state.artifacts||[],active=artifacts.find(x=>x.id===run.selected_artifact_id),checkpoint=artifacts.find(x=>x.id===run.selected_checkpoint_artifact_id);
    $('cooperativeSelection').innerHTML=active?`<h3>当前版本与检查点</h3><div class="cooperative-versions">${card(active,'当前选中版本')}${checkpoint&&checkpoint.id!==active.id?card(checkpoint,'可回退的检查点'):''}</div>
      <div class="cooperative-actions"><button data-coop-action="accept" ${busy||!flags.canAccept?'disabled':''}>接受当前版本</button><button data-coop-action="rollback" ${busy||!flags.canRollback?'disabled':''}>回到检查点</button></div>
      ${(run.state.contract.text_intents||[]).filter(x=>x.mode==='exact'&&!run.fulfilled_text_intent_ids.includes(x.id)).map(x=>`<p>待排文字：${esc(x.content)} <button data-coop-text="${esc(x.id)}" ${busy||!flags.idle?'disabled':''}>设置文字位置</button></p>`).join('')}
      ${run.repair_job_id?`<p>局部修改：${esc(run.repair_job_id)} · ${esc(jobs[run.repair_job_id]?.stage||jobs[run.repair_job_id]?.status||'等待状态')}</p>`:''}
      <p>${esc((run.state.open_defects||[]).map(x=>x.description).join('\n')||'没有已记录的未解决缺陷；这不是画面正确的证明。')}</p>`:'';
    $('cooperativeHistory').textContent=(run.state.history||[]).join('\n');
  }
  const watcher=JobWatcher.create({
    async readJob(id,signal){
      const run=await readTracked(`/api/adaptive-runs/${encodeURIComponent(id)}`,signal);
      const ids=[...new Set([...Object.values(run.candidate_job_ids||{}),run.repair_job_id].filter(Boolean))];
      const entries=await Promise.all(ids.map(async job=>{try{return [job,await readTracked(`/api/jobs/${encodeURIComponent(job)}`,signal)];}catch(error){return[job,{observation_error:error.message}];}}));
      return {id,status:run.state.status==='accepted'?'completed':run.state.status,run,jobs:Object.fromEntries(entries)};
    },
    onJob(value){render(value.run,value.jobs);},onConnection(value){note(value?`读取暂时失败，${Math.ceil(value.retryMs/1000)} 秒后重试；不会重新提交生成。`:'',Boolean(value));},
    onMissing(id){note(`服务找不到任务 ${id}。已停止自动读取；可手动重试，不会重新生成。`,true);},
    onRenderError(error){note(`任务已读取，但显示失败：${error.message}`,true);},interval:2000,
  });
  async function watch(id){if(!/^[a-zA-Z0-9_-]{1,100}$/.test(id))throw new Error('任务编号无效');revision++;trackedId=id;show();renderKey='';
    if(current?.id!==id){current=null;$('cooperativeCandidates').replaceChildren();$('cooperativeSelection').replaceChildren();$('cooperativeStatus').textContent=`正在读取 ${id}…`;panel.querySelectorAll('[data-coop-action]').forEach(b=>b.disabled=true);}
    try{localStorage.setItem(key,id);}catch{}return watcher.watch(id);}
  async function list(){try{const version=revision,runs=await api('/api/adaptive-runs');if(version!==revision)return;
    $('cooperativeRunList').replaceChildren(new Option('选择任务',''),...runs.map(x=>new Option(`${x.id} · ${describe(x).title} · ${x.state.contract.source_text}`,x.id)));
    if(current)$('cooperativeRunList').value=current.id;
  }catch(error){note(`列表读取失败：${error.message}`,true);}}
  async function action(name,payload={}){
    if(!current||busy)return;
    const id=current.id,version=revision;busy=true;watcher.stop();render(current);$('cooperativeOperation').textContent='正在提交操作…';
    try{const run=await api(`/api/adaptive-runs/${encodeURIComponent(id)}/${name}`,{method:'POST',body:JSON.stringify(payload)});if(version===revision&&current?.id===id){render(run);$('cooperativeOperation').textContent='操作已保存。';}}
    catch(error){if(version===revision)$('cooperativeOperation').textContent=`操作未确认：${error.message}。请读取任务状态后再决定，不会自动重发操作。`;}
    finally{busy=false;if(version===revision&&current?.id===id){render(current);await watcher.watch(id);}}
  }
  async function openArt(id,intent=null){
    const runId=current?.id;if(!runId)return;
    await openArtwork(id,{runId,intentId:intent?.id||''});
    if(intent&&state.artwork?.id===id){$('exactText').value=intent.content;document.querySelector('[data-tool="text"]').click();}
  }
  panel.addEventListener('click',event=>{const b=event.target.closest('button');if(!b)return;
    if(b.dataset.coopAction){if(b.dataset.coopAction==='cancel'&&!confirm('取消此协作任务及它已提交的任务？已保存图片不会删除。'))return;action(b.dataset.coopAction);}
    else if(b.dataset.coopSelect)action('select',{artifact_id:b.dataset.coopSelect,locked_refs:[]});
    else if(b.dataset.coopArt)openArt(b.dataset.coopArt);
    else if(b.dataset.coopText){const intent=current.state.contract.text_intents.find(x=>x.id===b.dataset.coopText);openArt(current.selected_artifact_id,intent);}
  });
  $('openCooperative').onclick=()=>{show();if(current)watch(current.id);list();};
  $('closeCooperative').onclick=()=>{single({forget:true});if(state.job?.id)pollJob(state.job.id);};
  $('reloadCooperativeList').onclick=list;$('cooperativeRunList').onchange=()=>{const id=$('cooperativeRunList').value;if(id)watch(id);};
  $('refreshCooperative').onclick=()=>{if(trackedId)watch(trackedId);else list();};
  $('cooperativeGraph').onclick=()=>{if(current){PipelineDiagram.mountEvidence($('graphContent'),{graph:current.graph,state:current.state});$('graphDialog').showModal();}};
  $('generationStrategy').onchange=()=>{updateConfigContext();root.CooperativeUI.updateSettings();};
  root.CooperativeUI={
    requested:()=>$('proMode').checked&&$('generationStrategy').value==='cooperative',
    updateSettings(){
      const enabled=root.CooperativeUI.requested();$('cooperativeSettingsNote').classList.toggle('hidden',!enabled);
      $('quality').disabled=enabled;
      if(enabled){for(const name of ['batch_size','highres_scale','highres_steps','highres_denoise','face_detail_enabled','face_detail_steps','face_detail_denoise','check_enabled']){const field=$(`setting-${name}`);if(field)field.disabled=true;}}
    },
    begin:()=>single(),single:()=>single({forget:true}),
    async launch(request){
      if(!request.scene_contract)throw new Error('缺少当前语义契约，请重新解析。');
      const run=await api('/api/adaptive-runs/launch',{method:'POST',body:JSON.stringify({contract:request.scene_contract,request})});
      if($('promptDialog').open)$('promptDialog').close();render(run);watch(run.id);list();
    },
    async preview(request){const graph=await api('/api/v3/graph/preview',{method:'POST',body:JSON.stringify(request.scene_contract)});PipelineDiagram.mountEvidence($('graphContent'),graph);$('graphDialog').showModal();},
    editContext(artworkId){
      const context=state.artworkContext;if(!context?.runId)return {};
      const run=selected(context.runId);
      if(run.selected_artifact_id!==artworkId||!describe(run).idle)throw new Error('请先在协作任务中选择此版本，并等待当前阶段结束，再进行修改。');
      return {adaptive_run_id:run.id,...(context.intentId?{intent_id:context.intentId}:{})};
    },
    async review(id,text,context){
      const run=await api(`/api/adaptive-runs/${encodeURIComponent(context.runId)}/review`,{method:'POST',body:JSON.stringify({artifact_id:id,text})});
      if(current?.id===run.id)await watch(run.id);
      return run.saved_review;
    },
    refresh:async id=>{if(current?.id===id)await watch(id);},
  };
  window.addEventListener('DOMContentLoaded',()=>{
    let id=new URLSearchParams(location.search).get('run');
    if(!id&&(root.CreativeSessionUI?.hasSaved()||new URLSearchParams(location.search).get('session')))return;
    if(!id)try{id=localStorage.getItem(key);}catch{}
    if(id)watch(id).catch(error=>note(error.message,true));
  },{once:true});
})(globalThis);
