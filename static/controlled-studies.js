(function(root){
  'use strict';
  const ui={data:null,active:null,revision:0,preview:null,token:null,loading:null,busy:false,overrides:{}};
  const node=id=>document.getElementById(id);
  const html=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const copy=value=>JSON.parse(JSON.stringify(value));
  function countJobs(id,cases,seeds,denoises=3,phase='full',routes=null){
    if(phase==='baseline')return cases*4;
    if(routes&&(!Array.isArray(routes)||routes.some(r=>!Array.isArray(r))))return NaN;
    const prefixes=new Set((routes||[]).flatMap(r=>r.map((_,i)=>JSON.stringify(r.slice(0,i+1)))));
    return cases*seeds*(id==='loops'?(routes?prefixes.size:76):id==='handoff'?4+16*denoises:8);
  }
  function routeDiagram(routes){
    const names={newbie:'NewBie',anima:'Anima',illustrious:'Illustrious',z_image_turbo:'Z-Image'};
    return '<details><summary>查看全部循环路线图（共享前缀只执行一次）</summary><div style="overflow:auto"><svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="循环路径图，按左到右执行" viewBox="0 0 760 '+(routes.length*50+35)+'" style="min-width:560px;width:100%"><title>循环路径：每一列是一个保留版本</title>'+routes.map((route,row)=>route.map((m,col)=>{
      const x=35+col*180,y=20+row*50;
      return (col?'<path d="M '+(x-25)+' '+(y+15)+' h 20 l -5 -4 m 5 4 l -5 4" fill="none" stroke="currentColor"/>':'')+'<rect x="'+x+'" y="'+y+'" width="145" height="32" rx="6" fill="none" stroke="currentColor"/><text x="'+(x+72)+'" y="'+(y+21)+'" text-anchor="middle" fill="currentColor" font-size="13">'+html(names[m]||m)+'</text>';
    }).join('')).join('')+'</svg></div></details>';
  }
  function updateActions(){
    const prepared=ui.preview,full=node('studyPhase').value==='full';
    const reusable=prepared?.result.baseline_candidates?.some(x=>x.id===prepared.design.baseline_experiment_id&&x.reusable);
    node('createStudy').disabled=ui.busy||!prepared||!node('studyConfirmed').checked||(full&&(!reusable||!node('studyBaselineReviewed').checked));
    let total=prepared?.result.total;
    if(total===undefined&&ui.active?.available){
      const cases=node('studyCases').querySelectorAll('[data-enabled]:checked').length;
      let routes=null;
      if(ui.active.id==='loops'){try{routes=JSON.parse(node('studyLoopRoutes').value);}catch(e){routes=[];}}
      total=countJobs(ui.active.id,cases,numbers('studySeeds').length,numbers('studyDenoises').length,full?'full':'baseline',routes);
    }
    const label=(full?'完整矩阵':'基线校验')+' · '+(Number.isFinite(total)?total:'待校验')+' 张';
    node('studyScope').textContent='本次：'+(ui.active?.name||'')+' / '+label+(prepared?'（已校验数量）':'（预计数量，需预览）');
    node('createStudy').textContent='创建'+label+'（暂停）';
    node('previewStudy').textContent='预览'+label;
  }
  function invalidate(){ui.revision++;ui.preview=null;ui.token=null;node('studyConfirmed').checked=false;node('studyBaselineReviewed').checked=false;node('previewStudy').disabled=false;node('studyPreviewNote').textContent='设计已修改，需要重新预览。';node('studyBaselineNote').textContent='预览时检查已有基线是否可复用；不会自动切回基线阶段。';updateActions();}
  function parse(text,label){try{return JSON.parse(text);}catch(e){throw new Error(label+' 的 JSON 无法解析：'+e.message);}}
  function numbers(id){return node(id).value.split(/[,，\s]+/).filter(Boolean).map(Number);}
  function readDesign(){
    const design=copy(ui.data.defaults);
    design.seeds=numbers('studySeeds');design.denoises=numbers('studyDenoises');
    design.width=Number(node('studyWidth').value);design.height=Number(node('studyHeight').value);
    design.phase=node('studyPhase').value;design.baseline_experiment_id=node('studyBaseline').value;
    if(ui.active.id==='loops')design.loop={routes:parse(node('studyLoopRoutes').value,'循环路线'),target_denoises:parse(node('studyLoopDenoises').value,'目标强度')};
    design.cases=design.cases.filter(c=>node('studyCases').querySelector('[data-case="'+c.id+'"] [data-enabled]').checked);
    for(const c of design.cases){
      const box=node('studyCases').querySelector('[data-case="'+c.id+'"]');
      c.intent=box.querySelector('[data-intent]').value;
      c.inspect=box.querySelector('[data-inspect]').value.split('\n').map(x=>x.trim()).filter(Boolean);
      for(const level of ['short','detailed']){
        c[level]=parse(box.querySelector('[data-contract="'+level+'"]').value,c.name+' '+level);
        c[level].source_text=c.intent;
      }
      c.prompt_overrides=copy(ui.overrides[c.id]||{});
    }
    for(const [model,value] of Object.entries(design.models)){
      const box=node('studyModels').querySelector('[data-model="'+model+'"]');
      value.parameters=parse(box.querySelector('[data-parameters]').value,model+' 参数');
      value.negative=box.querySelector('[data-negative]').value;
      value.empty_negative_reason=box.querySelector('[data-negative-reason]').value;
      value.baseline_source=box.querySelector('[data-baseline-source]').value;
    }
    return design;
  }
  function contractEditor(c,level){
    const value=c[level],summary=value.entities.map(e=>e.name).join(' · ');
    return '<fieldset><legend>'+(level==='short'?'简略场景要求':'完整场景要求')+'</legend><p>'+html(summary)+'</p><label>完整可编辑场景契约（JSON）<textarea data-contract="'+level+'" spellcheck="false">'+html(JSON.stringify(value,null,2))+'</textarea></label><small>entities：角色/道具/环境；constraints：外观、服装、动作；relations：接触/空间关系；composition：构图；style：画风；global_exclusions：禁止项。修改后需要重新预览实际文本，不自动调用语言模型。</small></fieldset>';
  }
  function modelEditor(model,value){
    return '<details data-model="'+model+'"><summary>'+html(model)+' · 独立模型配置</summary><label>采样、权重精度与 LoRA（JSON，完整生效）<textarea data-parameters spellcheck="false">'+html(JSON.stringify(value.parameters,null,2))+'</textarea></label><label>负面基线<textarea data-negative>'+html(value.negative)+'</textarea></label><label>配置来源<input data-baseline-source value="'+html(value.baseline_source)+'"></label><label>若主动清空负面词，说明实验理由<input data-negative-reason value="'+html(value.empty_negative_reason)+'"></label><small>空负面词不会被自动恢复；CFG=1 时通常不使用负面条件。NewBie / Illustrious 可用兼容 LoRA；未实现的模型适配器会明确报错，不会假装加载。</small></details>';
  }
  async function baselineOptions(){
    const load=ui.baselineLoad=(ui.baselineLoad||0)+1;
    try{
      const values=await api('/api/experiments');
      if(load!==ui.baselineLoad)return;
      const selected=node('studyBaseline').value;
      node('studyBaseline').innerHTML='<option value="">选择已运行并查看的基线实验</option>'+values.filter(x=>x.design?.phase==='baseline').map(x=>'<option value="'+html(x.id)+'">'+html(x.name)+' · '+html(x.state)+'</option>').join('');
      node('studyBaseline').value=selected;
    }catch(error){node('studyStatus').textContent='基线列表读取失败：'+error.message;}
  }
  async function choose(id){
    ui.active=ui.data.studies.find(s=>s.id===id);ui.revision++;ui.preview=null;ui.token=null;ui.overrides={};
    const s=ui.active;node('studyEditor').hidden=false;node('studyEditorTitle').textContent=s.name;
    node('studyQuestion').textContent=s.question;node('studyControls').textContent='改变：'+s.change+'\n固定：'+s.hold;
    node('studyLimits').textContent=s.limits;node('studyExecution').hidden=!s.available;node('studyCases').hidden=!s.available;
    node('studyConfirmed').checked=false;node('studyBaselineReviewed').checked=false;node('createStudy').disabled=true;node('previewStudy').disabled=false;
    node('studyPreview').replaceChildren();node('studyPreviewNote').textContent='先预览，审阅完整原生文本、有效参数与实际编码节点。';
    node('studyStatus').textContent=s.available?'默认创建所选项目的完整矩阵；预览会检查可复用基线。仅需校验配置时，单独选择基线阶段。':'此项尚不可执行；其他可执行研究项目不受影响。';
    node('studySeeds').value=ui.data.defaults.seeds.join(', ');node('studyDenoises').value=ui.data.defaults.denoises.join(', ');
    node('studyWidth').value=ui.data.defaults.width;node('studyHeight').value=ui.data.defaults.height;node('studyPhase').value='full';node('studyBaseline').value='';
    node('studyLoopControls').hidden=s.id!=='loops';node('studyDenoisesLabel').hidden=s.id!=='handoff';
    const loops=ui.data.defaults.loop||{routes:ui.data.loop_presets?.pairs||[],target_denoises:{newbie:.15,anima:.30,illustrious:.30,z_image_turbo:.30}};
    node('studyLoopRoutes').value=JSON.stringify(loops.routes,null,2);node('studyLoopDenoises').value=JSON.stringify(loops.target_denoises,null,2);
    node('studyLoopPreset').value=ui.data.defaults.loop?'custom':'pairs';
    node('studyBaselineNote').textContent='预览时检查已有基线是否可复用；不会自动切回基线阶段。';
    node('studyCases').innerHTML=ui.data.defaults.cases.map(c=>'<details data-case="'+html(c.id)+'"><summary>'+html(c.name)+'</summary><label><input type="checkbox" data-enabled checked>纳入本次实验（保留全部模型/方向）</label><label>中文意图<textarea data-intent>'+html(c.intent)+'</textarea></label><label>人工观察项目（一行一项）<textarea data-inspect>'+html(c.inspect.join('\n'))+'</textarea></label><div class="study-text-columns">'+['short','detailed'].map(l=>contractEditor(c,l)).join('')+'</div></details>').join('');
    node('studyModels').innerHTML='<h4>各模型独立配置</h4>'+Object.entries(ui.data.defaults.models).map(([m,v])=>modelEditor(m,v)).join('');
    for(const id of ['studyCases','studyModels'])node(id).querySelectorAll('input,textarea').forEach(n=>n.addEventListener('input',invalidate));
    if(s.id!=='detail')node('studyCases').querySelectorAll('fieldset').forEach(n=>{if(n.querySelector('[data-contract="short"]'))n.hidden=true;});
    updateActions();await baselineOptions();
  }
  async function load(){
    if(ui.data)return;if(ui.loading)return ui.loading;
    ui.loading=(async()=>{try{
      ui.data=await api('/api/research/studies');
      node('controlledStudyCards').innerHTML=ui.data.studies.map(s=>'<article><h3>'+html(s.name)+'</h3><p>'+html(s.question)+'</p><small>'+(s.available?'默认完整矩阵 '+s.default_images+' 张 · 已有基线可复用':'执行器缺项 · 可查看设计')+'</small><button data-study="'+html(s.id)+'">'+(s.available?'查看与调整实验':'查看缺项')+'</button></article>').join('');
      node('controlledStudyCards').querySelectorAll('[data-study]').forEach(b=>b.onclick=()=>choose(b.dataset.study));
      node('studiesLoadStatus').textContent='完整输入、独立模型设置、基线校验；创建暂停，进度、对照与自然语言评价沿用研究工作区。';
    }catch(error){node('studiesLoadStatus').textContent='项目读取失败：'+error.message+'。可点击重新读取。';}
    finally{ui.loading=null;}})();return ui.loading;
  }
  function rememberOverride(row,element){
    const {scene,model,prompt_form:variant}=row.axes;
    const target=((ui.overrides[scene]??={})[variant]??={})[model]??={note:''};
    if(element.hasAttribute('data-edit-positive'))target.positive=element.value;
    if(element.hasAttribute('data-edit-negative'))target.negative=element.value;
    if(element.hasAttribute('data-edit-note'))target.note=element.value;
    invalidate();
  }
  async function previewStudy(){
    if(!ui.active?.available||ui.busy)return;
    const id=ui.active.id,revision=++ui.revision;let design;
    try{design=readDesign();}catch(error){node('studyStatus').textContent=error.message;return;}
    ui.preview=null;ui.token=null;ui.busy=true;node('createStudy').disabled=true;node('previewStudy').disabled=true;node('studyConfirmed').checked=false;
    node('studyStatus').textContent='正在校验完整条件与执行图，不调用生图模型…';
    try{
      let result=await api('/api/research/studies/'+id+'/preview',{method:'POST',body:JSON.stringify(design)});
      if(revision!==ui.revision||ui.active.id!==id)return;
      if(!Array.isArray(result.baseline_candidates))throw new Error('后端尚未加载基线复用更新，请重启服务；当前不会创建实验。');
      // A late list request must not replace the verified selection.
      ui.baselineLoad=(ui.baselineLoad||0)+1;
      const candidates=result.baseline_candidates,matching=candidates.filter(x=>x.reusable);
      node('studyBaseline').innerHTML='<option value="">尚未选择基线</option>'+candidates.map(x=>'<option value="'+html(x.id)+'">'+html(x.name)+' · '+html(x.id)+' · '+(x.reusable?'可复用':html(x.reason))+'</option>').join('');
      let selected=design.baseline_experiment_id;
      if(!selected&&design.phase==='full'&&matching.length)selected=matching[0].id;
      node('studyBaseline').value=selected;
      if(selected!==design.baseline_experiment_id){
        design.baseline_experiment_id=selected;
        node('studyBaselineReviewed').checked=false;
        result=await api('/api/research/studies/'+id+'/preview',{method:'POST',body:JSON.stringify(design)});
        if(revision!==ui.revision||ui.active.id!==id)return;
      }
      const chosen=result.baseline_candidates.find(x=>x.id===selected);
      node('studyBaselineNote').textContent=design.phase==='baseline'
        ?(matching.length?'已有 '+matching.length+' 个可复用基线；继续创建会重复校验。可切换“完整矩阵”使用已有结果。':'本次仅校验配置，不包含完整对照或模型交接。')
        :(chosen?.reusable?'已匹配基线 '+selected+'，无需重跑。请确认已查看图片。':chosen?'所选基线不可复用：'+chosen.reason:'没有可复用基线。可从历史基线恢复配置，或显式选择“基线校验”创建；本次仍保持完整矩阵。');
      ui.preview={result,design};ui.token=crypto.randomUUID?crypto.randomUUID():Array.from(crypto.getRandomValues(new Uint8Array(16)),n=>n.toString(16).padStart(2,'0')).join('');
      node('studyPreviewNote').textContent=result.total+' 个采样作业：'+result.source_jobs+' 个直接生成，'+result.handoff_jobs+' 个'+(id==='loops'?'逐步交接（每一步接收直接父版本）':'同源交接')+'。'+(design.phase==='baseline'?'这是基线校验，不是完整矩阵。':'这是完整矩阵。');
      const seen=new Set();
      const rows=result.snapshot.manifest.items.filter(x=>{const key=JSON.stringify([x.axes.scene,x.axes.model,x.axes.prompt_form]);if(seen.has(key)||x.pixel_source)return false;seen.add(key);return true;});
      node('studyPreview').innerHTML='<p>'+result.warnings.map(html).join('<br>')+'</p>'+(id==='loops'&&design.phase==='full'?routeDiagram(design.loop.routes):'')+rows.map((x,index)=>{
        const key=x.axes.scene+':'+x.axes.model+':'+x.axes.prompt_form;
        const execution=result.snapshot.execution[key], override=ui.overrides[x.axes.scene]?.[x.axes.prompt_form]?.[x.axes.model];
        return '<details data-preview-row="'+index+'"><summary>'+html(key)+'</summary><label>实际正向文本（可以直接编辑）<textarea data-edit-positive spellcheck="false">'+html(x.request.prompt)+'</textarea></label><label>实际负向文本（可以直接编辑）<textarea data-edit-negative>'+html(x.request.negative_prompt)+'</textarea></label><label>直接改写的理由（编辑上方文本后必填）<input data-edit-note value="'+html(override?.note||'')+'"></label><button data-reset-override>清除该条件的手工覆盖</button><details><summary>实际编码节点与完整执行工作流</summary><pre>'+html(JSON.stringify(execution.workflow,null,2))+'</pre></details><small>steps '+x.request.steps+' · CFG '+x.request.cfg+' · '+html(x.request.sampler)+' / '+html(x.request.scheduler)+' · 权重精度 '+html(x.request.weight_dtype??'checkpoint default')+'</small></details>';
      }).join('');
      node('studyPreview').querySelectorAll('[data-preview-row]').forEach(box=>{
        const row=rows[Number(box.dataset.previewRow)];
        box.querySelectorAll('[data-edit-positive],[data-edit-negative],[data-edit-note]').forEach(n=>n.oninput=()=>rememberOverride(row,n));
        box.querySelector('[data-reset-override]').onclick=()=>{delete ui.overrides[row.axes.scene]?.[row.axes.prompt_form]?.[row.axes.model];invalidate();previewStudy();};
      });
      if(result.snapshot.handoff_templates){
        node('studyPreview').insertAdjacentHTML('beforeend','<details><summary>交接执行图模板：目标 VAE、采样与强度（源图运行时绑定）</summary>'+Object.entries(result.snapshot.handoff_templates).map(([key,x])=>'<details><summary>'+html(key)+'</summary><p>'+html(x.binding)+'</p><pre>'+html(JSON.stringify(x.workflow,null,2))+'</pre></details>').join('')+'</details>');
      }
      node('studyStatus').textContent='预览已就绪。请审阅实际条件；修改文本后再次预览。确认只创建暂停任务。';
    }catch(error){if(revision===ui.revision)node('studyStatus').textContent=error.message;}
    finally{ui.busy=false;if(revision===ui.revision){node('previewStudy').disabled=false;updateActions();}}
  }
  async function createStudy(){
    updateActions();if(node('createStudy').disabled)return;
    const id=ui.active.id,revision=ui.revision,prepared=ui.preview,token=ui.token;ui.busy=true;node('createStudy').disabled=true;
    try{
      const result=await api('/api/research/studies/'+id+'/create',{method:'POST',body:JSON.stringify({design:prepared.design,digest:prepared.result.digest,confirmed:true,creation_token:token,baseline_reviewed:node('studyBaselineReviewed').checked})});
      if(revision!==ui.revision)return;
      node('studyStatus').textContent='已创建暂停实验：'+result.name+'。到“进度与结果”点击继续才开始。';
      await loadResearch();if(revision!==ui.revision)return;
      node('experimentSelect').value=result.id;await loadExperiment();selectLab('run');
    }catch(error){if(revision===ui.revision)node('studyStatus').textContent='创建未确认成功：'+error.message+'。重试会复用同一创建标识。';}
    finally{ui.busy=false;if(revision===ui.revision)updateActions();}
  }
  async function restoreDesign(raw,studyId){
    // Validate the complete imported snapshot before replacing the editor.
    const result=await api('/api/research/studies/'+studyId+'/preview',{method:'POST',body:JSON.stringify(raw)});
    const saved=result.snapshot.design;
    ui.data.defaults=copy(saved);await choose(studyId);
    ui.overrides=Object.fromEntries(saved.cases.map(c=>[c.id,copy(c.prompt_overrides||{})]));
    node('studyPhase').value=saved.phase;
    node('studyBaseline').value=saved.baseline_experiment_id||'';
    invalidate();node('studyStatus').textContent='完整设计已恢复，包含原生文本覆盖与各模型参数；请重新预览、确认。';
  }
  root.ControlledStudies={load,countJobs};
  if(typeof module!=='undefined')module.exports={countJobs};
  if(typeof document==='undefined')return;
  node('reloadStudies').onclick=()=>{ui.data=null;return load();};node('previewStudy').onclick=previewStudy;node('createStudy').onclick=createStudy;
  node('studyConfirmed').onchange=updateActions;node('studyBaselineReviewed').onchange=updateActions;
  node('exportStudyDesign').onclick=()=>{try{node('studyDesignJson').value=JSON.stringify(readDesign(),null,2);}catch(error){node('studyStatus').textContent=error.message;}};
  node('importStudyDesign').onclick=async()=>{try{await restoreDesign(parse(node('studyDesignJson').value,'实验设计'),ui.active.id);}catch(error){node('studyStatus').textContent=error.message;}};
  node('restoreBaselineDesign').onclick=async()=>{try{
    const studyId=ui.active.id,revision=ui.revision;
    const id=node('studyBaseline').value;if(!id)throw new Error('请先选择基线实验');
    const run=await api('/api/experiments/'+encodeURIComponent(id));
    if(revision!==ui.revision)return;
    if(run.design?.phase!=='baseline'||!run.design?.input)throw new Error('该实验没有可恢复的基线设计');
    const raw=copy(run.design.input),native={format:'structured',detail:'detailed',handoff:'source',loops:'source'};
    if(studyId==='loops')raw.loop={routes:parse(node('studyLoopRoutes').value,'循环路线'),target_denoises:parse(node('studyLoopDenoises').value,'目标强度')};
    if(studyId!==run.design.study_id){
      for(const c of raw.cases){
        const overrides=c.prompt_overrides||{},source=overrides[native[run.design.study_id]];
        c.prompt_overrides=source?{[native[studyId]]:copy(source)}:{};
      }
    }
    await restoreDesign({...raw,phase:'full',baseline_experiment_id:id},studyId);
    node('studyStatus').textContent='已恢复基线配置，保留当前实验“'+ui.active.name+'”；完整矩阵需重新预览并确认。';
  }catch(error){node('studyStatus').textContent=error.message;}};
  for(const id of ['studySeeds','studyWidth','studyHeight','studyDenoises','studyPhase','studyBaseline'])node(id).addEventListener('input',invalidate);
  for(const id of ['studyLoopRoutes','studyLoopDenoises'])node(id).addEventListener('input',()=>{node('studyLoopPreset').value='custom';invalidate();});
  node('studyLoopPreset').onchange=()=>{const routes=ui.data.loop_presets?.[node('studyLoopPreset').value];if(routes){node('studyLoopRoutes').value=JSON.stringify(routes,null,2);invalidate();}};
  document.addEventListener('DOMContentLoaded',load);
})(globalThis);
