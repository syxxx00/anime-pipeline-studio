const $=id=>document.getElementById(id);const state={view:'create',lab:'run',compiled:null,job:null,artwork:null,knowledge:null,knowledgeOn:false,styles:[],experiments:[],experiment:null,boundary:null,checks:null,poll:null,drawing:false};
async function api(path,options={}){const response=await fetch(path,{headers:{'Content-Type':'application/json'},...options});if(!response.ok){const error=new Error(await response.text());error.status=response.status;throw error;}return response.json()}
const esc=value=>String(value??'').replace(/[&<>"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]));
function message(id,text,bad=false,source='protected'){const el=$(id);el.textContent=text;el.dataset.messageSource=source;el.classList.remove('hidden');el.classList.toggle('bad',bad)}
function selectView(name){state.view=['research','library'].includes(name)?name:'create';document.querySelectorAll('.view').forEach(el=>el.classList.toggle('active',el.id===`${state.view}View`));document.querySelectorAll('[data-view]').forEach(el=>el.classList.toggle('active',el.dataset.view===state.view));history.replaceState(null,'',state.view==='create'?'/':`?view=${state.view}`);if(state.view==='research')loadResearch();if(state.view==='library')loadArtworks()}
function selectLab(name){state.lab=['design','run','analysis'].includes(name)?name:'run';document.querySelectorAll('.lab').forEach(el=>el.classList.toggle('active',el.id===`${state.lab}Lab`));document.querySelectorAll('[data-lab]').forEach(el=>el.classList.toggle('active',el.dataset.lab===state.lab));if(state.lab==='analysis')loadAnalysis()}
function populateStyles(){const options=state.styles.map(x=>`<option value="${esc(x.id)}">${esc(x.name)}</option>`).join('');$('stylePreset').innerHTML='<option value="">由描述决定，不追加配方</option>'+options}
function compilePayload(){const base={sentence:$('intent').value,style_mode:$('styleMode').value,...selectedKnowledge()};if(!base.sentence.trim())throw new Error('请先描述你想要的画面');return base}
function showSession(stage,percent,detail,source='protected'){state.sessionSource=source;if(source!=='job'){$('jobReceiptControls')?.classList.add('hidden');$('jobReceiptFeedback')?.classList.add('hidden');}$('sessionEmpty').classList.add('hidden');$('sessionLive').classList.remove('hidden');$('sessionStage').textContent=stage;$('sessionPercent').textContent=percent==null?'':`${Math.round(percent)}%`;$('sessionMeter').parentElement.hidden=percent==null;$('sessionMeter').style.width=percent==null?'0':`${percent}%`;$('sessionDetail').textContent=detail;const marks=[8,20,40,82,96];[...$('stageList').children].forEach((el,i)=>el.classList.toggle('done',percent!=null&&percent>=marks[i]))}
function resolvedCreationConfig(){
  try{
  const config=AuthorPreparation.outputConfig(CreationConfig.resolve(creationCatalog,{professional:$('proMode').checked,values:professionalValues(),quality:$('quality').value,style:$('stylePreset').value,seed:state.compilationSeed??(state.compilationSeed=Math.floor(Math.random()*2147483647))}),{strategy:$('generationStrategy').value||'single',quality:$('quality').value});
  if($('generationStrategy').value==='creative'&&(CreationConfig.bridges.has(config.render_pipeline)||config.source_parameters))throw new Error('创作任务这里选择起点，不执行预排固定桥接或源阶段参数。请明确选择一个起点；固定路线在单路线兼容／研究入口保留。没有删除设置或静默改用 NewBie。');
  CreationConfig.validateStyles(config,state.styles);
  if(config.lora_name&&!creationLoras.some(item=>item.name===config.lora_name&&item.compatible_models?.includes(config.model_id)))throw new Error('所选 LoRA 未确认与当前模型兼容。');
  if(config.source_parameters?.lora_name&&!creationLoras.some(item=>item.name===config.source_parameters.lora_name&&item.compatible_models?.includes(config.source_parameters.model_id)))throw new Error('源阶段 LoRA 未确认与当前模型兼容。');
  return config;
  }catch(error){error.creationErrorSource='local_config';throw error;}
}
// Only a locally rejected configuration can expire when its actual controls
// change. Preparation/server failures and uncertain dispatch receipts are not
// configuration errors, and this path never polls, submits, cancels or resumes.
function creationControlFingerprint(){return AuthorPreparation.key([...$('createView').querySelectorAll('input,select')].map(field=>({id:field.id,value:field.value,checked:field.type==='checkbox'?field.checked:null})));}
function hasProtectedCreationRecord(){const saved=preparationObserver.saved();return Boolean(state.creativeSession||state.job||state.preparation||saved?.outcome_unknown||saved?.meta?.dispatch);}
function showLocalConfigError(error){
  const el=$('createMessage'),previous=state.localConfigError;
  const previousMessage=previous&&el.dataset.messageSource==='local_config'?previous.previousMessage:{text:el.textContent,hidden:el.classList.contains('hidden'),bad:el.classList.contains('bad'),source:el.dataset.messageSource||'protected'};
  state.localConfigError={source:'local_config',fingerprint:creationControlFingerprint(),message:error.message,previousMessage};
  message('createMessage',error.message,true,'local_config');
  if(!hasProtectedCreationRecord())showSession('无法继续',null,error.message,'local_config');
}
function reconcileLocalConfigError(){
  const previous=state.localConfigError;
  if(!previous||previous.source!=='local_config'||state.preparationBusy||state.submitting||state.rebuilding||creationControlFingerprint()===previous.fingerprint)return false;
  try{const config=resolvedCreationConfig();if($('generationStrategy').value==='local_iteration')AuthorPreparation.validateIterationRequest(config);}
  catch(error){showLocalConfigError(error);return false;}
  const el=$('createMessage');
  if(el.dataset.messageSource==='local_config'&&el.textContent===previous.message){
    const restored=previous.previousMessage;
    if(restored.hidden){el.textContent='';el.classList.add('hidden');el.classList.remove('bad');el.dataset.messageSource='protected';}
    else message('createMessage',restored.text,restored.bad,restored.source);
  }
  if(state.sessionSource==='local_config'&&$('sessionStage').textContent==='无法继续'&&$('sessionDetail').textContent===previous.message&&!hasProtectedCreationRecord()){
    $('sessionLive').classList.add('hidden');$('sessionEmpty').classList.remove('hidden');$('sessionStage').textContent='';$('sessionDetail').textContent='';state.sessionSource='empty';
  }
  state.localConfigError=null;return true;
}
function preparationPayload(){return AuthorPreparation.input({...compilePayload(),config:resolvedCreationConfig()});}
function currentPreparationMatches(record){return Boolean(record&&AuthorPreparation.sameInput(preparationPayload(),record.input_snapshot));}
function readAuthorEditor(){return{scene_narrative:$('authorNarrative').value,visual_logic:$('authorLogic').value,positions:$('authorPositions').value,...(state.compiled?.author_proposal&&Object.prototype.hasOwnProperty.call(state.compiled.author_proposal,'layout')?{layout:$('authorLayout').value}:{})};}
function populateAuthorEditors(){
  const proposal=state.compiled?.author_proposal;if(!proposal)return;
  $('authorOriginal').textContent=state.compiled.creation_intent?.original_text||state.compilationIntent.sentence;
  $('authorNarrative').value=proposal.scene_narrative;$('authorLogic').value=proposal.visual_logic;$('authorPositions').value=JSON.stringify(proposal.positions,null,2);
  $('authorLayout').value=JSON.stringify(proposal.layout||[],null,2);$('authorLayoutGroup').hidden=!proposal.layout?.length&&!Object.prototype.hasOwnProperty.call(proposal,'layout');
  $('authorStatus').textContent='本次作者提案，不是用户原始要求。修改后需重建；格式齐全不代表画面质量达标。';
}
function applyPreparation(record){
  if(!currentPreparationMatches(record))throw new Error('描述、知识或完整配置已变化，旧提案不会用于生成；请重新准备。');
  if(!['ready','needs_confirmation'].includes(record.status)||!record.compiled)throw new Error((record.status==='failed'?AuthorPreparation.describe(record):record.error||record.detail)||'本次构思尚未完成');
  if(AuthorPreparation.key(record.effective_config)!==AuthorPreparation.key(record.input_snapshot.config))throw new Error('准备配置与输入快照不一致，未提交生成。');
  state.preparation=record;state.compilationIntent=record.input_snapshot;state.compiled=record.compiled;
  $('sourceReview').hidden=true;
  populateCompiledEditors();populateAuthorEditors();$('openPromptReview').disabled=false;
  if(state.compiled.warnings?.length)message('createMessage',state.compiled.warnings.join('\n'));
}
function showSourceReview(record){
  if(record.compiled||!record.source_review?.blocking)return false;
  state.preparation=record;state.compiled=null;$('openPromptReview').disabled=true;
  globalThis.SourceReview?.render($('sourceReview'),record);
  const continuation=record.source_continuation_requests?.[record.source_continuation_request_id];
  $('sourceReviewStatus').textContent=record.error||[
    continuation?.author_concern?.reason,
    ...(continuation?.author_concern?.concerns||[]).map(item=>`原文：${item.source_quote}\n当前疑点：${item.reason}`),
    record.detail].filter(Boolean).join('\n\n');
  const revision=record.source_reconsideration;
  const notes=revision?.revision_art_raw?.change_annotations;
  const candidate=revision?.preview&&Array.isArray(notes)
    ?['系统重新阅读后提出了候选解释，尚未采纳；以下理由也不是已核定的结论。',
      ...notes.map(item=>`原文：${item.source_quote}\n重看理由：${item.reason}`)].join('\n\n'):'';
  const questions=(record.source_review.questions||[]).map(item=>`${item.question}\n${item.reason}\n${item.action}`).join('\n\n');
  message('createMessage',[candidate,questions].filter(Boolean).join('\n\n'),true);
  showSession('需要核对解释或资源',null,record.detail||'本次解释仍需核对，未提交生图。');
  return true;
}
async function continueSourceReview(action){
  if(state.preparationBusy||state.submitting)return;
  const record=state.preparation,view=globalThis.SourceReview?.model(record);
  if(!view)return;
  try{
    if(!currentPreparationMatches(record))throw new Error('描述或资源已改变，请重新准备；没有采用旧候选。');
    if(action==='adopt'&&!view.canAdopt)throw new Error('这份候选仍有冲突，请纠正理解或调整输入。');
    const clarification=$('sourceClarification').value;
    if(action==='clarify'&&!clarification.trim())throw new Error('请用自然语言说明哪处理解错了。');
    state.preparationBusy=true;state.autoPreparation=false;$('make').disabled=true;
    $('adoptSource').disabled=true;$('clarifySource').disabled=true;
    $('sourceReviewStatus').textContent=action==='adopt'?'请求按这份解释重新构思；不自动生图。':'请求重新理解；新候选仍需要你采用。';
    const next=await preparationObserver.resumeSource({action,expected_revision:record.revision,
      candidate_sha256:view.candidate_sha256,...(action==='clarify'?{clarification}:{})});
    if(!next)return;
    if(!currentPreparationMatches(next))throw new Error('接续期间描述或资源已变化，旧返回未用于生成，请重新准备。');
    if(showSourceReview(next))return;
    if(next.compiled&&['ready','needs_confirmation'].includes(next.status)){
      applyPreparation(next);showSession('提案已准备',null,'采用后的新画面方案已准备，尚未生图。');
      message('createMessage','已按采用的解释重新构思。可审阅提示词，或明确点击“生成画面”；尚未生图。');
    }else throw new Error(next.error||next.detail||'接续尚未成功；没有生图。');
  }catch(error){$('sourceReviewStatus').textContent=`${error.message} 没有自动生图或另开任务。`;}
  finally{state.preparationBusy=false;$('make').disabled=!creationCatalog;
    if(state.preparation&&!state.compiled)globalThis.SourceReview?.render($('sourceReview'),state.preparation);}
}
const preparationObserver=AuthorPreparation.create({read:async(path,options={})=>{
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),20000);
  try{return await api(path,{...options,signal:controller.signal});}finally{clearTimeout(timer);}
},storage:localStorage,onRecord(record){
  const restoration=state.preparationRestoreContext;
  if(restoration&&!preparationRestoreOwnsView(restoration)){
    showPreparationHistory(record);return;
  }
  state.preparation=record;
  if(restoration&&historicalPreparationCanStandAside(record)){
    showPreparationHistory(record);showCreationStandby();return;
  }
  if(['queued','running'].includes(record.status)){$('adoptSource').disabled=true;$('clarifySource').disabled=true;}
  $('cancelPreparation').classList.toggle('hidden',!['queued','running'].includes(record.status));
  const historical=state.restoringPreparationHistory;
  showSession((historical?'历史兼容准备 · ':'')+AuthorPreparation.stage(record),null,
    `${historical?`只读恢复旧准备 ${record.id}，没有重新执行。 `:''}${AuthorPreparation.describe(record)} 准备只形成提案与模型条件，不等于画面已生成。`,'preparation');
},onConnection(error){if(state.preparationRestoreContext&&!preparationRestoreOwnsView(state.preparationRestoreContext))return;$('jobConnection').textContent=error?'暂时无法读取准备状态，正在重试读取；不会重启构思或提交生图。':'';}});
async function beginCreation(){
  if(state.preparationBusy||state.submitting)return;
  if(!$('generationStrategy').value)$('generationStrategy').value='creative';
  if($('generationStrategy').value==='creative')return beginCreativeCreation();
  globalThis.CreativeSessionUI?.hide();
  const professional=$('proMode').checked,strategy=$('generationStrategy').value||'creative';state.preparationBusy=true;state.autoPreparation=!professional;state.preparationCancelled=false;
  try{
    if(strategy==='local_iteration'){try{AuthorPreparation.validateIterationRequest(resolvedCreationConfig());}catch(error){error.creationErrorSource='local_config';throw error;}}
    if(typeof jobWatcher!=='undefined')jobWatcher.stop();globalThis.CooperativeUI?.begin();
    $('make').disabled=true;$('sessionResult').classList.add('hidden');$('createMessage').classList.add('hidden');
    $('sourceReview').hidden=true;
    let record=state.preparation;
    const reusable=record&&['ready','needs_confirmation'].includes(record.status)&&currentPreparationMatches(record)&&!preparationObserver.dispatch(record);
    if(!reusable){
      const pending=preparationObserver.saved();
      if(pending?.meta?.dispatch?.status==='unknown'){
        const latest=await preparationObserver.refresh();
        if(recoverPreparedSubmission(latest))return;
      }
      if(pending?.meta?.dispatch?.status==='unknown'&&!confirm('上次生图提交结果未确认，请先核对队列和作品。仍要另开一次？这可能产生额外图片，不是安全重试。'))return;
      const recoveringUnknown=pending?.outcome_unknown&&currentPreparationMatches({input_snapshot:pending.input});
      if(!recoveringUnknown)state.compilationSeed=Math.floor(Math.random()*2147483647);
      state.compiled=null;$('openPromptReview').disabled=true;
      showSession('构思画面',null,'保留完整原句、知识选择和配置，准备一次可修改的作者提案。');
      record=await preparationObserver.start(preparationPayload(),{professional,quality:$('quality').value,style:$('stylePreset').value,strategy:$('generationStrategy').value});
    }
    if(!record)return;
    if(state.preparationCancelled)return;
    if(record.status==='cancelled'){showSession('准备已取消',null,'本次未提交生图。');return;}
    if(showSourceReview(record))return;
    if(professional!==$('proMode').checked)throw new Error('准备期间模式已改变，未自动提交。请审阅或重新准备。');
    if(strategy!==($('generationStrategy').value||'creative'))throw new Error('准备期间生成方式已改变，未自动提交。请审阅或重新准备。');
    if(state.compiled!==record.compiled)applyPreparation(record);
    showSession('提案已准备',null,'已经形成模型条件；准备完成不证明理解和画面质量成立。');
    if(state.compiled.professional_review?.blocking&&!$('proMode').checked){
      const questions=state.compiled.professional_review.questions||[];
      message('createMessage',questions.map(item=>`${item.question}\n${item.reason}\n${item.action}`).join('\n\n'),true);
      showSession('需要确认',null,'描述存在冲突或关键缺失。修改原描述后再生成，不需要填写专业参数。');return;
    }
    if($('proMode').checked)openPromptReview();
    else if(state.autoPreparation)await submitCreation();
  }catch(error){if(error.creationErrorSource==='local_config')showLocalConfigError(error);else{message('createMessage',error.message,true);showSession('无法继续',null,error.message)}}
  finally{state.preparationBusy=false;state.autoPreparation=false;$('make').disabled=false}
}
async function rebuildContract(){
  if(state.submitting||state.rebuilding)return;
  state.rebuilding=true;$('rebuildContract').disabled=true;
  try{
    if(!currentPreparationMatches(state.preparation))throw new Error('原描述、知识或配置已变化，请重新准备。');
    const spec=JSON.parse($('contractJson').value),proposal=AuthorPreparation.proposal(state.compiled,readAuthorEditor());
    const record=await preparationObserver.rebuild({expected_revision:state.preparation.revision,proposal,spec});
    applyPreparation(record);message('promptMessage',record.status==='needs_confirmation'?'已重建，仍有冲突或关键缺失；没有生图。':'提案、规格与各模型条件已重建。文本同步不证明语义正确；没有生图。');
  }catch(error){message('promptMessage',error.message,true);}
  finally{state.rebuilding=false;$('rebuildContract').disabled=false;}
}
function jobPayload(){
  if(!currentPreparationMatches(state.preparation)||state.preparation.status!=='ready')throw new Error('描述、知识或配置已变化，或准备尚未就绪；请重新准备后生成。');
  if(state.compiled?.professional_review?.blocking)throw new Error('请先处理生成前确认中的冲突或关键缺失，再重新解析或修改规格并重建格式。');
  if($('proMode').checked&&AuthorPreparation.proposalChanged(state.compiled,readAuthorEditor()))throw new Error('作者提案已修改，请先用提案与规格重建条件。');
  const resolved=StudioRequest.resolveCompilation(state.compiled,$('proMode').checked?readCompiledEditor():null);
  const config=resolvedCreationConfig(),proposal=AuthorPreparation.metadata(state.compiled,resolved),manual=proposal.condition_status==='historical_after_native_override';
  if(manual&&globalThis.CooperativeUI?.requested())throw new Error('原生文本手改后旧场景未同步，不能用于固定四候选比较。请审阅提案并重建，或选择单路线。');
  if(manual&&$('generationStrategy')?.value==='local_iteration')throw new Error('原生文本手改后旧场景未同步，不能用于按结果逐步创作。请审阅提案并重建；没有自动替换你的文本。');
  if(manual&&config.render_pipeline==='adaptive')throw new Error('原生文本手改后旧场景未同步，自动路线无法据此选择。请选择显式模型或路线，或审阅提案并重建；没有自动换路。');
  if(manual)message('promptMessage','将使用手改的原生正负文本（包括空负面）；旧提案只作为历史，不再当作当前语义契约。');
  const binding=globalThis.CooperativeUI?.requested()?{}:{preparation_id:state.preparation.id,preparation_revision:state.preparation.revision};
  return {...config,...CreationConfig.conditioning(config,resolved,state.compiled),...binding,pipeline_version:'adaptive',mode:'character',output_group:'one_sentence',output_label:`${resolved.spec.character_tag||'character'}__${Date.now()}`,semantic_spec:manual?null:resolved.spec,scene_contract:manual?null:state.compiled.contract_v3,creation_intent:state.compiled.creation_intent,author_proposal:proposal};
}
function recoverPreparedSubmission(record){
  const receipt=AuthorPreparation.receipt(record);if(!receipt)return false;
  state.preparation=record;
  if(receipt.kind==='iteration'&&receipt.status==='submitted'&&receipt.run_id){
    preparationObserver.rememberDispatch(record,{status:'submitted',kind:'iteration',run_id:receipt.run_id});
    $('confirmMake').disabled=true;if($('promptDialog').open)$('promptDialog').close();
    message('createMessage',`已找到逐步创作任务 ${receipt.run_id}，仅读取已有任务，不重新生成或自动恢复。`);
    openIterationRun(receipt.run_id);return true;
  }
  if(receipt.kind==='iteration'&&receipt.status==='submitted'){
    preparationObserver.rememberDispatch(record,{status:'unknown',kind:'iteration'});$('confirmMake').disabled=true;
    message('createMessage','服务端已有迭代提交回执，但任务编号不完整。不会猜测编号、恢复或重新提交；请核对服务端记录。',true);return true;
  }
  if(receipt.status==='submitted'&&receipt.job_id){
    preparationObserver.rememberDispatch(record,{status:'submitted',job_id:receipt.job_id});
    $('confirmMake').disabled=true;if($('promptDialog').open)$('promptDialog').close();
    message('createMessage',`已恢复服务端关联任务 ${receipt.job_id}，没有再次提交生图。`);pollJob(receipt.job_id);return true;
  }
  if(['submitting','unknown'].includes(receipt.status)){
    preparationObserver.rememberDispatch(record,{status:'unknown'});$('confirmMake').disabled=true;
    message('createMessage','服务端提交回执尚未核验，不会再次派发或另造输出标识；请核对已有任务。',true);
    showSession('提交待核验',null,'仅读取已有准备与队列状态，不重复生成。');return true;
  }
  return false;
}
function openIterationRun(id){
  if(!/^[a-zA-Z0-9_-]{1,100}$/.test(id))throw new Error('服务端返回的迭代任务编号不合法，未打开任务。');
  if(typeof jobWatcher!=='undefined')jobWatcher.stop();
  location.assign(`/iteration?id=${encodeURIComponent(id)}`);
}
async function submitCreation(){
  if(state.submitting||state.rebuilding)return;
  let attempted=false,record=state.preparation;state.submitting=true;$('confirmMake').disabled=true;
  try{
    const local=preparationObserver.dispatch(record);
    const latest=await preparationObserver.refresh();
    if(latest.id!==record?.id||latest.revision!==record.revision)throw new Error('准备版本或状态已变化，请重新准备并审阅；未提交生图。');
    if(recoverPreparedSubmission(latest))return;
    if(latest.status!=='ready')throw new Error('准备尚未就绪或已取消，未提交生图。');
    if(local&&local.status!=='prepared'&&AuthorPreparation.receipt(latest)?.status!=='rejected')throw new Error('本版本已有提交记录，服务端未确认可重试；不会重复生成。请查看已有任务。');
    const currentRequest=jobPayload();
    const strategy=$('generationStrategy')?.value||'single',iteration=strategy==='local_iteration';
    let fullPayload=currentRequest;
    if(iteration){
      const models=await api('/api/models');
      if(($('generationStrategy')?.value||'single')!==strategy)throw new Error('读取模型目录期间生成方式已变化，未启动迭代。');
      const refreshed=jobPayload();
      if(AuthorPreparation.key({...currentRequest,output_label:''})!==AuthorPreparation.key({...refreshed,output_label:''}))throw new Error('读取模型目录期间完整输入已变化，请重新审阅；未启动迭代。');
      fullPayload=AuthorPreparation.iterationPayload(currentRequest,models);
    }
    const request=preparationObserver.freezePayload(record,fullPayload);attempted=true;preparationObserver.rememberDispatch(record,{status:'unknown',...(iteration?{kind:'iteration'}:{})});
    if(iteration){
      showSession('建立逐步创作任务',null,'保留完整原句、提案、正负条件与参数；后续根据真实图像提出操作，不是固定模型链。');
      const run=await api(`/api/creation-preparations/${encodeURIComponent(record.id)}/iteration`,{method:'POST',body:JSON.stringify(request)});
      preparationObserver.rememberDispatch(record,{status:'submitted',kind:'iteration',run_id:run.id});
      if($('promptDialog').open)$('promptDialog').close();openIterationRun(run.id);return;
    }
    if(globalThis.CooperativeUI?.requested()){await CooperativeUI.launch(request);preparationObserver.rememberDispatch(record,{status:'submitted',kind:'cooperative'});return;}
    showSession('加入本地队列',20,'将执行当前选定路线，并保留阶段记录。');
    state.job=await api('/api/jobs',{method:'POST',body:JSON.stringify(request)});
    preparationObserver.rememberDispatch(record,{status:'submitted',job_id:state.job.id});
    if($('promptDialog').open)$('promptDialog').close();pollJob(state.job.id);
  }catch(error){
    if(attempted&&error.status>=400&&error.status<500&&error.status!==408)preparationObserver.rememberDispatch(record,{status:'rejected',error:error.message});
    message($('promptDialog').open?'promptMessage':'createMessage',attempted&&(!error.status||error.status>=500)?`提交结果尚未确认，不会盲目重试。请查看任务队列。${error.message}`:error.message,true);
  }finally{state.submitting=false;$('confirmMake').disabled=Boolean(preparationObserver.dispatch(record));}
}
function stageInfo(job,runtime){
  const terminal=['completed','failed','cancelled'].includes(job.status);
  const stopUnknown=job.status==='cancellation_uncertain'||job.stage==='cancellation_uncertain'
    ||(!terminal&&Boolean(job.cancel_requested||job.cancel_error));
  if(stopUnknown)return['停止尚未确认',null,[
    '停止请求已记录；是否完成停止仍待真实回执核对，不会据此重派。',
    job.cancel_error||job.error,...(Array.isArray(job.notes)?job.notes:[])].filter(Boolean).join('\n')];
  if(job.stage==='waiting_for_unconfirmed_engine_receipt')return['等待未确认任务回执',null,[
    '本任务尚未提交；正在等待其它任务的确切引擎回执。',
    ...(Array.isArray(job.notes)?job.notes:[])].filter(Boolean).join('\n')];
  if(job.status==='completed')return['完成',100,'图片、参数与检查结果已经保存。'];
  if(job.status==='failed')return['生成失败',Math.round(job.progress*100),job.error||'推理任务失败'];
  if(job.status==='cancelled')return['已取消',Math.round(job.progress*100),'任务已停止'];
  const map={queued:['等待执行',22],starting_engine:['加载模型',28],queued_in_engine:['进入采样器',34],source_composition:['源阶段生成',40],anima_composition:['源阶段生成',40],newbie_identity:['后续 NewBie 阶段',65],sampling:['生成画面',40+Math.min(40,(runtime.sampling_percent||0)*.4)],check_detect:['检查画面',86],text_detect:['检查文字',90],saving:['保存作品',96]};
  const [name,pct]=map[job.stage]||['处理中',30];return[name,pct,`${job.stage}${runtime.steps?` · ${runtime.step}/${runtime.steps}`:''}`];
}
let jobReceiptBusy=false;
function jobReceiptUnknown(job){
  return Boolean(job&&(job.status==='cancellation_uncertain'||job.stage==='cancellation_uncertain'
    ||(!['completed','failed','cancelled'].includes(job.status)&&(job.cancel_requested||job.cancel_error))));
}
function updateJobReceiptControls(job){
  const visible=state.sessionSource==='job'&&jobReceiptUnknown(job),valid=/^[a-zA-Z0-9_-]{1,100}$/.test(job?.id||'');
  $('jobReceiptControls')?.classList.toggle('hidden',!visible);
  for(const id of ['inspectJobReceipt','cancelJobReceipt'])if($(id))$(id).disabled=jobReceiptBusy||!visible||!valid;
  const saved=state.sessionSource==='job'&&state.jobReceiptFeedback?.jobId===job?.id?state.jobReceiptFeedback:null;
  if($('jobReceiptFeedback')){$('jobReceiptFeedback').textContent=saved?.text||'';$('jobReceiptFeedback').classList.toggle('hidden',!saved);$('jobReceiptFeedback').classList.toggle('bad',Boolean(saved?.bad));}
}
function renderTrackedJob(job,runtime){
  if(state.job?.id!==job.id||state.sessionSource!=='job')return;
  state.job=job;const inspected=state.jobReceiptFeedback?.jobId===job.id;
  const terminal=['completed','failed','cancelled'].includes(job.status);
  const [name,pct,detail]=inspected&&terminal?['原任务状态已读取',null,
    `原记录状态：${job.status} · ${job.stage}。终态不代表作品成功；没有接收旧图或重新派发。`]:stageInfo(job,runtime||{});
  showSession(name,pct,detail,'job');updateJobReceiptControls(job);
  if(inspected)$('sessionResult').classList.add('hidden');
  if(job.status==='completed'&&!inspected){
    const artId=`${job.id}-01`;
    if(job.images?.[0]){$('sessionImage').src=job.images[0];$('sessionFile').textContent=job.files?.[0]||artId;$('openSessionArtwork').dataset.id=artId;$('sessionResult').classList.remove('hidden');}
    loadArtworks();
  }
}
async function jobReceiptRead(path,options={}){
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),20000);
  try{return await api(path,{...options,signal:controller.signal});}finally{clearTimeout(timer);}
}
async function performJobReceipt(operation){
  const job=state.job;if(jobReceiptBusy||!['inspect','cancel'].includes(operation)||!jobReceiptUnknown(job)
    ||state.sessionSource!=='job'||!/^[a-zA-Z0-9_-]{1,100}$/.test(job?.id||''))return null;
  const jobId=job.id,base=`/api/jobs/${encodeURIComponent(jobId)}`,label=operation==='inspect'?'核对原任务接口':'停止原任务接口';
  jobReceiptBusy=true;jobWatcher.stop();state.jobReceiptFeedback={jobId,operation,text:`正在${operation==='inspect'?'核对':'请求停止'}原任务 ${jobId}；尚未确认，不会重派。`,bad:false};updateJobReceiptControls(job);
  const errors=[];
  try{
    try{await jobReceiptRead(`${base}/${operation==='inspect'?'inspect-engine':'cancel'}`,{method:'POST'});}
    catch(error){errors.push(`${label}未确认：${error.message}`);}
    try{const latest=await jobReceiptRead(base);if(latest?.id!==jobId)throw new Error('返回的任务编号与原记录不一致，未采用其它任务状态。');if(state.job?.id===jobId)renderTrackedJob(latest,null);}
    catch(error){errors.push(`原任务状态读取未确认：${error.message}`);}
    state.jobReceiptFeedback={jobId,operation,bad:Boolean(errors.length),text:errors.length?errors.join('\n'):
      `原任务 ${jobId} 的请求已返回并重新读取状态；这不代表图片或作品成功，不接旧图、不重派。必要时停止旧创作，再明确开始新构想。`};
    return {jobId,error:Boolean(errors.length)};
  }finally{jobReceiptBusy=false;updateJobReceiptControls(state.job);
    if(state.job?.id===jobId&&state.sessionSource==='job'&&jobReceiptUnknown(state.job))jobWatcher.watch(jobId);}
}
const trackedJobKey='pipeline-studio.current-job';
async function readTracked(path,signal){
  const response=await fetch(path,{signal});
  if(!response.ok){const error=new Error(`HTTP ${response.status}: ${await response.text()}`);error.status=response.status;throw error;}
  return response.json();
}
const jobWatcher=JobWatcher.create({
  readJob:(id,signal)=>readTracked(`/api/jobs/${encodeURIComponent(id)}`,signal),
  readRuntime:signal=>readTracked('/api/runtime',signal),
  onJob:renderTrackedJob,
  onConnection(value){$('jobConnection').textContent=value?`暂时无法读取任务 ${value.jobId}；${Math.ceil(value.retryMs/1000)} 秒后重新读取。不会重新提交生成。`:'';$('retryJobRead').classList.toggle('hidden',!value);},
  onMissing(id){$('jobConnection').textContent=`服务找不到任务 ${id} 的记录，已停止自动读取。这不代表任务被取消；可手动重试。`;$('retryJobRead').classList.remove('hidden');},
  onRuntimeUnavailable(error){$('runtimeConnection').textContent=error?'资源监控暂不可用；任务状态仍会继续读取。':'';},
  onRenderError(error){message('createMessage',`任务已读取，但界面更新失败：${error.message}`,true);},
});
function pollJob(id){
  if(!/^[a-zA-Z0-9_-]{1,100}$/.test(id))return;
  globalThis.CooperativeUI?.single();
  clearTimeout(state.poll);state.job=state.job?.id===id?state.job:{id};state.sessionSource='job';updateJobReceiptControls(state.job);StageImages.attachJob(id);
  try{localStorage.setItem(trackedJobKey,id);}catch{}
  return jobWatcher.watch(id);
}
function restoreTrackedJob(){
  if(creativeUI.hasSaved()||new URLSearchParams(location.search).get('session'))return;
  const preparing=preparationObserver.saved();
  if(preparing?.meta?.dispatch?.kind==='iteration'||AuthorPreparation.receipt(preparing?.record)?.kind==='iteration')return;
  if(preparing?.id&&['queued','running'].includes(preparing.record?.status)&&!preparing.meta?.dispatch)return;
  try{if(localStorage.getItem('pipeline-studio.current-cooperative-run'))return;}catch{}
  let id;try{id=localStorage.getItem(trackedJobKey);}catch{return;}
  if(!id||!/^[a-zA-Z0-9_-]{1,100}$/.test(id))return;
  showSession('恢复任务视图',0,`正在读取已有任务 ${id}，不会重新提交生成。`);pollJob(id);
}
async function cancelPreparation(){
  state.autoPreparation=false;state.preparationCancelled=true;$('cancelPreparation').disabled=true;
  try{
    const record=await preparationObserver.cancel();
    if(!record)throw new Error('准备请求还未返回编号，尚不能确认服务端取消；不会自动提交生图。');
    if(record.status==='cancelled'){state.compiled=null;$('openPromptReview').disabled=true;}
    message('createMessage',record.detail||'已请求取消；请等待真实停止状态。');
  }catch(error){message('createMessage',`未确认取消：${error.message} 不会自动提交生图。`,true);}
  finally{$('cancelPreparation').disabled=false;}
}
async function restorePreparationInputs(saved){
  if(state.inputTouched||$('intent').value.trim()||!saved?.input)return;
  const input=saved.input,meta=saved.meta||{},config=input.config||{};
  $('intent').value=input.sentence;$('styleMode').value=input.style_mode;
  $('proMode').checked=meta.professional===true;
  // A recorded output choice remains explicit; absence is not permission to
  // replace the current draft with the historical high-detail default.
  if(['high','standard','iteration_base'].includes(meta.quality))$('quality').value=meta.quality;
  $('stylePreset').value=meta.style||'';
  state.compilationSeed=config.seed;
  const declared=['creative','single','local_iteration','cooperative'].includes(meta.strategy)?meta.strategy:'creative';
  $('generationStrategy').value=$('proMode').checked?declared:(declared==='local_iteration'?'local_iteration':'creative');
  if($('proMode').checked)state.professionalGenerationStrategy=$('generationStrategy').value;
  if($('proMode').checked){
    setValues(config);$('loraChoice').value=config.lora_name||'';
    $('sourceStyle').value=config.composition_style_preset??'inherit';$('finalStyle').value=config.character_style_preset??'inherit';
    updateConfigContext();
    if(config.source_parameters&&$('sourceOverride')){
      $('sourceOverride').checked=true;
      for(const field of document.querySelectorAll('[data-source-setting]'))field.value=config.source_parameters[field.dataset.sourceSetting]??'';
      $('sourceOverride').onchange();
    }
  }
  professionalModeChanged();
  if(input.knowledge_pack){
    $('knowledgePack').value=input.knowledge_pack;await selectKnowledgePack();
    if(state.inputTouched)return;
    $('operator').value=input.knowledge_operator;populateAppearances();$('appearance').value=input.knowledge_appearance;updateKnowledgeNote();
  }
}
function preparationRestoreOwnsView(context){
  return state.preparationRestoreContext===context
    &&(state.creationPresentationRevision||0)===context.presentationRevision
    &&state.creativeSession===context.creativeSession
    &&$('generationStrategy').value===context.strategy&&$('proMode').checked===context.professional;
}
function historicalPreparationCanStandAside(record){
  if($('proMode').checked||$('generationStrategy').value!=='creative'
      ||!['ready','needs_confirmation','failed','cancelled'].includes(record?.status))return false;
  const saved=preparationObserver.saved(),receipt=AuthorPreparation.receipt(record);
  return !saved?.outcome_unknown&&!saved?.source_resume?.unknown
    &&!['unknown','submitting'].includes(receipt?.status)
    &&!(receipt?.status==='submitted'&&(receipt.job_id||receipt.run_id))
    &&saved?.meta?.dispatch?.status!=='unknown';
}
function showPreparationHistory(record){
  state.historicalPreparation=record;
  const history=$('preparationHistory');
  if(history.dataset.recordId!==record.id)history.open=false;
  history.dataset.recordId=record.id;history.hidden=false;
  $('preparationHistorySummary').textContent=`上次兼容准备 · ${record.id} · ${AuthorPreparation.stage(record)}`;
  $('preparationHistoryOriginal').textContent=record.input_snapshot?.sentence||'原始描述未记录，未补造';
  $('preparationHistoryDetail').textContent=AuthorPreparation.describe(record)||record.detail||'原记录保留。';
  $('preparationHistoryLink').href=`/api/creation-preparations/${encodeURIComponent(record.id)}`;
}
function showCreationStandby(){
  // Only a retired preparation owns this transition. It does not clear the
  // saved record, user draft, selected quality, or any current creative task.
  if(state.creativeSession||state.job||!state.preparationRestoreContext
      ||!preparationRestoreOwnsView(state.preparationRestoreContext))return;
  $('sessionLive').classList.add('hidden');$('sessionEmpty').classList.remove('hidden');
  $('sessionStage').textContent='';$('sessionDetail').textContent='';state.sessionSource='empty';
  $('cancelPreparation').classList.add('hidden');$('sourceReview').hidden=true;
}
async function restoreCreationPreparation(){
  const saved=preparationObserver.saved();if(!saved)return;
  // Fixed candidate execution has its own GET-only run observer and is not a
  // public first-job submission in the preparation receipt directory.
  if(saved.meta?.dispatch&&saved.meta.strategy==='cooperative')return;
  if(state.creativeSession)return;
  const context={id:saved.id,presentationRevision:state.creationPresentationRevision||0,
    creativeSession:state.creativeSession,strategy:$('generationStrategy').value,professional:$('proMode').checked};
  state.preparationRestoreContext=context;
  state.preparationBusy=true;state.autoPreparation=false;state.restoringPreparationHistory=true;$('make').disabled=true;
  try{
    await restorePreparationInputs(saved);
    if(state.preparationRestoreContext!==context||(state.creationPresentationRevision||0)!==context.presentationRevision
        ||state.creativeSession!==context.creativeSession)return;
    context.strategy=$('generationStrategy').value;context.professional=$('proMode').checked;
    if(!saved.id){message('createMessage','上次准备请求的回执未确认。原请求与输入保留；刷新不会再次提交，请先核对原记录，未确认前不跨入口新建。',true);return;}
    if(typeof jobWatcher!=='undefined')jobWatcher.stop();globalThis.CooperativeUI?.begin();
    const record=await preparationObserver.restore();if(!record)return;
    if(!preparationRestoreOwnsView(context)){showPreparationHistory(record);return;}
    if(historicalPreparationCanStandAside(record)){
      showPreparationHistory(record);showCreationStandby();return;
    }
    if(recoverPreparedSubmission(record))return;
    if($('generationStrategy')?.value==='creative'){
      $('sourceReview').hidden=true;
      message('createMessage',`历史兼容准备 ${record.id}（只读，未重跑）。\n${AuthorPreparation.describe(record)||record.detail||'原记录保留。'}\n下次明确生成使用创作任务，不采用或续跑这份旧提案。`,record.status==='failed');
      return;
    }
    if(showSourceReview(record))return;
    if(saved.meta?.dispatch){
      message('createMessage','本地保存了提交记录，但服务端尚未返回关联任务。不会自动重试，请先核对队列。',true);return;
    }
    if(['ready','needs_confirmation'].includes(record.status)){
      applyPreparation(record);
      message('createMessage','已恢复本次提案，仅读取记录，没有生图。可审阅，或明确点击生成。');
    }else message('createMessage',AuthorPreparation.describe(record)||record.error||'准备没有完成。',record.status==='failed');
  }catch(error){if(preparationRestoreOwnsView(context))message('createMessage',error.message,true);}
  finally{if(state.preparationRestoreContext===context){state.preparationRestoreContext=null;state.preparationBusy=false;state.restoringPreparationHistory=false;$('make').disabled=!creationCatalog;}}
}
async function previewGraph(){try{const request=jobPayload();if($('generationStrategy')?.value==='local_iteration'){message('promptMessage','按结果逐步创作没有预设模型链。提交后在任务页查看实际连线图；这里不将某条固定路线冒充尚未发生的迭代。');return;}if(globalThis.CooperativeUI?.requested()){await CooperativeUI.preview(request);return;}const result=await api('/api/workflow-preview',{method:'POST',body:JSON.stringify(request)});PipelineDiagram.mountEvidence($('graphContent'),result);$('graphDialog').showModal()}catch(error){message($('promptDialog').open?'promptMessage':'createMessage',error.message,true)}}
const libraryState={items:[],offset:0,sequence:0,more:false,timer:null};
async function loadArtworks({append=false}={}){
  const sequence=++libraryState.sequence,query=$('artSearch').value.trim(),origin=$('artOrigin').value;
  const offset=append?libraryState.offset:0,params=new URLSearchParams({limit:'24',offset:String(offset),query});
  if(origin)params.set('origin',origin);
  $('moreArtworks').disabled=true;$('libraryStatus').textContent='正在读取作品…';
  try{
    const values=await api(`/api/artworks?${params}`);
    if(sequence!==libraryState.sequence||query!==$('artSearch').value.trim()||origin!==$('artOrigin').value)return;
    const existing=new Set(append?libraryState.items.map(item=>item.id):[]),fresh=values.filter(item=>!existing.has(item.id));
    libraryState.items=append?[...libraryState.items,...fresh]:values;
    libraryState.offset=offset+values.length;libraryState.more=values.length===24&&(!append||fresh.length>0);
    $('artGrid').innerHTML=libraryState.items.length?libraryState.items.map(a=>`<button class="art-card" data-art="${esc(a.id)}"><img loading="lazy" src="${esc(a.image_url)}" alt="作品 ${esc(a.id)}"><div><strong>${esc(a.id)}</strong><small>${esc(a.instruction||a.operation)}</small></div></button>`).join(''):'<p class="empty-copy">没有符合条件的作品。可更换关键词或作品来源。</p>';
    $('artGrid').querySelectorAll('[data-art]').forEach(button=>button.onclick=()=>openArtwork(button.dataset.art));
    $('libraryStatus').textContent=`已显示 ${libraryState.items.length} 张${libraryState.more?' · 可继续加载':' · 已到当前结果末尾'}`;
    if(append&&values.length&&!fresh.length)$('libraryStatus').textContent='服务返回了重复页面，请确认服务已更新后刷新重试。';
    $('moreArtworks').classList.toggle('hidden',!libraryState.more);
  }catch(error){if(sequence===libraryState.sequence)$('libraryStatus').textContent=`读取失败：${error.message}。已有作品保留，可刷新重试。`;}
  finally{if(sequence===libraryState.sequence)$('moreArtworks').disabled=false;}
}
function renderReviews(){const list=state.artwork?.reviews||[];$('reviews').innerHTML=list.length?list.map(r=>{const origin=reviewOriginLabel(r);return `<article class="review-entry">${origin?`<small>${esc(origin)}</small>`:''}<p>${esc(r.text)}</p></article>`;}).join(''):'<p class="empty-copy">还没有评价。直接用自然语言描述即可。</p>'}
let artworkRevision=0,maskRevision=0;
const reviewDrafts=new Map();
function updateArtworkCreativeActions(session){
  const id=state.artwork?.id;if(!id)return;
  const held=creativeUI.current?.();
  if(session?.id===state.artworkSessionId)state.artworkCreativeSession=session;
  const known=held?.id===state.artworkSessionId&&(!state.artworkCreativeSession||held.revision>=state.artworkCreativeSession.revision)?held:state.artworkCreativeSession;
  const maskReady=$('maskCanvas').dataset.artworkId===id;
  $('submitRepair').textContent='创建圈改版本';$('continueReview').textContent='按这段反馈继续';
  const taskActions=$('artTaskActions');if(taskActions)taskActions.hidden=true;
  if($('artOpenTask'))$('artOpenTask').disabled=true;
  if($('artSelectVersion'))$('artSelectVersion').disabled=true;
  if($('artRevertVersion'))$('artRevertVersion').disabled=true;
  if($('artContext'))$('artContext').textContent=`作品 ${id} · 正在核对所属任务`;
  if(!state.artworkSessionId){
    const unlinked=state.artworkSessionLookupStatus==='unlinked';$('continueReview').disabled=true;$('submitRepair').disabled=!unlinked||!maskReady;
    $('continueReviewNote').textContent=unlinked?'此旧作品没有关联创作任务；不会凭空补写原描述。保存评价与圈选修改仍可使用。':'作品的任务归属读取未完成或失败；未提交新的操作。';
    $('repairSessionNote').textContent=unlinked?'此旧作品沿用独立圈改入口；不补造创作任务或原句。会保存新版本，修改效果需要前后比较。':'作品归属读取失败或尚未完成，圈改未提交；重新打开后核对。';
    if($('artContext'))$('artContext').textContent=`作品 ${id} · ${unlinked?'独立作品，未关联创作任务':'所属任务尚未核对'}`;return;
  }
  if(!known||known.id!==state.artworkSessionId){
    $('continueReview').disabled=true;$('submitRepair').disabled=true;
    $('continueReviewNote').textContent='已知属于当前创作任务，但尚未读到真实状态与预算；重新读取后才可开始新的操作。';
    $('repairSessionNote').textContent='已知属于当前创作任务；状态、预算与源图尚待核对，不绕过任务派发。';return;
  }
  const view=CreativeSession.artworkState(known,id,{pending_action:creativeUI.controller?.saved?.()?.id===known.id?creativeUI.controller.saved().pending_action:null,busy:creativeUI.controller?.busy?.()});
  const {flags}=view;
  if($('artContext'))$('artContext').textContent=`作品 ${id} · ${view.belongs?`创作 ${known.id} · ${CreativeSession.statusLabel(known.status)}${view.selected?` · ${view.selectedLabel}`:''}`:'不属于所示任务，操作未启用'}`;
  if(taskActions)taskActions.hidden=!view.belongs;
  if($('artOpenTask'))$('artOpenTask').disabled=!view.belongs;
  if($('artSelectVersion')){$('artSelectVersion').disabled=!view.canChoose||(view.selected&&!view.selectedByAssistant);$('artSelectVersion').textContent=view.selected&&!view.selectedByAssistant?view.selectedLabel:'选用此版本';
    $('artSelectVersion').title=view.selectedByAssistant?'当前仅供助手功能验收使用；你可明确选用，不代表作品质量已认证。':'';}
  if($('artRevertVersion'))$('artRevertVersion').disabled=!view.canChoose||view.selected;
  const feedback=$('reviewText')?.value?.trim()||latestUserArtworkReview(state.artwork).trim();
  $('continueReview').disabled=!view.canContinue||!feedback;$('submitRepair').disabled=!view.canLocalEdit||!maskReady;
  const roundNote=flags.userOperationBudget?'开始一个新的有界回合，原图、原句与历史保留。':'';
  $('continueReview').textContent=(flags.userOperationBudget?'新回合 · ':'')+($('reviewText')?.value?.trim()?'按这段反馈继续':feedback?'按最新评价继续':'填写反馈后继续');
  $('submitRepair').textContent=flags.userOperationBudget?'新回合 · 创建圈改版本':'创建圈改版本';
  const fresh=known.status==='failed'?'原失败与响应保留；这是明确的新操作，不恢复或复跑失败步骤。':'';
  $('continueReviewNote').textContent=(view.canContinue?(feedback?'作品、原句与这段反馈一起传递，在同一任务里开始新接续；先观察再决定，不保证再生图。':'先写下希望保留和调整什么。保存评价只做记录；点击继续才会开始新的创作操作。'):flags.continueUnavailableReason||'当前作品不是已完成的真实候选。')+(view.canContinue?roundNote:'')+fresh;
  $('repairSessionNote').textContent=(view.canLocalEdit?'修改保留在同一创作任务中，形成待比较的新版本，不覆盖原图；消耗图片与时间预算，不需要作者调用，后续比较仍受调用预算约束。':flags.localEditUnavailableReason||'当前作品不是已完成的真实候选。')+(view.canLocalEdit?roundNote:'')+fresh;
}
function latestUserArtworkReview(artwork){return [...(artwork?.reviews||[])].reverse().find(review=>review&&(review.origin==='user'||!Object.prototype.hasOwnProperty.call(review,'origin'))&&typeof review.text==='string'&&review.text.trim())?.text||'';}
function reviewOriginLabel(review){if(review?.origin==='assistant_validation')return '助手验证 · 不代表用户评价';return review&&(review.origin==='user'||!Object.prototype.hasOwnProperty.call(review,'origin'))?'':'来源未核对';}
function renderArtworkVersions(art,line){
  const nodes=[...(line.ancestors||[]),art,...(line.children||[])];
  $('lineage').innerHTML=nodes.map(x=>`<button type="button" data-line="${esc(x.id)}" aria-pressed="${x.id===art.id}">${x.id===art.id?'正在查看 · ':''}${esc(CreativeSession.operationLabel(x.operation))}<small>${esc(x.id)}</small></button>`).join('');
  // Each different version resolves its own ownership; an ancestor need not
  // belong to the child's creative task or cooperative run.
  $('lineage').querySelectorAll('[data-line]').forEach(b=>b.onclick=()=>openArtwork(b.dataset.line));
  const parent=(line.ancestors||[]).find(x=>x.id===art.parent_artwork_id),compare=$('artVersionCompare');
  if(compare){compare.hidden=!parent;compare.open=false;
    const panel=$('artVersionImages');if(panel)panel.innerHTML=parent?[parent,art].map((x,index)=>{const href=CreativeSession.imageHref(x.image_url);return `<figure><figcaption>${index?'正在查看的版本':'直接来源版本'} · ${esc(x.id)}</figcaption>${href?`<a href="${esc(href)}" target="_blank" rel="noopener"><img src="${esc(href)}" alt="${index?'当前':'来源'}版本 ${esc(x.id)}" loading="lazy"></a>`:'<p>原图地址不可读取</p>'}</figure>`;}).join(''):'';}
  const original=$('artOriginal');if(original){const href=CreativeSession.imageHref(art.image_url);original.href=href;original.hidden=!href;original.download=`${art.id}.png`;}
}
async function openArtwork(id,context=null){
  if(state.artwork)reviewDrafts.set(state.artwork.id,$('reviewText').value);
  const revision=++artworkRevision;
  try{
    const [art,line]=await Promise.all([api(`/api/artworks/${id}`),api(`/api/artworks/${id}/lineage`)]);
    if(revision!==artworkRevision)return;
    if(context?.artworkId&&context.artworkId!==id)context=null;
    state.artwork=art;state.artworkContext=context?{...context,artworkId:id}:null;state.maskPoint=null;state.drawing=false;maskRevision++;
    state.artworkSessionId=context?.sessionId||null;state.artworkSessionLookupStatus=context?.sessionId?'linked':'pending';state.artworkCreativeSession=null;
    $('continueReview').disabled=true;$('continueReviewNote').textContent='正在核对关联创作任务；旧作品仍可评价与圈选修改。';
    $('repairSessionNote').textContent=state.artworkSessionId?'修改将保留在同一创作任务中，形成新候选，不覆盖原图。':'正在核对作品归属；完成后才可提交圈改。';
    $('reviewText').value=reviewDrafts.get(id)||'';
    $('maskCanvas').dataset.artworkId='';$('submitRepair').disabled=true;
    $('artImage').onload=()=>{if(state.artwork?.id===art.id&&$('repairTool').classList.contains('active'))setupMask();};
    $('artImage').src=art.image_url;$('artPath').textContent=art.file_path;$('artTitle').textContent=art.id;
    $('artPrompt').textContent=art.instruction||'没有记录原始描述';renderReviews();
    renderArtworkVersions(art,line);updateArtworkCreativeActions();
    $('evidence').textContent='尚未读取。';$('artMessage').classList.add('hidden');$('artDialog').showModal();
    creativeUI.forArtwork(art.id).then(session=>{if(revision!==artworkRevision)return;
      state.artworkSessionId=session?.id||context?.sessionId||null;state.artworkSessionLookupStatus=state.artworkSessionId?'linked':'unlinked';updateArtworkCreativeActions(session);
    }).catch(error=>{if(revision===artworkRevision){state.artworkSessionId=context?.sessionId||null;state.artworkSessionLookupStatus=state.artworkSessionId?'linked':'failed';updateArtworkCreativeActions();$('continueReviewNote').textContent+=' 暂时无法读取关联任务：'+error.message+'。未确认归属的作品不会改走旧端点提交。';}});
    if($('repairTool').classList.contains('active'))setupMask();
  }catch(error){if(revision===artworkRevision)alert(error.message);}
}
async function saveReview(){
  const id=state.artwork?.id,draft=$('reviewText').value,text=draft.trim(),context=state.artworkContext;
  if(!id||!text||$('saveReview').disabled)return;
  $('saveReview').disabled=true;reviewDrafts.set(id,draft);
  try{
    let savedReview;
    if(context?.runId)await CooperativeUI.review(id,text,context);
    else savedReview=await api(`/api/artworks/${encodeURIComponent(id)}/reviews`,{method:'POST',body:JSON.stringify({text,image_index:0})});
    if(reviewDrafts.get(id)===draft)reviewDrafts.delete(id);
    if(state.artwork?.id===id&&$('reviewText').value===draft)$('reviewText').value='';
    const art=await api(`/api/artworks/${encodeURIComponent(id)}`);
    if(state.artwork?.id===id){state.artwork=art;renderReviews();updateArtworkCreativeActions();message('artMessage',savedReview?.origin==='assistant_validation'?'助手验证原文已保存；不是用户评价，不会默认用于继续。需要继续时，请明确填写反馈。':'评价原文已保存；没有自动生成。需要调整时，可按这条评价继续。');}
    if(state.view==='research')loadExperiment();
  }catch(error){if(state.artwork?.id===id)message('artMessage',error.message,true);}
  finally{$('saveReview').disabled=false;}
}
async function continueReview(){
  const artwork=state.artwork,id=artwork?.id;if(!id||$('continueReview').disabled)return;
  const feedback=$('reviewText').value.trim()||latestUserArtworkReview(artwork);
  $('continueReview').disabled=true;
  try{
    const record=await creativeUI.continueArtwork(id,feedback,state.artworkSessionId);
    if(record&&state.artwork?.id===id){$('artDialog').close();$('generationStrategy').value='creative';updateIterationStrategyNote();}
  }catch(error){if(state.artwork?.id===id)message('artMessage',error.message,true);}
  finally{if(state.artwork?.id===id)updateArtworkCreativeActions();}
}
async function setupMask(){
  const id=state.artwork?.id,revision=++maskRevision,img=$('artImage');$('submitRepair').disabled=true;
  try{
    await img.decode();if(revision!==maskRevision||state.artwork?.id!==id)return;
    const image=$('imageCanvas'),mask=$('maskCanvas'),w=img.naturalWidth,h=img.naturalHeight;
    if(!w||!h)throw new Error('原图尚未加载完成');
    for(const canvas of[image,mask]){canvas.width=w;canvas.height=h;}
    image.getContext('2d').drawImage(img,0,0,w,h);mask.dataset.artworkId=id;
    state.maskPoint=null;updateArtworkCreativeActions();
  }catch(error){if(revision===maskRevision)message('artMessage','无法准备圈选画布，请重新打开图片：'+error.message,true);}
}
function paintMask(event){
  const c=$('maskCanvas');if(!state.drawing||c.dataset.artworkId!==state.artwork?.id)return;
  const rect=c.getBoundingClientRect(),x=(event.clientX-rect.left)*c.width/rect.width,y=(event.clientY-rect.top)*c.height/rect.height;
  const ctx=c.getContext('2d'),diameter=Number($('repairBrushSize').value);
  if(event.type==='pointerdown')state.maskPoint=null;
  ctx.fillStyle=ctx.strokeStyle='rgba(255,255,255,.92)';ctx.lineWidth=diameter;ctx.lineCap='round';ctx.lineJoin='round';
  ctx.beginPath();if(state.maskPoint){ctx.moveTo(...state.maskPoint);ctx.lineTo(x,y);ctx.stroke();}else{ctx.arc(x,y,diameter/2,0,Math.PI*2);ctx.fill();}
  state.maskPoint=[x,y];
}
async function submitRepair(){
  const id=state.artwork?.id;
  try{
    if(!id)throw new Error('请先打开实际作品');
    const instruction=$('repairInstruction').value;if(!instruction.trim())throw new Error('请说明圈选区域怎样修改');
    if($('maskCanvas').dataset.artworkId!==id)throw new Error('请等待当前图片加载后重新圈选');
    if(['pending','failed'].includes(state.artworkSessionLookupStatus))throw new Error('作品的任务归属尚未核验；请重新打开图片核对，不会盲改走旧端点。');
    $('submitRepair').disabled=true;
    const steps=Number($('repairSteps').value);if(!Number.isInteger(steps)||steps<5||steps>40)throw new Error('局部修改步数须为 5–40 的整数');
    const payload={instruction,mask_data_url:$('maskCanvas').toDataURL('image/png'),denoise:Number($('repairDenoise').value)/100,steps};
    const sessionId=state.artworkSessionId||state.artworkContext?.sessionId;
    if(sessionId){
      const record=await creativeUI.localEditArtwork(id,payload,sessionId);
      if(record&&state.artwork?.id===id)$('artDialog').close();
      $('generationStrategy').value='creative';updateIterationStrategyNote();
      return;
    }
    const context=globalThis.CooperativeUI?.editContext(id)||{};
    const job=await api(`/api/artworks/${encodeURIComponent(id)}/region-repair`,{method:'POST',body:JSON.stringify({...payload,...context})});
    if(context.adaptive_run_id){if(state.artwork?.id===id)$('artDialog').close();selectView('create');await CooperativeUI.refresh(context.adaptive_run_id);return;}
    message('artMessage',`局部修改任务 ${job.id} 已加入队列`);$('artDialog').close();state.job=job;
    showSession('局部修改',20,'保存前会校验圈外像素；圈内修改质量仍需比较。');selectView('create');pollJob(job.id);
  }catch(error){if(state.artwork?.id===id)message('artMessage',error.message,true);}
  finally{if(state.artwork?.id===id)updateArtworkCreativeActions();}
}
async function submitText(){
  const id=state.artwork?.id,artContext=state.artworkContext;
  try{const content=$('exactText').value;if(!content.trim())throw new Error('请输入精确文字');
    const context=globalThis.CooperativeUI?.editContext(id)||{};$('submitText').disabled=true;
    const art=await api(`/api/artworks/${encodeURIComponent(id)}/text-layer`,{method:'POST',body:JSON.stringify({content,x:Number($('textX').value),y:Number($('textY').value),font_size:Number($('fontSize').value),width:.7,height:.18,align:'center',...context})});
    if(context.adaptive_run_id)await CooperativeUI.refresh(context.adaptive_run_id);
    if(state.artwork?.id===id)await openArtwork(art.id,artContext?{runId:artContext.runId}:null);await loadArtworks();
  }catch(error){if(state.artwork?.id===id)message('artMessage',error.message,true);}finally{$('submitText').disabled=false;}
}
async function loadWorkflow(){const artwork=state.artwork;try{if(!artwork?.job_id)throw new Error('此版本没有关联生成任务，无法提供执行工作流。');const base=`/api/jobs/${encodeURIComponent(artwork.job_id)}`;const result=await api(artwork.source_stage?`${base}/stages/${encodeURIComponent(artwork.source_stage.id)}/workflow`:`${base}/workflow`);if(state.artwork?.id!==artwork.id)return;PipelineDiagram.mountEvidence($('evidence'),result);if(result.warning)$('evidence').prepend(document.createTextNode(result.warning));const link=document.createElement('a');link.href=`/pipeline?job=${encodeURIComponent(artwork.job_id)}`;link.target='_blank';link.rel='noopener';link.textContent='在独立管线页查看任务整体记录 ↗';$('evidence').prepend(link)}catch(error){if(state.artwork?.id===artwork?.id)$('evidence').textContent=error.message}}
function duration(sec){if(sec==null)return'未知';const m=Math.ceil(sec/60);return m<60?`${m} 分钟`:`${Math.floor(m/60)} 小时 ${m%60} 分`}
async function loadResearch(){try{if(!state.boundary){const [plan,checks]=await Promise.all([api('/api/research/boundary-plan'),api('/api/check-capabilities')]);state.boundary=plan;state.checks=checks;$('boundaryCount').textContent=`${plan.total_items} 个测试单元`;$('boundaryBadge').textContent=`${plan.total_items} 项`;$('boundarySummary').textContent=`${plan.texts.length} 条文本 × ${plan.recipes.length} 条完整路线 × ${plan.seeds.length} Seed，另含 ${plan.focused_items} 项协作强度扫描。`;$('boundaryTexts').innerHTML=plan.texts.map((x,i)=>`<article><strong>${i+1}. ${esc(x.character)} · ${esc(x.boundary)}</strong><p>${esc(x.sentence)}</p></article>`).join('');const anatomy=checks.hand_anatomy_judgement;$('checkHeadline').textContent=anatomy.status==='unavailable'?'定位可用 · 解剖需人工':'候选检测待校准';$('capability').textContent=`手部定位：${checks.hand_localization.status}；手指与关节：${anatomy.status==='unavailable'?'不能可靠自动判断':'候选待校准'}；严格 OOC 与审美质量仍需人工评价。`;}state.experiments=await api('/api/experiments');const old=$('experimentSelect').value;$('experimentSelect').innerHTML=state.experiments.length?state.experiments.map(x=>`<option value="${x.id}">${esc(x.name)} · ${x.finished}/${x.total}</option>`).join(''):'<option value="">尚无实验</option>';if(old&&state.experiments.some(x=>x.id===old))$('experimentSelect').value=old;renderStudy();if($('experimentSelect').value)await loadExperiment()}catch(error){$('studyName').textContent='研究数据读取失败';$('studyMeta').textContent=error.message}}
function renderStudy(){const x=state.experiments.find(e=>e.id===$('experimentSelect').value)||state.experiments.find(e=>e.state==='running')||state.experiments[0];if(!x){$('studyName').textContent='还没有实验';$('studyMeta').textContent='在设计页创建一个暂停实验。';return}$('experimentSelect').value=x.id;$('studyName').textContent=x.name;$('studyMeta').textContent=`${x.finished}/${x.total} 执行结束 · ${x.state} · ${x.review_count||0} 条评价 · ${x.quality_evidence?.warning_items??'—'} 项有检查警告 · 执行完成不代表质量合格`}
async function loadExperiment(){return refreshSelectedExperiment();}
async function experimentAction(action){if(!state.experiment)return;try{await api(`/api/experiments/${state.experiment.id}/${action}`,{method:'POST'});await loadResearch()}catch(error){alert(error.message)}}
function metricCard(row){return`<article><strong>${esc(row.value)}</strong><small>完成 ${Math.round((row.completion_rate||0)*100)}% · 评价覆盖 ${Math.round((row.review_coverage||0)*100)}% · 正向摘录 ${row.positive_mentions} · 问题摘录 ${row.problem_mentions}</small></article>`}
let analysisRevision=0;
async function loadAnalysis(){
  const id=$('experimentSelect').value,revision=++analysisRevision;if(!id)return;
  try{
    const a=await api(`/api/experiments/${id}/analysis`),s=a.summary;
    if(revision!==analysisRevision||$('experimentSelect').value!==id)return;
    $('analysisSummary').textContent=`${s.completed}/${s.total} 项完成 · ${s.completed_images??s.completed} 张图片 · ${s.reviewed_images} 张有评价 · 覆盖 ${Math.round(s.review_coverage*100)}%。${s.warning||'描述统计不等于模型能力结论。'}`;
    $('analysisGroups').innerHTML=[['recipe','实验配方'],['pipeline','执行路线'],['model','请求主模型（协作任务不止一个模型）'],['scene','场景'],['prompt_form','提示词形式']].map(([k,n])=>`<section><h3>${n}</h3><div>${(a.groups[k]||[]).map(metricCard).join('')||'<p>无数据</p>'}</div></section>`).join('');
    $('analysisReviews').innerHTML=(a.review_evidence||[]).map(entry=>{const origin=reviewOriginLabel(entry.review);return `<article><header><strong>${esc(entry.item_id)} · ${esc(entry.axes.recipe||entry.axes.pipeline)}</strong>${entry.artwork_id?`<button data-review-art="${esc(entry.artwork_id)}">查看原图</button>`:''}</header>${origin?`<small>${esc(origin)}</small>`:''}<blockquote>${esc(entry.review.text)}</blockquote><small>自动检索索引（不是质量裁决）：${esc((entry.review.analysis?.facets||[]).join(' / ')||'未归类')}。请以原文为准。</small></article>`;}).join('')||'<p>本实验尚无自然语言评价。未评价不代表质量好或差。</p>';
    $('analysisReviews').querySelectorAll('[data-review-art]').forEach(button=>button.onclick=()=>openArtwork(button.dataset.reviewArt));
  }catch(error){if(revision===analysisRevision&&$('experimentSelect').value===id)$('analysisSummary').textContent=error.message;}
}
async function createBoundary(){try{const x=await api('/api/research/boundary-plan/create',{method:'POST',body:JSON.stringify({confirmed:$('confirmBoundary').checked})});await loadResearch();$('experimentSelect').value=x.id;await loadExperiment();selectLab('run')}catch(error){alert(error.message)}}
async function importManifest(){try{const x=await api('/api/experiments',{method:'POST',body:JSON.stringify(JSON.parse($('manifest').value))});await loadResearch();$('experimentSelect').value=x.id;await loadExperiment();selectLab('run')}catch(error){alert(`清单无效：${error.message}`)}}
async function init(){document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>selectView(b.dataset.view));document.querySelectorAll('[data-lab]').forEach(b=>b.onclick=()=>selectLab(b.dataset.lab));document.querySelectorAll('[data-jump]').forEach(b=>b.onclick=()=>selectLab(b.dataset.jump));document.querySelectorAll('[data-close]').forEach(b=>b.onclick=()=>$((b.dataset.close)).close());document.querySelectorAll('[data-tool]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-tool]').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.tool').forEach(x=>x.classList.toggle('active',x.id===`${b.dataset.tool}Tool`));if(b.dataset.tool==='repair')setupMask()});$('make').onclick=beginCreation;$('confirmMake').onclick=submitCreation;$('previewGraph').onclick=previewGraph;$('reloadArtworks').onclick=loadArtworks;$('openSessionArtwork').onclick=()=>openArtwork($('openSessionArtwork').dataset.id);$('saveReview').onclick=saveReview;$('submitRepair').onclick=submitRepair;$('submitText').onclick=submitText;$('loadWorkflow').onclick=loadWorkflow;$('clearMask').onclick=()=>$('maskCanvas').getContext('2d').clearRect(0,0,$('maskCanvas').width,$('maskCanvas').height);const mask=$('maskCanvas');mask.onpointerdown=e=>{state.drawing=true;mask.setPointerCapture(e.pointerId);paintMask(e)};mask.onpointermove=paintMask;mask.onpointerup=()=>state.drawing=false;mask.onpointercancel=()=>state.drawing=false;$('confirmBoundary').onchange=()=>$('createBoundary').disabled=!$('confirmBoundary').checked;$('createBoundary').onclick=createBoundary;$('importManifest').onclick=importManifest;$('experimentSelect').onchange=()=>{renderStudy();loadExperiment();if(state.lab==='analysis')loadAnalysis()};document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>experimentAction(b.dataset.action));$('refreshAnalysis').onclick=loadAnalysis;$('make').disabled=true;const [health,styles]=await Promise.allSettled([api('/api/health'),api('/api/style-presets')]);if(health.status==='fulfilled'){$('engine').className='engine ok';$('engine').querySelector('span').textContent='服务在线'}else{$('engine').className='engine bad';$('engine').querySelector('span').textContent='服务异常'}if(styles.status==='fulfilled'){state.styles=styles.value;populateStyles()}await initCreationControls().catch(error=>message('createMessage',error.message,true));$('make').disabled=!creationCatalog;await loadArtworks();const q=new URLSearchParams(location.search);if(q.get('view')==='research')selectView('research')}
$('rebuildContract').onclick=rebuildContract;
$('openPromptReview').onclick=()=>{if($('generationStrategy').value==='creative')creativeUI.review();else openPromptReview();};
$('cancelPreparation').onclick=cancelPreparation;
$('adoptSource').onclick=()=>continueSourceReview('adopt');
$('clarifySource').onclick=()=>continueSourceReview('clarify');
function creativeCycleSettingsForNewRequest(){
  if(!$('proMode').checked)return {result_guided:true,max_jobs:6,max_calls:8};
  const max_jobs=Number($('creativeMaxJobs').value),max_calls=Number($('creativeMaxCalls').value);
  if(!Number.isInteger(max_jobs)||max_jobs<1||max_jobs>6)throw new Error('候选预算须为 1–6 的整数；未提交创作。');
  if(!Number.isInteger(max_calls)||max_calls<1||max_calls>8)throw new Error('模型调用预算须为 1–8 的整数；未提交创作。');
  return {result_guided:$('creativeResultGuided').checked,max_jobs,max_calls};
}
function creativeCycleSettingsNote(record){
  const snapshot=record?.input_snapshot;
  const context=!record?'':typeof snapshot?.result_guided==='boolean'
    ?`正在读取任务 ${record.id}；其已保存的结果回合${snapshot.result_guided?'开启':'关闭'}，上限 ${record.limits?.jobs??snapshot.max_jobs??'未记录'} / ${record.limits?.calls??snapshot.max_calls??'未记录'}。`
    :`正在读取旧任务 ${record.id}；它未声明自动结果接续，读取不会为它开启。下方勾选默认仅用于新任务。`;
  $('creativeCycleSettingsNote').textContent=context+'这些控制仅约束下一次新建创作，不修改现有任务；普通模式采用开启与 6 / 8 默认。暂停或停止请使用任务区操作。';
}
function restoreCreativeCycleSettings(record){
  creativeCycleSettingsNote(record);
  if(!record?.id||state.creativeCycleRestoredId===record.id)return;
  const firstRead=!state.creativeCycleRestoredId;
  state.creativeCycleRestoredId=record.id;
  if(firstRead&&state.creativeCycleDraftTouched)return;
  state.creativeCycleDraftTouched=false;
  const snapshot=record.input_snapshot||{};
  if(typeof snapshot.result_guided==='boolean'){
    $('creativeResultGuided').checked=snapshot.result_guided;
    // These controls declare the NEXT new task's per-round allowance. limits
    // grows cumulatively after explicit user operations and is display-only;
    // importing it here can exceed the controls/API's 6-image/8-call bounds.
    $('creativeMaxJobs').value=String(snapshot.max_jobs??6);
    $('creativeMaxCalls').value=String(snapshot.max_calls??8);
  }else{
    $('creativeResultGuided').checked=true;$('creativeMaxJobs').value='6';$('creativeMaxCalls').value='8';
  }
}
function resetCreativeCycleSettingsForNextRequest(){
  $('creativeResultGuided').checked=true;$('creativeMaxJobs').value='6';$('creativeMaxCalls').value='8';
  state.creativeCycleDraftTouched=true;creativeCycleSettingsNote(state.creativeSession);
}
const creativeUI=CreativeSession.mount({document,read:async(path,options={})=>{
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),20000);
  try{return await api(path,{...options,signal:controller.signal});}finally{clearTimeout(timer);}
},storage:localStorage,showSession,loadArtworks,openArtwork,selectView,onRecord(record){state.creativeSession=record;
  if($('composerContextNote')){$('composerContextNote').hidden=false;$('composerContextNote').textContent=`正在查看创作 ${record.id}。下方描述和专业设置用于新建创作，不改这份记录；从某张图继续，请打开该图的评价或圈改入口。`;}
  restoreCreativeCycleSettings(record);
  if(typeof updatePictureAuthoringControls==='function')updatePictureAuthoringControls(record);
  if(state.artworkSessionId===record.id)updateArtworkCreativeActions(record);
  if($('generationStrategy').value==='creative')$('openPromptReview').disabled=!CreativeSession.reviewState(record).canReview;
}});
globalThis.CreativeSessionUI=creativeUI;
$('continueReview').onclick=continueReview;
$('reviewText').oninput=()=>updateArtworkCreativeActions();
async function artworkTaskAction(action){
  const id=state.artwork?.id,sessionId=state.artworkSessionId;
  const button=$({open:'artOpenTask',select:'artSelectVersion',revert:'artRevertVersion'}[action]);
  if(!id||!sessionId||!button||button.disabled)return;
  button.disabled=true;
  try{
    const record=action==='open'?await creativeUI.openArtworkTask(id,sessionId)
      :await creativeUI.chooseArtwork(id,sessionId,{revert:action==='revert'});
    if(record&&state.artwork?.id===id)$('artDialog').close();
  }catch(error){if(state.artwork?.id===id)message('artMessage',error.message,true);}
  finally{if(state.artwork?.id===id)updateArtworkCreativeActions();}
}
$('artOpenTask').onclick=()=>artworkTaskAction('open');
$('artSelectVersion').onclick=()=>artworkTaskAction('select');
$('artRevertVersion').onclick=()=>artworkTaskAction('revert');
$('artRefresh').onclick=()=>{if(state.artwork)openArtwork(state.artwork.id,state.artworkContext);};
let creativeHistorySequence=0;
async function loadCreativeHistory(){
  const version=++creativeHistorySequence,limit=$('creativeHistoryLimit').value||'20';
  $('creativeHistoryStatus').textContent='正在读取创作记录…';$('creativeHistoryRefresh').disabled=true;
  try{
    const rows=await api(`/api/creative-sessions?limit=${encodeURIComponent(limit)}`);
    if(version!==creativeHistorySequence)return;
    if(!Array.isArray(rows))throw new Error('创作记录响应不是列表');
    $('creativeHistoryList').innerHTML=CreativeSession.historyHtml(rows);
    $('creativeHistoryStatus').textContent=`最近 ${rows.length} 条创作 · 按创建时间排列 · 只读取，不恢复运行`;
    for(const b of $('creativeHistoryList').querySelectorAll('[data-open-creative-history]'))b.onclick=async()=>{
      b.disabled=true;
      try{const record=await creativeUI.restore(b.dataset.openCreativeHistory);
        if(record){globalThis.CooperativeUI?.begin();selectView('create');$('creativeHistoryDialog').close();}}
      catch(error){$('creativeHistoryStatus').textContent=`读取未确认：${error.message}。没有恢复或重派。`;}
      finally{b.disabled=false;}
    };
  }catch(error){if(version===creativeHistorySequence)$('creativeHistoryStatus').textContent=`读取失败：${error.message}。已有记录保留，可重新读取。`;}
  finally{if(version===creativeHistorySequence)$('creativeHistoryRefresh').disabled=false;}
}
function connectCreativeHistory(){
  const legacy=$('openCooperative'),entry=$('legacyHistoryEntry');
  if(legacy&&entry&&legacy.parentElement!==entry){
    const openLegacy=legacy.onclick;legacy.textContent='查看旧协作记录';
    legacy.onclick=()=>{$('creativeHistoryDialog').close();openLegacy?.();};entry.append(legacy);
  }
  $('openCreativeHistory').onclick=()=>{$('creativeHistoryDialog').showModal();loadCreativeHistory();};
  $('creativeHistoryRefresh').onclick=loadCreativeHistory;$('creativeHistoryLimit').onchange=loadCreativeHistory;
}
async function beginCreativeCreation(){
  const previous=preparationObserver.saved?.(),previousReceipt=previous?.id&&previous.record?.id===previous.id?AuthorPreparation.receipt(previous.record):null;
  if(previous?.outcome_unknown||['unknown','submitting'].includes(previousReceipt?.status)
      ||(previous?.meta?.dispatch?.status==='unknown'
      &&!(previousReceipt?.status==='submitted'&&(previousReceipt.job_id||previousReceipt.run_id)))){
    message('createMessage','旧准备或生图提交回执尚未确认；请先读取核对原记录，不跨入口另开同一构想。没有自动恢复或重派。',true);return;
  }
  const current=creativeUI.current();
  if(current&&['queued','running','needs_confirmation','paused'].includes(current.status)&&!confirm('当前任务尚未结束。另开一个创作任务？原任务会保留，不会自动停止或重跑。'))return;
  state.preparationBusy=true;$('make').disabled=true;
  try{
    if(creationCatalog?.capabilities?.creative_initial_selection!==true
      ||creationCatalog?.capabilities?.sampling_parameter_sources!==true)
      throw new Error('后端尚未载入本轮选路更新，需重启项目服务；没有提交新创作或删除采样来源。旧任务仍可查看、评价和停止。');
    const professional=$('proMode').checked;
    const payload={...compilePayload(),auto_generate:!professional,
      picture_authoring_method:CreativeSession.authoringMethodForNewRequest({professional,method:$('creative-picture-authoring-method')?.value,capabilities:creationCatalog.capabilities}),
      authoring_profile:CreativeSession.authoringProfileForNewRequest({professional,profile:$('creative-authoring-profile')?.value}),
      parameter_sources:CreativeSession.samplingParameterSources(professional),...creativeCycleSettingsForNewRequest(),
      parameter_scope:professional?$('creativeParameterScope')?.value||'initial':'initial'};
    const requestInput=CreativeSession.input(payload);
    if(requestInput.research_mode==='catalog'&&creationCatalog.capabilities.creative_research_catalog!==true)
      throw new Error('当前服务尚未载入按需研究读取更新，请载入后刷新；旧作品不会重跑。');
    if((requestInput.joint_picture_decision||requestInput.parameter_scope!==undefined)&&creationCatalog.capabilities.creative_joint_picture_decision!==true)
      throw new Error('当前服务尚未载入创作参数与完整画面共同决定的更新，请等待本次更新载入后刷新；现有作品仍可查看和操作。');
    if(requestInput.understand_request&&creationCatalog.capabilities.creative_request_understanding!==true)
      throw new Error('后端尚未载入原句理解更新，请载入更新后刷新再新建创作；原有记录仍可查看和操作，未自动补做理解或提交新任务。');
    const sampling=CreativeSession.samplingValuesForNewRequest(creationCatalog,{professional,values:professionalValues(),parameter_sources:payload.parameter_sources});
    // Historical/programmatic display is not an edit. Sync only defaults;
    // the actual user-edited controls and their provenance remain untouched.
    setValues(Object.fromEntries(Object.entries(sampling).filter(([field])=>payload.parameter_sources[field]==='model_default')));
    payload.config=resolvedCreationConfig();
    state.preparationRestoreContext=null;state.restoringPreparationHistory=false;
    state.creationPresentationRevision=(state.creationPresentationRevision||0)+1;
    if(typeof jobWatcher!=='undefined')jobWatcher.stop();preparationObserver.stop();globalThis.CooperativeUI?.begin();
    $('sourceReview').hidden=true;$('createMessage').classList.add('hidden');$('openPromptReview').disabled=true;
    await creativeUI.start(payload);
  }catch(error){if(error.creationErrorSource==='local_config')showLocalConfigError(error);else message('createMessage',error.message,true);}
  finally{state.preparationBusy=false;$('make').disabled=!creationCatalog;}
}
let creativeRestorePromise=null;
function restoreCreativeSession(){
  const requested=new URLSearchParams(location.search).get('session');
  if(new URLSearchParams(location.search).get('run')&&!requested)return Promise.resolve(null);
  // Empty startup is also a completed restore decision. A late library read
  // must not restore a new task's in-flight submission and invalidate its receipt.
  if(!requested&&!creativeUI.hasSaved())return creativeRestorePromise??=Promise.resolve(null);
  if(!creativeRestorePromise){$('generationStrategy').value='creative';creativeRestorePromise=creativeUI.restore(requested||undefined).catch(error=>{
    message('createMessage',error.message,true);showSession(error.status===404?'任务不存在':'任务读取未确认',null,'没有重新提交；可点击重新读取核对实际状态。','creative');return null;
  });}
  return creativeRestorePromise;
}
function updateIterationStrategyNote(){
  syncCreationEntryMode();
  const field=$('generationStrategy'),professional=$('proMode').checked;
  const cooperative=field.querySelector('option[value="cooperative"]');if(cooperative)cooperative.disabled=!professional;
  if(!professional&&field.value==='cooperative')field.value='creative';
  $('localIterationSettingsNote').classList.toggle('hidden',field.value!=='local_iteration');
  const output=$('quality').querySelector('option[value="iteration_base"]');if(output)output.disabled=field.value!=='local_iteration';
  if(creationCatalog)updateConfigContext();
}
$('generationStrategy').addEventListener('change',updateIterationStrategyNote);
$('proMode').addEventListener('change',updateIterationStrategyNote);
$('resetSettings').addEventListener('click',()=>{resetCreativeCycleSettingsForNextRequest();updateIterationStrategyNote();reconcileLocalConfigError();});
for(const id of ['creativeResultGuided','creativeMaxJobs','creativeMaxCalls'])for(const type of ['input','change'])$(id).addEventListener(type,()=>{state.creativeCycleDraftTouched=true;creativeCycleSettingsNote(state.creativeSession);});
for(const type of ['input','change'])$('createView').addEventListener(type,event=>{state.inputTouched=true;state.creationPresentationRevision=(state.creationPresentationRevision||0)+1;if(event.target.matches('input,select'))reconcileLocalConfigError();});
const jobReadStatus=document.createElement('div');jobReadStatus.className='job-read-status';jobReadStatus.innerHTML='<p id="jobConnection" role="status"></p><p id="runtimeConnection"></p><button id="retryJobRead" class="hidden">重新读取任务</button>';
$('sessionLive').append(jobReadStatus);
$('retryJobRead').onclick=()=>{if(state.job?.id)pollJob(state.job.id);};
$('inspectJobReceipt').onclick=()=>performJobReceipt('inspect');
$('cancelJobReceipt').onclick=()=>performJobReceipt('cancel');
$('artSearch').oninput=()=>{clearTimeout(libraryState.timer);libraryState.timer=setTimeout(()=>loadArtworks(),250);};
$('artOrigin').onchange=()=>loadArtworks();
$('moreArtworks').onclick=()=>loadArtworks({append:true});
document.querySelectorAll('[data-prompt-tab]').forEach(button=>button.onclick=()=>selectPromptTab(button.dataset.promptTab));
selectPromptTab('author');
selectLab('run');
const repairBrush=document.createElement('label');
repairBrush.innerHTML='笔刷直径（原图像素） <output id="repairBrushValue">24 px</output><input id="repairBrushSize" type="range" min="2" max="160" value="24" aria-label="笔刷直径">';
$('repairTool').querySelector('.mask-wrap').before(repairBrush);
$('repairBrushSize').oninput=()=>$('repairBrushValue').textContent=$('repairBrushSize').value+' px';
$('repairDenoise').oninput=()=>$('repairDenoiseValue').textContent=(Number($('repairDenoise').value)/100).toFixed(2);
const repairEngineNote=document.createElement('p');repairEngineNote.textContent='当前使用 NewBie 修补，其他模型原图会适配采样参数；圈外像素校验通过才会保存，圈内效果需要你比较确认。';
repairBrush.before(repairEngineNote);
const initialLocation=new URLSearchParams(location.search);
init().then(async()=>{connectCreativeHistory();updateIterationStrategyNote();
  const creative=await restoreCreativeSession();
  if(creative&&creationCatalog){await restorePreparationInputs({input:creative.input_snapshot,meta:{strategy:'creative',professional:creative.input_snapshot?.auto_generate===false,quality:creative.input_snapshot?.config?.highres_enabled?'high':'standard',style:creative.input_snapshot?.config?.style_preset||''}});updateIterationStrategyNote();}
  else if(creationCatalog&&!creativeUI.hasSaved()&&!initialLocation.get('session'))restoreCreationPreparation().then(updateIterationStrategyNote);
  if(initialLocation.get('view')==='library')selectView('library');const artwork=initialLocation.get('artwork');if(artwork)openArtwork(artwork);
}).catch(error=>{console.error(error);$('engine').className='engine bad';$('engine').querySelector('span').textContent='初始化失败'});
// Tracking must not wait for optional catalog/health/library endpoints to finish.
window.addEventListener('DOMContentLoaded',restoreTrackedJob,{once:true});
window.addEventListener('DOMContentLoaded',restoreCreativeSession,{once:true});
