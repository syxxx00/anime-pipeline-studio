/* One creative task, actual candidates and explicit user actions. Restores are
   GET-only; an uncertain dispatch receipt is never silently submitted again. */
(function(root){
  const storageKey='pipeline-studio.creative-session';
  const esc=value=>String(value??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const stable=value=>Array.isArray(value)?value.map(stable):value&&typeof value==='object'
    ?Object.fromEntries(Object.keys(value).sort().map(key=>[key,stable(value[key])])):value;
  const key=value=>JSON.stringify(stable(value));
  const requestId=()=>root.crypto?.randomUUID?.()||`creative-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const samplingFields=['steps','cfg','sampler','scheduler','shift','cfg_rescale'];
  const samplingControlIds={steps:'setting-steps',cfg:'setting-cfg',sampler:'samplerChoice',scheduler:'schedulerChoice',shift:'setting-shift',cfg_rescale:'setting-cfg_rescale'};
  const samplingTrackers=new WeakMap();let activeSamplingDocument=null;
  const pictureMethods={prose:'完整画面描述',whole:'结构化画面选择',relational_steps:'分段关系构思'};
  const authoringProfiles={local_4b:'本地 4B',local_7b:'本地 7B（旧 8K/mmap 配置）',local_7b_bounded:'本地 7B（受限日常配置）'};
  function pictureAuthoringMethod(record){return record?.input_snapshot?.picture_authoring_method||'whole';}
  function pictureMethodLabel(method){return pictureMethods[method]||String(method||'未记录');}
  function authoringProfile(record){return record?.input_snapshot?.authoring_profile??'local_4b';}
  function authoringProfileLabel(profile){return authoringProfiles[profile]||String(profile??'未记录');}
  function authoringProfileForNewRequest({professional,profile}={}){
    if(typeof professional!=='boolean')throw new Error('构思模型需要明确的当前模式');
    const selected=professional?(profile||'local_4b'):'local_4b';
    if(!['local_4b','local_7b_bounded'].includes(selected))throw new Error('请选择当前的构思模型配置；旧 7B 入口请刷新后明确选择受限日常配置，没有转换历史配置');
    return selected;
  }
  function authoringMethodForNewRequest({professional,method,capabilities}={}){
    if(typeof professional!=='boolean')throw new Error('画面组织需要明确的当前模式');
    const selected=professional?(method||'prose'):'prose';
    if(!Object.prototype.hasOwnProperty.call(pictureMethods,selected))throw new Error('请选择有效的画面组织方式');
    if(selected==='prose'&&capabilities?.creative_prose_picture!==true)
      throw new Error('后端尚未载入完整画面描述更新，请载入项目更新后刷新；专业模式仍可明确选择结构化画面选择或分段关系构思，原路线与研究入口保留。没有提交未知方法。');
    return selected;
  }
  function samplingParameterSources(professional){
    if(typeof professional!=='boolean')throw new Error('新创作采样来源需要明确的当前模式');
    const doc=activeSamplingDocument,edits=doc&&samplingTrackers.get(doc);
    return Object.fromEntries(samplingFields.map(field=>{
      const edit=professional&&edits?.get(field),control=doc?.getElementById(samplingControlIds[field]);
      // Values only verify an already recorded edit is still the same draft.
      // They never infer user origin by comparing with catalogue defaults.
      const explicit=Boolean(edit&&edit.control===control&&edit.value===control.value);
      if(professional&&edit&&!explicit)edits.delete(field);
      return [field,explicit?'user_override':'model_default'];
    }));
  }
  function clearSamplingParameterSources(fields=samplingFields){
    const edits=activeSamplingDocument&&samplingTrackers.get(activeSamplingDocument);
    for(const field of fields)if(samplingFields.includes(field))edits?.delete(field);
  }
  function samplingValuesForNewRequest(catalog,{professional,values={},parameter_sources}={}){
    if(typeof professional!=='boolean')throw new Error('新建采样需要明确的当前模式');
    const routeId=professional?(values.render_pipeline||catalog?.defaults?.render_pipeline):catalog?.defaults?.render_pipeline;
    const route=catalog?.routes?.find(item=>item.id===routeId);
    if(!route?.profile)throw new Error('所选起点没有当前模型配置，未猜测新建参数');
    const schemas=new Map((catalog.groups||[]).flatMap(group=>group.fields||[]).map(field=>[field.key,field.schema]));
    return Object.fromEntries(samplingFields.map(field=>{
      const source=parameter_sources?.[field];
      if(!['model_default','user_override'].includes(source))throw new Error('新建采样需明确六项参数的真实来源');
      if(source==='user_override')return [field,values[field]];
      const value=Object.prototype.hasOwnProperty.call(route.profile,field)?route.profile[field]:schemas.get(field)?.default;
      if(value==null)throw new Error(`${field}缺少当前模型或配置声明的默认值，未猜测新建参数`);
      return [field,value];
    }));
  }
  function bindSamplingParameterSources(doc){
    if(!doc?.addEventListener)return;
    activeSamplingDocument=doc;
    if(samplingTrackers.has(doc))return;
    const edits=new Map();samplingTrackers.set(doc,edits);
    const recordEdit=event=>{
      if(event.isTrusted!==true||!doc.getElementById('proMode')?.checked)return;
      const control=event.target,field=samplingFields.find(name=>samplingControlIds[name]===control?.id);
      if(!field||control.disabled||control.readOnly||doc.getElementById(control.id)!==control)return;
      edits.set(field,{control,value:control.value});
    };
    for(const type of ['input','change'])doc.addEventListener(type,recordEdit,true);
  }
  function target(config={}){return {anima_turbo:'anima',anima:'anima',z_image_turbo:'z_image',z_image:'z_image'}[config.model_id]||config.model_id||'newbie';}
  function input(value){
    if(!value?.sentence?.trim())throw new Error('请先描述你想要的画面');
    const support=Object.prototype.hasOwnProperty.call(value,'support_mode')?value.support_mode:'auto';
    const backend=Object.prototype.hasOwnProperty.call(value,'picture_authoring_backend')?value.picture_authoring_backend:'text';
    if(!['auto','text','sketch','compare'].includes(support))throw new Error('请选择有效的构思支持方式');
    if(!['text','paired_visual'].includes(backend))throw new Error('请选择有效的画面构思入口');
    const result={sentence:value.sentence,config:structuredClone(value.config||{}),style_mode:value.style_mode||'standard_anime',
      knowledge_pack:value.knowledge_pack||'',knowledge_operator:value.knowledge_operator||'',
      knowledge_appearance:value.knowledge_appearance||'',auto_generate:value.auto_generate!==false,support_mode:support};
    if(Object.prototype.hasOwnProperty.call(value,'picture_authoring_backend'))result.picture_authoring_backend=backend;
    for(const field of ['research_support','research_pixel_use'])
      if(Object.prototype.hasOwnProperty.call(value,field))result[field]=structuredClone(value[field]);
    const method=Object.prototype.hasOwnProperty.call(value,'picture_authoring_method')?value.picture_authoring_method:(support==='auto'?'prose':'whole');
    if(!Object.prototype.hasOwnProperty.call(pictureMethods,method))throw new Error('请选择有效的画面组织方式');
    result.picture_authoring_method=method;
    const jointEligible=method==='prose'&&support==='auto'&&backend==='text';
    const joint=Object.prototype.hasOwnProperty.call(value,'joint_picture_decision')?value.joint_picture_decision:jointEligible;
    if(typeof joint!=='boolean'||(joint&&!jointEligible))throw new Error('联合画面决策须为明确开关，且仅适用于自动完整文字画面描述');
    result.joint_picture_decision=joint;
    const researchMode=Object.prototype.hasOwnProperty.call(value,'research_mode')?value.research_mode:(joint?'catalog':'legacy_full');
    if(!['legacy_full','catalog'].includes(researchMode)||(researchMode==='catalog'&&!joint))throw new Error('按需研究读取需使用完整画面共同决定');
    result.research_mode=researchMode;
    const understand=Object.prototype.hasOwnProperty.call(value,'understand_request')?value.understand_request:!joint;
    if(typeof understand!=='boolean')throw new Error('独立原句理解须为明确开关');
    result.understand_request=understand;
    if(joint||Object.prototype.hasOwnProperty.call(value,'parameter_scope')){
      const scope=Object.prototype.hasOwnProperty.call(value,'parameter_scope')?value.parameter_scope:'initial';
      if(!['initial','all'].includes(scope))throw new Error('采样参数作用域须为当前起点或全程');
      result.parameter_scope=scope;
    }
    const profile=Object.prototype.hasOwnProperty.call(value,'authoring_profile')?value.authoring_profile:'local_4b';
    if(!Object.prototype.hasOwnProperty.call(authoringProfiles,profile))throw new Error('请选择现有的构思模型配置');
    result.authoring_profile=profile;
    if(Object.prototype.hasOwnProperty.call(value,'parameter_sources')){
      const sources=value.parameter_sources;
      if(!sources||Array.isArray(sources)||typeof sources!=='object'
        ||Object.entries(sources).some(([field,source])=>!samplingFields.includes(field)||!['model_default','user_override'].includes(source)))
        throw new Error('采样来源只允许六项采样参数的模型默认或用户明确值');
      result.parameter_sources=structuredClone(sources);
    }
    if(Object.prototype.hasOwnProperty.call(value,'result_guided')){
      if(typeof value.result_guided!=='boolean')throw new Error('结果回合选择应为明确的开或关');
      result.result_guided=value.result_guided;
    }
    for(const [field,max] of [['max_jobs',6],['max_calls',8]]){
      if(!Object.prototype.hasOwnProperty.call(value,field))continue;
      const budget=value[field];
      if(!Number.isInteger(budget)||budget<1||budget>max)throw new Error(`${field} 须为 1–${max} 的整数；没有自动截断预算`);
      result[field]=budget;
    }
    return result;
  }
  function requestReadingDetails(record){
    const reading=record?.request_reading;
    if(reading==null)return null;
    const index=reading.call_index,call=Number.isInteger(index)&&index>=0?record.calls?.[index]??null:null;
    return {
      '保存的理解步骤状态':reading.status??null,
      '完整原文':reading.original_text??null,'原文指纹':reading.original_sha256??null,
      '模型原始解释（未认证）':reading.raw??null,
      '来源目标 creation_intent（未认证，可纠正）':reading.creation_intent??null,
      '当次理解输入（原件）':reading.input??null,'输入绑定指纹':reading.binding_sha256??null,
      '模型调用序号':index??null,'对应调用原件（返回与引用不认证理解）':call,
      '显示关联核对（不认证语义或效果）':{
        '原文与本任务输入相同':reading.original_text===record.input_snapshot?.sentence,
        '调用已返回':call?.status==='returned',
        '原回应与保存解释相同':Boolean(reading.raw!=null&&call?.data!=null&&key(reading.raw)===key(call.data)),
      },
      '后续反馈记录（保留各自来源，不是原始解释）':record.feedback||[],
    };
  }
  function reviewState(record){
    const submitted=record?.generation_request,proposal=submitted?.author_proposal;
    const readingVisible=record?.request_reading!=null;
    const prose=pictureAuthoringMethod(record)==='prose'||record?.prose_picture_choice!=null||proposal?.prose_picture_choice!=null;
    if(prose){
      const raw=record?.prose_picture_choice,translated=record?.translated_prose_picture_choice??null,receipt=record?.prose_picture_receipt;
      const channel=target(submitted||record?.input_snapshot?.config),native=record?.native_prompts?.[channel];
      const boundOriginal=Boolean(typeof raw?.scene_narrative==='string'&&raw.scene_narrative.trim()
        &&receipt?.method==='prose'&&proposal?.scene_narrative_origin==='model_authored_complete_prose'
        &&key(proposal.prose_picture_choice)===key(raw)
        &&key(proposal.translated_prose_picture_choice??null)===key(translated)
        &&key(proposal.prose_picture_receipt)===key(receipt));
      const boundNative=Boolean(typeof native?.positive==='string'&&native.positive.trim()&&typeof native?.negative==='string'
        &&submitted?.prompt===native.positive&&submitted?.negative_prompt===native.negative
        &&(!submitted?.native_prompts||(submitted.native_prompts[channel]?.positive===native.positive
          &&submitted.native_prompts[channel]?.negative===native.negative)));
      const explicitNativeOverride=Boolean(boundOriginal&&boundNative&&record?.native_user_override
        &&proposal?.condition_status==='historical_after_native_override'
        &&submitted?.semantic_spec==null&&submitted?.scene_contract==null);
      const complete=Boolean(record?.plan&&boundOriginal&&boundNative&&proposal?.condition_status!=='historical_after_native_override');
      return {prose:true,relational:false,canonical:false,hasParts:false,canReview:readingVisible||raw!=null||Boolean(record?.plan),complete,
        nativeReadable:complete||explicitNativeOverride,explicitNativeOverride};
    }
    const parts=record?.relational_picture,relational=record?.input_snapshot?.picture_authoring_method==='relational_steps'||Boolean(parts);
    const hasParts=Boolean(parts&&(parts.core!=null||parts.finish!=null));
    const canonical=!relational&&[record?.canonical_picture_choice,record?.translated_canonical_picture_choice,
      record?.picture_projection,proposal?.canonical_picture_choice,proposal?.picture_projection].some(value=>value!=null);
    if(canonical){
      const raw=record?.canonical_picture_choice,translated=record?.translated_canonical_picture_choice??null,
        receipt=record?.picture_projection,channel=target(submitted||record?.input_snapshot?.config),native=record?.native_prompts?.[channel];
      // This binds the saved originals/receipt to the current submitted words;
      // it does not reconstruct the proposal, verify its meaning or certify art.
      const boundOriginal=Boolean(raw!=null&&receipt?.kind==='host_mechanical_projection_of_canonical_author_choice'
        &&['original_choice_sha256','effective_choice_sha256'].every(name=>/^[0-9a-f]{64}$/.test(receipt[name]||''))
        &&proposal?.canonical_picture_choice!=null&&key(proposal.canonical_picture_choice)===key(raw)
        &&key(proposal.translated_canonical_picture_choice??null)===key(translated)
        &&proposal.picture_projection&&key(proposal.picture_projection)===key(receipt));
      const boundNative=Boolean(typeof native?.positive==='string'&&Boolean(native.positive.trim())&&typeof native?.negative==='string'
        &&submitted?.prompt===native.positive&&submitted?.negative_prompt===native.negative
        &&(!submitted?.native_prompts||(submitted.native_prompts[channel]?.positive===native.positive
          &&submitted.native_prompts[channel]?.negative===native.negative)));
      const explicitNativeOverride=Boolean(boundOriginal&&boundNative&&record?.native_user_override
        &&proposal?.condition_status==='historical_after_native_override'
        &&submitted?.semantic_spec==null&&submitted?.scene_contract==null);
      const complete=Boolean(record?.plan&&boundOriginal&&boundNative&&proposal?.condition_status!=='historical_after_native_override');
      return {relational:false,canonical:true,hasParts:false,canReview:true,complete,
        nativeReadable:complete||explicitNativeOverride,explicitNativeOverride};
    }
    if(!relational)return {relational:false,canonical:false,hasParts:false,canReview:readingVisible||Boolean(record?.plan),complete:Boolean(record?.plan)};
    const saved=record?.generation_request?.author_proposal?.relational_picture,receipt=parts?.receipt;
    const hashes=['core_sha256','finish_sha256','merged_sha256'];
    const channel=target(submitted||record?.input_snapshot?.config),native=record?.native_prompts?.[channel];
    // A complete older plan is not the current partial proposal. This is a
    // display binding to saved compiler receipts, not frontend art approval or
    // a reimplementation of the backend's hash/semantic validation.
    const complete=Boolean(record?.plan&&parts?.status==='merged_unverified'&&parts.core!=null&&parts.finish!=null
      &&receipt?.method==='relational_steps'&&hashes.every(name=>/^[0-9a-f]{64}$/.test(receipt[name]||''))
      &&saved&&key(saved)===key({core:parts.core,finish:parts.finish,receipt})
      &&typeof native?.positive==='string'&&Boolean(native.positive.trim())&&typeof native?.negative==='string'
      &&submitted?.prompt===native.positive&&submitted?.negative_prompt===native.negative);
    return {relational:true,canonical:false,hasParts,canReview:readingVisible||hasParts||Boolean(record?.plan),complete};
  }
  function comparisonCompletionState(record){
    const capability=record?.comparison_completion,entries=record?.comparison_completions;
    const lastId=record?.last_comparison_completion_request_id,last=lastId&&entries?.[lastId];
    const entryPaused=record?.status==='paused'&&last?.status==='paused'
      &&last.scope==='saved_pair_comparison_only_no_author_or_generation'
      &&typeof last.turn_id==='string'&&Boolean(last.turn_id.trim());
    const paused=entryPaused||(record?.status==='paused'&&capability?.resume_existing===true);
    const turns=Array.isArray(record?.result_cycle?.turns)?record.result_cycle.turns:[];
    const turn=turns.find(value=>value?.id===capability?.turn_id);
    const valid=capability?.available===true&&typeof capability.turn_id==='string'&&Boolean(capability.turn_id.trim())
      &&capability.scope==='saved_pair_comparison_only_no_author_or_generation'
      &&capability.new_images===0&&Number.isInteger(capability.maximum_new_calls)
      &&((capability.mode==='new_call'&&capability.maximum_new_calls===1)
        ||(capability.mode==='reuse_return'&&capability.maximum_new_calls===0))
      &&(!Object.prototype.hasOwnProperty.call(capability,'resume_existing')||typeof capability.resume_existing==='boolean')
      &&turn?.phase==='comparing'&&turn.comparison==null
      &&typeof turn.source_artwork_id==='string'&&Boolean(turn.source_artwork_id)
      &&typeof turn.child_artwork_id==='string'&&Boolean(turn.child_artwork_id)
      &&turn.source_artwork_id!==turn.child_artwork_id;
    const resumeExisting=valid&&capability.resume_existing===true&&entryPaused&&last.turn_id===capability.turn_id;
    return {capability:valid?capability:null,paused,
      resumeExisting:Boolean(resumeExisting),
      newActionAvailable:Boolean(valid&&!paused&&capability.resume_existing!==true)};
  }
  function describe(record,receipt={}){
    const active=record?.active_job,working=['queued','running'].includes(record?.status);
    const jobInFlight=active&&!['completed','failed','cancelled'].includes(active.status);
    const activeStopUnknown=Boolean(active&&(active.status==='cancellation_uncertain'||active.stage==='cancellation_uncertain'
      ||(jobInFlight&&(active.cancel_requested||active.cancel_error))));
    const stopUnknown=Boolean(record?.cancel_error||record?.stage==='cancellation_uncertain'||activeStopUnknown);
    const waitingReceipt=Boolean(jobInFlight&&active.stage==='waiting_for_unconfirmed_engine_receipt');
    const pausing=working&&record?.pause_requested;
    const failedCall=record?.status==='failed'?[...(record?.calls||[])].reverse().find(call=>call?.status==='failed'):null;
    const rawError=record?.error||failedCall?.error;
    // This is the saved pre-launch resource code, not a diagnosis inferred
    // from an English error, an empty candidate list or another old call.
    const startResourcesFailed=record?.calls?.at(-1)===failedCall&&failedCall?.error_code==='insufficient_start_resources'
      &&(!record.error||record.error===failedCall.error);
    const composingNotLoaded=startResourcesFailed&&failedCall.purpose==='composing';
    const choiceReturned=composingNotLoaded&&record.calls.some(call=>call?.purpose==='choosing_start'&&call.status==='returned');
    const noImageDispatched=startResourcesFailed&&record.plan==null&&record.generation_request==null
      &&!record.active_job_id&&!active&&!receipt.pending_action&&!record.local_edit
      &&Array.isArray(record.candidates)&&record.candidates.length===0&&!(record.generation_dispatches?.length)
      &&!record.continuation;
    const resourceDetail=startResourcesFailed
      ?`${choiceReturned?'选路已返回；':''}${composingNotLoaded?'本次完整构思的文字模型未加载。':failedCall.purpose==='choosing_start'?'本次选路的文字模型未加载。':'当前步骤的文字模型未加载。'}可用运行资源未达到本次模型的启动条件，任务已停止。${noImageDispatched?'本任务没有派图。':''}原记录与已有作品保留，没有自动重试。`:'';
    const resourceCallError=startResourcesFailed&&failedCall.error&&failedCall.error!==rawError?failedCall.error:'';
    const incompleteLegacy=record?.status==='failed'&&failedCall?.error_code==='incomplete_output'
      &&rawError==='Authoring did not produce one complete stopped response';
    const error=incompleteLegacy?'构思输出未完整结束，原响应保留；没有自动生成或重试':rawError;
    // Translate only this exact historical generic error. A more specific
    // recorded call error remains visible, without inventing its cause.
    const specificCallError=incompleteLegacy&&failedCall?.error&&failedCall.error!==rawError?failedCall.error:'';
    const phase=jobInFlight?({queued:'等待图像资源',starting_engine:'加载图像模型',queued_in_engine:'等待采样',sampling:'生成画面',saving:'保存候选',check_detect:'读取检查结果'})[active.stage]
      :({understanding_request:'理解原句与输出要求',interpreting_source:'理解原句与输出要求',choosing_start:'选择创作起点',planning_directions:'规划独立表现方向',composing:'构思画面与视觉支撑',native_language:'准备原生语言条件',relational_core:'形成身体、注意对象与空间关系',relational_finish:'复核关系并发展环境、光色与画法',generating:'生成候选',locating_region:'定位修改区域',local_edit:'局部修改与圈外校验',observing:'观察实际候选',choosing:'选择下一操作',comparing:'比较实际父子版本',comparing_directions:'比较独立表现方向'})[record?.stage];
    const directionPhaseNote=record?.stage==='planning_directions'?'方向规划是作者提案，不是用户原话或已准备好的全部原生条件；不表示图片已生成或要求已满足。'
      :record?.stage==='comparing_directions'?'这是独立方向之间的模型比较，尚未认证，不是用户评价或要求已满足；父子修改比较另行保留。':'';
    const review=reviewState(record);
    const failed=record?.status==='failed',terminal=['ready','paused','failed'].includes(record?.status);
    const hasCandidate=Boolean(record?.candidates?.some(c=>c.artwork_id&&c.status==='completed'&&imageHref(c.image_url)));
    const dispatchPending=Boolean(record?.generation_dispatches?.some(item=>!item.completed));
    const edit=record?.local_edit;
    const unknownReceipt=Boolean(receipt.pending_action||stopUnknown
      ||record?.generation_dispatches?.some(item=>!item.completed&&!item.job_id)
      ||edit?.submission_uncertain||(edit?.submission_started&&!edit?.job_id));
    const uncertain=unknownReceipt||dispatchPending||Boolean(edit?.job_id&&!edit.completed);
    const comparisonCompletion=comparisonCompletionState(record);
    const nativePreparation=record?.saved_prose_preparation;
    const nativePreparationBudgetKnown=nativePreparation&&Number.isInteger(nativePreparation.minimum_max_calls)
      &&Number.isInteger(nativePreparation.suggested_max_calls)&&Number.isInteger(nativePreparation.maximum_max_calls)
      &&nativePreparation.minimum_max_calls>=1&&nativePreparation.minimum_max_calls<=nativePreparation.suggested_max_calls
      &&nativePreparation.suggested_max_calls<=nativePreparation.maximum_max_calls&&nativePreparation.maximum_max_calls<=8;
    const prepareNativeUnavailableReason=comparisonCompletion.paused?'当前暂停的是既有两图比较，使用继续本回合，不重新准备或生图'
      :!['failed','paused'].includes(record?.status)?'当前状态不允许重新准备保存的原稿'
      :jobInFlight||record?.active_job_id||receipt.busy?'当前操作或图像任务仍在途，先读取真实回执'
      :uncertain?'上次操作或派发回执尚未确认，先重新读取，不重复准备'
      :!review.canReview||!(review.prose||(nativePreparation?.choice_kind==='canonical'&&review.canonical))
        ?'没有可查看的完整画面原稿，或后端未明确声明此原稿类型可重新准备'
      :nativePreparation?.available!==true||typeof nativePreparation.language_new_call_required!=='boolean'
        ?nativePreparation?.reason||'后端未声明当前原稿可重新准备条件；请载入更新后重新读取，不猜测资格'
      :nativePreparation.language_new_call_required&&!nativePreparationBudgetKnown?'后端未提供有效的有界语言调用额度，请重新读取':'';
    const canPrepareNative=!prepareNativeUnavailableReason;
    const remaining=record?.remaining_budget,next=record?.next_operation_budget;
    const resumeExhausted=record?.status==='paused'&&Number.isFinite(remaining?.active_seconds)&&remaining.active_seconds<=0;
    // Only the new API's explicit user-operation capability can offer a fresh
    // bounded round. Missing/old responses retain their saved-budget meaning.
    const userOperationBudget=receipt.initiator!=='assistant_validation'&&next?.allows_user_new_round===true
      &&Number.isInteger(next.jobs)&&next.jobs>=1&&next.jobs<=6
      &&Number.isInteger(next.calls)&&next.calls>=1&&next.calls<=8
      &&Number.isFinite(next.active_seconds)&&next.active_seconds>=60&&next.active_seconds<=3600
      ?{jobs:next.jobs,calls:next.calls,active_seconds:next.active_seconds}:null;
    const operationBudget=userOperationBudget||remaining;
    const resumeUnavailableReason=comparisonCompletion.paused
      ?comparisonCompletion.resumeExisting?'':record?.comparison_completion?.reason||'既有比较的继续资格或回执尚未确认；先重新读取，不扩额或重派'
      :canPrepareNative?'此暂停原稿须明确重新准备条件，不恢复旧步骤或自动生图'
      :resumeExhausted?'本回合执行额度已用完，不能继续旧回合；原图与材料保留':'';
    const pausedBudgetNote=resumeExhausted?(userOperationBudget&&hasCandidate
      ?'本回合执行额度已用完，不再继续旧回合。已保存图仍可在相应入口明确开始新的接续、圈改或重新构思；新操作需要你的明确选择，不自动增加额度。'
      :'本回合执行额度已用完，原图与材料保留；当前回合不能继续。'):'';
    const budgetKnown=operationBudget&&['jobs','calls','active_seconds'].every(name=>Number.isFinite(operationBudget[name]));
    const commonBlock=!terminal?'当前任务状态不能开始新的后继操作':jobInFlight||record?.active_job_id||receipt.busy?'当前操作或图像任务仍在途，先读取真实回执'
      :uncertain?'上次派发、操作或取消回执尚未确认，先重新读取；不重派或绕过'
      :failed&&!budgetKnown?'失败任务的剩余预算尚不可核对，先重新读取'
      :operationBudget&&!(operationBudget.jobs>0)?'剩余图片预算不足，不安排新候选'
      :operationBudget&&!(operationBudget.active_seconds>0)?'剩余执行时间不足，不安排新操作':'';
    const sourceBlock=commonBlock||(failed&&!hasCandidate?'失败任务尚无已完成的真实候选，不能借旧方案继续':'');
    const callsBlock=(minimum,block=commonBlock)=>block||(operationBudget&&!(operationBudget.calls>=minimum)?`剩余模型调用不足；本操作至少需要 ${minimum} 次新调用`:'');
    const continueUnavailableReason=callsBlock(2,sourceBlock),regenerateUnavailableReason=callsBlock(record?.input_snapshot?.picture_authoring_method==='relational_steps'?2:1);
    const canContinue=!continueUnavailableReason,canLocalEdit=!sourceBlock,canRegenerate=!regenerateUnavailableReason;
    const canCompleteComparison=comparisonCompletion.newActionAvailable&&terminal&&!jobInFlight
      &&!record?.active_job_id&&!uncertain&&!receipt.busy;
    const newOperationNote=failed&&(canContinue||canLocalEdit||canRegenerate)
      ?`${hasCandidate?'已有完成候选可用于明确的新接续、圈选修改或重新构思':'可按完整原句明确重新构思，不需要旧候选'}；这是新操作，不是恢复或复跑失败步骤。原失败与响应保留。`:'';
    return {title:stopUnknown?'停止尚未确认':waitingReceipt?'等待未确认任务回执':pausing?'正在请求暂停后续派发':record?.status==='cancelled'&&jobInFlight?'已请求停止，当前图像任务待结束':working&&phase?phase:startResourcesFailed?'运行资源不足，模型未加载':({queued:'等待创作资源',running:'正在创作',needs_confirmation:'方案待审阅',ready:'候选可查看',paused:jobInFlight?'后续已暂停，当前采样仍在进行':'已暂停后续派发',cancelled:'已停止',failed:'创作停止在错误处'})[record?.status]||'读取创作任务',
      detail:[stopUnknown?'停止请求已记录；是否完成停止仍待真实回执核对，不会据此重派。':waitingReceipt?'本图像任务尚未提交；正在等待其它任务的确切引擎回执。':record?.detail,
        resourceDetail,error,resourceCallError,specificCallError,newOperationNote,pausedBudgetNote,directionPhaseNote,
        record?.cancel_error&&record.cancel_error!==error?record.cancel_error:'',
        active?.cancel_error&&active.cancel_error!==error&&active.cancel_error!==record?.cancel_error?active.cancel_error:'',
        stopUnknown&&active?.error!==active?.cancel_error?active?.error:'',
        ...((stopUnknown||waitingReceipt)&&Array.isArray(active?.notes)?active.notes:[]),
        !stopUnknown&&!waitingReceipt&&pausing&&jobInFlight?'已提交的图像任务可能继续采样，等待真实暂停状态；不会据此重派。':'',active?`实际图像任务 ${active.id} · ${active.stage}`:''].filter(Boolean).join('\n'),
      percent:!stopUnknown&&!waitingReceipt&&jobInFlight&&active&&Number.isFinite(active.progress)?Math.max(0,Math.min(100,active.progress<=1?active.progress*100:active.progress)):null,
      working:working||Boolean(jobInFlight)||unknownReceipt,canGenerate:record?.status==='needs_confirmation'&&!jobInFlight&&!unknownReceipt&&(!(review.prose||review.relational||review.canonical)||(review.nativeReadable??review.complete)),
      canPause:!pausing&&(working||(record?.status==='needs_confirmation'&&!jobInFlight&&!record.active_job_id&&!unknownReceipt&&!uncertain&&!receipt.busy)),
      canResume:record?.status==='paused'&&!jobInFlight&&!unknownReceipt&&!canPrepareNative&&!resumeExhausted
        &&(!comparisonCompletion.paused||comparisonCompletion.resumeExisting),resumeUnavailableReason,
      canStop:Boolean(jobInFlight)||unknownReceipt||!['failed','cancelled'].includes(record?.status),
      canContinue,canLocalEdit,canRegenerate,userOperationBudget,continueUnavailableReason,localEditUnavailableReason:sourceBlock,regenerateUnavailableReason,
      canPrepareNative,nativePreparation,prepareNativeUnavailableReason,
      canCompleteComparison,comparisonCompletion,
      canChoose:!working&&!jobInFlight&&!record?.active_job_id&&!unknownReceipt&&!dispatchPending&&!receipt.busy
        &&Boolean(record?.candidates?.some(c=>c.artwork_id&&c.status==='completed'&&imageHref(c.image_url)))};
  }
  function statusLabel(value){return ({queued:'等待执行',running:'正在执行',completed:'图片已保存',failed:'未完成',
    cancelled:'已停止',paused:'已暂停',ready:'候选可查看',needs_confirmation:'待审阅',
    cancellation_uncertain:'停止尚未确认'})[value]||value||'待核验';}
  function operationLabel(value){return ({generate:'生成',iteration_generate:'生成',creative_generate:'生成',
    handoff:'模型交接',image_handoff:'模型交接',local_edit:'圈选修改',region_repair:'圈选修改',text_layer:'精确文字',
    text_render:'精确文字',repair:'局部修改'})[value]||value||'作品';}
  function selectionOrigin(record){return record?.selection_origin==='assistant_validation'?'assistant_validation':'user';}
  function selectedVersionLabel(record){return selectionOrigin(record)==='assistant_validation'
    ?'助手验证所用版本（不代表用户认可）':'当前选用版本';}
  function artworkState(record,id,receipt={}){
    const candidate=record?.candidates?.find(c=>c.artwork_id===id),flags=describe(record,receipt);
    const belongs=Boolean(record?.id&&candidate),completed=Boolean(belongs&&candidate.status==='completed'&&imageHref(candidate.image_url));
    const selected=belongs&&record.selected_artwork_id===id;
    return {belongs,completed,selected,selectedByAssistant:selected&&selectionOrigin(record)==='assistant_validation',
      selectedLabel:selectedVersionLabel(record),flags,
      canContinue:completed&&flags.canContinue,canLocalEdit:completed&&flags.canLocalEdit,
      canChoose:completed&&flags.canChoose};
  }
  function historyHtml(records){
    return records.length?records.map(record=>{
      const count=(record.candidates||[]).filter(c=>c.status==='completed'&&imageHref(c.image_url)).length;
      return `<article class="creative-history-row"><div><strong>${esc(record.id)}</strong><span>${esc(statusLabel(record.status))} · ${count} 张已保存图片 · ${esc(pictureMethodLabel(pictureAuthoringMethod(record)))}${record.input_snapshot?.picture_authoring_method?'':'（旧记录未注明方法）'}</span><p>${esc(record.input_snapshot?.sentence||'原始构想未记录，不会补造')}</p><small>${esc(record.created_at||'')}</small></div><button type="button" data-open-creative-history="${esc(record.id)}">查看此创作</button></article>`;
    }).join(''):'<p class="empty-copy">还没有创作记录。已有研究和旧协作记录仍在各自入口。</p>';
  }
  function imageHref(value){
    if(typeof value!=='string'||!value)return '';
    if(value.startsWith('/')&&!value.startsWith('//'))return value;
    try{const url=new URL(value);return ['http:','https:'].includes(url.protocol)?value:'';}catch{return '';}
  }
  function receiptRecovery(record){
    const id=record?.active_job_id,active=record?.active_job;
    const visible=Boolean(record?.cancel_error||record?.stage==='cancellation_uncertain'
      ||(id&&!active)||active?.status==='cancellation_uncertain'||active?.stage==='cancellation_uncertain'
      ||(!['completed','failed','cancelled'].includes(active?.status)&&(active?.cancel_requested||active?.cancel_error)));
    const valid=typeof id==='string'&&/^[a-zA-Z0-9_-]{1,100}$/.test(id);
    const consistent=!active||active.id===id;
    return {visible,jobId:valid?id:null,allowed:visible&&valid&&consistent,
      reason:!valid?'缺少确切原图像任务编号；不能猜测其它任务。':!consistent?'原图像任务编号与当前投影不一致；未猜测归属。':''};
  }
  function directionSlots(record){
    const plan=record?.direction_plan,ids=new Set(),ordinals=new Set();
    if(plan?.requested_format!=='separate_candidates'||!Number.isInteger(plan.requested_count)||plan.requested_count<1)return [];
    return (Array.isArray(plan.slots)?plan.slots:[]).filter(slot=>{
      if(!slot||typeof slot.id!=='string'||!slot.id||!Number.isInteger(slot.ordinal)||slot.ordinal<1
        ||slot.ordinal>plan.requested_count||ids.has(slot.id)||ordinals.has(slot.ordinal))return false;
      ids.add(slot.id);ordinals.add(slot.ordinal);return true;
    });
  }
  function directionCandidate(record,slot){
    return record.candidates?.find(candidate=>candidate.direction_id===slot.id&&candidate.artwork_id===slot.artwork_id
      &&(!slot.job_id||candidate.job_id===slot.job_id)&&candidate.status==='completed'&&imageHref(candidate.image_url));
  }
  function reviewedDirection(record){
    const assignment=record?.generation_request?.author_proposal?.direction_assignment,plan=record?.direction_plan;
    if(!assignment||typeof plan?.plan_sha256!=='string'||!plan.plan_sha256||assignment.plan_sha256!==plan.plan_sha256)return null;
    return directionSlots(record).find(slot=>slot.id===assignment.id&&slot.ordinal===assignment.ordinal
      &&(!plan.active_slot_id||plan.active_slot_id===slot.id)
      &&slot.proposal!=null&&key(slot.proposal)===key(assignment.proposal))||null;
  }
  function directionPlanHtml(record){
    const plan=record?.direction_plan;if(!plan||typeof plan!=='object')return '';
    const slots=directionSlots(record),count=plan.requested_count;
    const known=plan.requested_format==='separate_candidates'&&Number.isInteger(count)&&count>0;
    const saved=slots.filter(slot=>['completed','candidate_saved'].includes(slot.status)&&directionCandidate(record,slot)).length;
    const unplanned=known?Math.max(0,count-slots.filter(slot=>slot.status!=='unplanned'&&slot.proposal!=null).length):null;
    const link=id=>`<button type="button" data-creative-art="${esc(id)}">${esc(id)}</button>`;
    const state=slot=>['completed','candidate_saved'].includes(slot.status)&&!directionCandidate(record,slot)?'保存记录的作品关联待核对'
      :({planned:'提案待准备',unplanned:'未规划',composing:'正在准备此方向',prepared:'此方向条件已准备，待逐项审阅',
        preparing:'正在准备此方向',needs_confirmation:'此方向待审阅',submitted:'已派图，尚待实际候选',completed:'候选已保存',
        candidate_saved:'候选已保存',cancelled:'已停止'})[slot.status]||String(slot.status??'未记录');
    const summary=known?`请求 ${count} 个方向 · 已有 ${saved} 个方向候选 · ${Math.max(0,count-saved)} 项未生成`
      :'方向记录 · 请求形式待核对';
    const review=reviewState(record),current=(review.nativeReadable??review.complete)?reviewedDirection(record):null;
    const active=slots.find(slot=>slot.id===plan.active_slot_id);
    const envelope=plan.comparison,independent=envelope?.scope==='independent_directions'
      ||(!Object.prototype.hasOwnProperty.call(envelope||{},'scope')&&envelope?.comparison_scope==='independent_directions');
    const comparison=envelope?.scope==='independent_directions'?envelope.raw:envelope;
    const pair=envelope?.scope==='independent_directions'?envelope.pair:plan.pair;
    const boundPair=independent&&comparison!=null&&Array.isArray(pair?.artwork_ids)&&pair.artwork_ids.length===2
      &&Array.isArray(pair.direction_ids)&&pair.direction_ids.length===2&&new Set(pair.artwork_ids).size===2&&new Set(pair.direction_ids).size===2
      &&pair.direction_ids.every((id,index)=>slots.some(slot=>slot.id===id)&&record.candidates?.some(candidate=>candidate.direction_id===id
        &&candidate.artwork_id===pair.artwork_ids[index]&&candidate.status==='completed'&&imageHref(candidate.image_url)));
    const rows=(name,values)=>`<p>${name}：</p>${Array.isArray(values)&&values.length?`<ul>${values.map(value=>`<li>${esc(value)}</li>`).join('')}</ul>`:'<p>未记录条目，不据此判断已解决或通过。</p>'}`;
    return `<summary>${esc(summary)}</summary><p>系统提案不是你的原话，提案和原文理解都可以纠正；保存候选或预算止步不等于符合要求，修改版本不算新方向。规划不表示全部生图条件已准备，双面板作品也不用于推算方向数。</p>
      ${known?`<p>未生成项中，${unplanned} 项尚未规划。</p>`:''}
      <p>请求出处：${esc(plan.source_quote??'未记录')}</p>
      ${typeof plan.raw?.collection_rationale==='string'?`<details data-direction-detail="collection-rationale"><summary>集合作者的差异理由</summary><p>以下解释属于整个集合，不是某一幅的画面条件：${esc(plan.raw.collection_rationale)}</p></details>`:''}
      ${plan.reason?`<p>方向集合止步说明（保存记录）：${esc(plan.reason)}</p>`:''}
      ${plan.status==='not_started_budget'?'<p>本回合预算不足，独立方向集合未开始；未分配的单幅候选（若有）不计作独立方向完成。</p>':''}
      ${current?`<p>当前${record.status==='needs_confirmation'?'待审':'保存条件对应的'}方向：${esc(current.ordinal)}。只确认或生成当前项，不一次批准全部方向；其它项是否已准备或已保存，以各方向记录为准。</p>`
        :active?`<p>当前分配方向：${esc(active.ordinal)} · 保存的准备状态：${esc(plan.status??'未记录')}。尚未关联此项可审的完整原生条件，不借用上一项条件。</p>`:''}
      ${slots.map(slot=>`<details data-direction-detail="${esc(slot.id)}"><summary>方向 ${esc(slot.ordinal)} · ${esc(slot.proposal?.title??'尚无作者提案')} · ${esc(state(slot))}</summary>
        <p>作者提案：${esc(slot.proposal?.approach??'此项尚未规划，不虚构内容。')}</p>${typeof slot.proposal?.contrast==='string'?`<p>旧提案对照：${esc(slot.proposal.contrast)}</p>`:''}
        ${directionCandidate(record,slot)?`<p>关联实际候选：${link(slot.artwork_id)}</p>`:''}</details>`).join('')}
      ${independent?`<details data-direction-detail="comparison"><summary>独立方向比较原回应</summary>
        ${boundPair?`<p>两张明确分配的独立方向候选，不是父子修改比较：${pair.artwork_ids.map((id,index)=>`方向 ${esc(slots.find(slot=>slot.id===pair.direction_ids[index]).ordinal)} ${link(id)}`).join(' · ')}。仅此两图参与，不表示全部方向已比较。</p>
          <p>原回应中的 parent / child 在这里仅为第一／第二输入标签，不代表版本谱系或用户选图。</p>
          ${rows('模型报告的收益',comparison.gains)}${rows('模型报告的损失',comparison.losses)}${rows('未解决或未知',comparison.unresolved)}
          <p>模型解释：${esc(comparison.reason??'未记录')}。不更改用户选择。</p>`:'<p>实际两图与方向绑定尚未核对，仅显示保存原件，不猜测比较对象或认证方向差异。</p>'}
        <pre>${esc(JSON.stringify(envelope,null,2))}</pre></details>`:''}
      <details data-direction-detail="raw"><summary>作者原回应与方向记录（JSON）</summary><pre>${esc(JSON.stringify(plan,null,2))}</pre></details>`;
  }
  function rejectedChoiceHtml(context){
    const rejection=context?.choice_rejection||context?.reference_choice_rejection;
    return rejection?.adopted===false&&[
      'operation_unavailable_for_reference','target_not_compatible_with_current_conditions'].includes(rejection.code)
      ?`<p>模型建议未采用：当前方法或目标不可执行。没有据此选图、改派任务或自动重试。</p><details data-result-detail="reference-rejection"><summary>查看未采用的原建议及原因</summary><pre>${esc(JSON.stringify(rejection,null,2))}</pre></details>`:'';
  }
  function resultCycleHtml(record){
    const standaloneRejection=rejectedChoiceHtml(record?.continuation);
    const cycle=record?.result_cycle;if(!cycle||typeof cycle!=='object')return standaloneRejection?`<summary>本轮建议未采用 · 原图保留</summary>${standaloneRejection}`:'';
    const turns=Array.isArray(cycle.turns)?cycle.turns:[];
    if(standaloneRejection&&record.input_snapshot?.result_guided!==true&&!turns.length)
      return `<summary>本轮建议未采用 · 原图保留</summary>${standaloneRejection}`;
    const link=id=>id?`<button type="button" data-creative-art="${esc(id)}">${esc(id)}</button>`:'尚未指定';
    const status=({idle:'等待实际结果',running:'正在审读与推进',paused:'内部回合暂停标记',stopped:'本轮已止步'})[cycle.status]||String(cycle.status??'未记录');
    const phase=value=>({observing_source:'观察源版本',advancing:'形成有目的的后继',comparing:'比较实际父子图',reviewed:'比较回应已保存（未认证）',no_image:'本回合没有新图'})[value]||String(value??'未记录');
    const rows=(name,values)=>`<p>${name}：</p>${Array.isArray(values)&&values.length
      ?`<ul>${values.map(value=>`<li>${esc(value)}</li>`).join('')}</ul>`:'<p>未记录条目，不据此判断已解决或通过。</p>'}`;
    const enabled=record.input_snapshot?.result_guided===true;
    return `<summary>按实际结果继续 · ${esc(status)} · 已保存 ${turns.length} 个回合</summary>
      <p>${enabled?'本任务启用结果回合，系统可以在预算内继续；读取页面不重新派发。':'本记录未声明启用自动结果回合；保存的过程不因读取而自动启动。'}暂停与停止仍以任务及实际采样状态为准。</p>
      <p>系统当前审读版本（临时参照，不是用户选图或质量认证）：${link(cycle.cursor_artwork_id)}</p>
      <p>${selectionOrigin(record)==='assistant_validation'?'助手验证所用版本（不代表用户认可）':'用户当前选择'}：${link(record.selected_artwork_id)}</p>
      ${cycle.active_turn_id?`<p>当前回合：${esc(cycle.active_turn_id)}</p>`:''}
      ${cycle.stop_reason?`<p>本轮止步原因（不代表作品合格）：${esc(cycle.stop_reason)}</p>`:''}
      ${standaloneRejection&&!turns.some(turn=>turn.continuation===record.continuation
        ||key(turn.continuation)===key(record.continuation))?standaloneRejection:''}
      ${turns.map(turn=>{
        const comparison=turn.comparison,base=comparison?.provisional_base;
        const material=turn.comparison_material,bindings=material?.image_bindings;
        const pairBound=material?.kind==='protected_local_crop_pair'
          &&Array.isArray(bindings)&&bindings.length===2&&turn.source_artwork_id!==turn.child_artwork_id
          &&bindings.every((binding,index)=>binding?.kind==='exact_mask_bbox_crop'
            &&binding.role===(index?'child':'parent')&&binding.artwork_id===(index?turn.child_artwork_id:turn.source_artwork_id)
            &&typeof binding.artwork_id==='string'&&Boolean(binding.artwork_id)
            &&/^[a-f0-9]{64}$/.test(binding.full_image_sha256||'')&&/^[a-f0-9]{64}$/.test(binding.sha256||''));
        const version=material?.contract_version,box=material?.crop_box,boundary=material?.pixel_boundary_verification,
          details=boundary?.details,sha=value=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value),
          positive=value=>Number.isInteger(value)&&value>0,
          validBox=value=>Array.isArray(value)&&value.length===4&&value.every(Number.isInteger)
            &&value[0]>=0&&value[1]>=0&&value[2]>value[0]&&value[3]>value[1];
        // These are display bindings to saved mechanical receipts, not a new
        // verification of image bytes or of the model's visual interpretation.
        const neutral=version==='local_visible_delta_v1'&&pairBound&&validBox(box)
          &&material.coordinates==='original_pixel_xyxy_right_bottom_exclusive'
          &&material.full_originals_retained_in_host===true&&material.whole_image_quality_verified===false
          &&sha(material.mask_sha256)&&bindings.every((binding,index)=>binding.index===index+1
            &&key(binding.crop_box)===key(box)&&binding.mask_sha256===material.mask_sha256
            &&sha(binding.continuation_snapshot_sha256)
            &&['width','height','bytes','full_image_width','full_image_height','full_image_bytes'].every(name=>positive(binding[name]))
            &&binding.width===box[2]-box[0]&&binding.height===box[3]-box[1]
            &&box[2]<=binding.full_image_width&&box[3]<=binding.full_image_height)
          &&['pass','warning'].includes(boundary?.status)&&details?.verification==='outside_mask_pixels_verified'
          &&details.source_sha256===bindings[0].full_image_sha256&&details.mask_sha256===material.mask_sha256
          &&(details.changed_pixel_bbox===null||(validBox(details.changed_pixel_bbox)
            &&details.changed_pixel_bbox[0]>=box[0]&&details.changed_pixel_bbox[1]>=box[1]
            &&details.changed_pixel_bbox[2]<=box[2]&&details.changed_pixel_bbox[3]<=box[3]));
        const fullBindingKeys=['index','role','artwork_id','sha256','width','height','bytes','continuation_snapshot_sha256'];
        const neutralFull=version==='full_visible_delta_v1'&&material?.kind==='full_original_pair'
          &&material.full_originals_retained_in_host===true&&material.whole_image_quality_verified===false
          &&Array.isArray(bindings)&&bindings.length===2&&turn.source_artwork_id!==turn.child_artwork_id
          &&!['crop_box','mask_sha256','coordinates','pixel_boundary_verification'].some(name=>name in material)
          &&bindings.every((binding,index)=>binding&&typeof binding==='object'
            &&Object.keys(binding).length===fullBindingKeys.length&&fullBindingKeys.every(name=>name in binding)
            &&binding.index===index+1&&binding.role===(index?'child':'parent')
            &&binding.artwork_id===(index?turn.child_artwork_id:turn.source_artwork_id)
            &&typeof binding.artwork_id==='string'&&Boolean(binding.artwork_id.trim())
            &&sha(binding.sha256)&&sha(binding.continuation_snapshot_sha256)
            &&['width','height','bytes'].every(name=>positive(binding[name])));
        const neutralVisible=neutral||neutralFull;
        const legacy=version==null&&material?.kind!=='full_original_pair'&&(material==null
          ||material.scope==='exact_mask_bbox_only_not_whole_image_observation_or_quality'),local=neutral||(legacy&&pairBound
          &&material.scope==='exact_mask_bbox_only_not_whole_image_observation_or_quality');
        const observationLabel=neutralFull?'后继全幅可见特征原回应（未认证）'
          :neutral?'后继局部可见特征原回应（未认证，非整图）'
          :local?'后继局部细节观察原回应（未认证，非整图）'
          :material!=null?'后继观察原回应（范围待核对，未认证）':'后继像素观察原回应（未认证）';
        const baseLabel=({parent:'源版本（父图）',child:'后继版本（子图）',neither:'暂不指定父图或子图'})[base]||String(base??'未记录');
        const reference=turn.reference_decision,choice=reference?.choice,referenceBase=reference?.reference_base;
        const rejectionHtml=rejectedChoiceHtml(turn.continuation);
        const completionReceipt=turn.comparison_completion_receipt,
          originalComparisonIndex=completionReceipt?.original_comparison_call_index,
          effectiveComparisonIndex=completionReceipt?.effective_comparison_call_index,
          effectiveComparisonCall=Number.isInteger(effectiveComparisonIndex)&&effectiveComparisonIndex>=0
            ?record.calls?.[effectiveComparisonIndex]:null,
          originalComparisonCall=Number.isInteger(originalComparisonIndex)&&originalComparisonIndex>=0
            ?record.calls?.[originalComparisonIndex]:null;
        const completedComparisonBound=neutralFull&&turn.phase==='reviewed'&&comparison!=null
          &&completionReceipt?.scope==='saved_pair_comparison_only'
          &&typeof completionReceipt.request_id==='string'&&Boolean(completionReceipt.request_id.trim())
          &&(originalComparisonIndex===null||(Number.isInteger(originalComparisonIndex)&&originalComparisonIndex>=0&&originalComparisonCall))
          &&effectiveComparisonCall?.status==='returned'&&key(effectiveComparisonCall.data)===key(comparison);
        const completionReceiptHtml=completionReceipt==null?'':`${completedComparisonBound
          ?`<p>本次补完只比较既有两张原图；没有作者、目的选源或新图片派发，不代表作品接受或用户选图。</p>
            <p>原比较调用：${Number.isInteger(originalComparisonIndex)?`第 ${originalComparisonIndex+1} 次`:'没有已开始的原调用'}；生效比较调用：第 ${effectiveComparisonIndex+1} 次。原调用记录仍保留${originalComparisonCall?.status==='failed'?'，包括原失败日志':''}。</p>
            ${originalComparisonCall?.error?`<p>原调用错误（历史记录）：${esc(originalComparisonCall.error)}</p>`:''}`
          :'<p>补比较回执与返回关联待核对；仅显示保存记录，不推定调用成功或作品接受。</p>'}
          <details data-result-detail="comparison-completion"><summary>补比较回执及原调用关联（不是作品接受）</summary><pre>${esc(JSON.stringify(completionReceipt,null,2))}</pre></details>`;
        const purposeBound=neutralVisible&&turn.phase==='reviewed'
          &&reference?.contract_version===(neutralFull?'full_reference_choice_v1':'local_reference_choice_v1')
          &&['parent','child','neither'].includes(referenceBase)&&choice?.reference_base===referenceBase
          &&['keep','regenerate','handoff','sketch','local_edit','stop'].includes(choice.operation)
          &&['newbie','illustrious','anima','z_image_turbo'].includes(choice.target)
          &&typeof choice.objective==='string'&&Boolean(choice.objective.trim())
          &&typeof choice.reason==='string'&&Boolean(choice.reason.trim())
          &&(referenceBase==='neither'?(reference.selected_artwork_id===null&&reference.selected_source_snapshot_sha256===null
            &&['regenerate','stop'].includes(choice.operation))
            :(reference.selected_artwork_id===(referenceBase==='parent'?turn.source_artwork_id:turn.child_artwork_id)
              &&sha(reference.selected_source_snapshot_sha256)
              &&(!neutralFull||reference.selected_source_snapshot_sha256
                ===bindings[referenceBase==='parent'?0:1].continuation_snapshot_sha256)));
        const purposeHtml=neutralVisible?`${purposeBound?`<p>后续用途选择（系统提案，非用户选图）：${esc(({parent:'源版本（父图）',child:'后继版本（子图）',neither:'父子图都不作为图像参照'})[referenceBase])} · ${referenceBase==='neither'?'不沿用父子图像':link(reference.selected_artwork_id)}</p>
            <p>下一操作：${esc(({keep:'保留',regenerate:'重新生成',handoff:'模型交接',sketch:'视觉支撑',local_edit:'局部修改',stop:'停止'})[choice.operation])} · 模型：${esc(choice.target)}</p>
            <p>本次目的：${esc(choice.objective)}</p><p>用途选择理由（未认证）：${esc(choice.reason)}。不更改用户选择。</p>`
            :reference!=null?'<p>用途选择合同或作品绑定待核对；仅显示保存原件，不推测下一参照。</p>'
              :`<p>尚无保存的用途选择；不从${neutralFull?'全幅':'局部'}变化原回应推测下一参照。</p>`}
            ${reference!=null?`<details data-result-detail="reference-decision"><summary>用途选择完整原回应与绑定（非用户选图）</summary><pre>${esc(JSON.stringify(reference,null,2))}</pre></details>`:''}`:'';
        return `<details data-result-turn="${esc(turn.id??'')}"><summary>回合 ${esc(turn.id??'未记录')} · ${esc(phase(turn.phase))}${local?' · 圈选细节':neutralFull?' · 全幅变化':''}</summary>
          <p>源版本：${link(turn.source_artwork_id)} · 后继版本：${link(turn.child_artwork_id)}</p>
          ${rejectionHtml}
          ${completionReceiptHtml}
          ${material!=null?`${local?'<p>本回合仅比较圈选范围的局部细节：按原遮罩外接框取父子图同位置裁剪，不是整图观察或质量认证；圈外像素校验是执行事实，不证明圈内效果。</p>'
            :neutralFull?'<p>本回合比较两张完整原 PNG 的可见变化，不是圈选裁片；原件绑定是取材事实，不证明凝视、情绪、身份或质量。</p>'
            :'<p>比较取材范围或父子图绑定待核对；以下原回应不当作整图或局部认证，不猜测裁剪对象。</p>'}
            <details data-result-detail="comparison-material"><summary>${local?'局部比较取材与来源记录':neutralFull?'全幅比较取材与来源记录':'比较取材与来源记录（范围待核对）'}</summary><pre>${esc(JSON.stringify(material,null,2))}</pre></details>`:''}
          ${comparison?`${neutralVisible?`<p>以下是模型对实际${neutralFull?'全幅原图':'局部裁片'}的可见变化记录（未认证），新增或消失不等于更好或更差，不是用户评价或合格判定。</p>
            ${rows('模型报告的新增特征',comparison.gains)}${rows('模型报告的消失特征',comparison.losses)}${rows('未解决或未知',comparison.unresolved)}
            <p>原回应 provisional_base 是旧格式占位，不用于选择下一参照。</p>`
            :legacy?`<p>以下是模型对实际父子图的未认证比较，不是用户评价或合格判定。</p>
              ${rows('模型报告的收益',comparison.gains)}${rows('模型报告的损失',comparison.losses)}${rows('未解决或未知',comparison.unresolved)}
              <p>临时下一观察参照：${esc(baseLabel)}；不更改用户选择。</p>`
              :`<p>比较合同或实际${version==='full_visible_delta_v1'?'全幅':'局部'}绑定待核对；仅显示保存原件，不推测字段含义或从原回应选择参照。</p>`}
            <p>理由（未认证）：${esc(comparison.reason??'未记录')}</p>
            <details><summary>${esc(observationLabel)}</summary><pre>${esc(JSON.stringify(comparison.current_observation??null,null,2))}</pre></details>
            <details><summary>${neutralFull?'全幅变化完整原回应（未认证）':neutral?'局部变化完整原回应（未认证）':'父子比较完整原回应（未认证）'}</summary><pre>${esc(JSON.stringify(comparison,null,2))}</pre></details>`
            :'<p>尚无保存的父子比较；不以回合状态认定出现收益。</p>'}
          ${purposeHtml}
        </details>`;
      }).join('')}`;
  }
  function candidatesHtml(record){
    const candidates=record?.candidates||[],flags=describe(record);
    return candidates.length?candidates.map(candidate=>{
      const image=imageHref(candidate.image_url),selected=candidate.artwork_id===record.selected_artwork_id,hasWork=candidate.artwork_id&&image&&candidate.status==='completed';
      const method=({text:'文本候选',sketch:'视觉支撑候选',handoff:'模型交接候选',research_pixels:'研究原图全幅像素引导候选',local_edit:'局部修改候选'})[candidate.method]||candidate.method||'候选';
      const source=candidate.source_artwork_id||candidate.parent_artwork_id;
      const reasonLabel=candidate.method==='local_edit'?'本次修改原话（效果待比较）':candidate.method==='research_pixels'?'此次显式采用理由（全幅会继承旧图内容，效果待比较）':'方案中的方法建议（未验证）';
      const examining=candidate.artwork_id&&candidate.artwork_id===record.result_cycle?.cursor_artwork_id;
      const direction=directionSlots(record).find(slot=>slot.id===candidate.direction_id);
      const realParent=candidate.parent_artwork_id&&record.candidates?.some(parent=>parent.artwork_id===candidate.parent_artwork_id&&parent.status==='completed'&&imageHref(parent.image_url));
      const revision=realParent&&candidate.direction_role!=='independent_candidate';
      const directionLabel=direction?`方向 ${direction.ordinal} · ${revision?'修改版本 · ':''}`:'';
      return `<article class="creative-candidate${selected?' selected':''}"><header><strong>${esc(directionLabel)}${esc(method)}</strong><span>${selected?`${esc(selectedVersionLabel(record))} · 非质量认证`:esc(statusLabel(candidate.status))}</span></header>
        ${examining?'<p>系统当前审读版本 · 临时参照，不是用户选图或通过判定</p>':''}
        ${image?`<a href="${esc(image)}" target="_blank" rel="noopener"><img src="${esc(image)}" alt="候选 ${esc(candidate.artwork_id||candidate.job_id||'')}" loading="lazy"></a>`:'<p class="creative-pending">尚无实际输出图片</p>'}
        <small>${esc(candidate.artwork_id||candidate.job_id||'尚未派发')}<br>${esc(candidate.file_path||'')}</small>${candidate.reason?`<p>${reasonLabel}：${esc(candidate.reason)}</p>`:''}${candidate.error?`<p class="creative-error">${esc(candidate.error)}</p>`:''}
        ${source?`<p>源版本：<button type="button" data-creative-art="${esc(source)}">${esc(source)}</button> · ${candidate.status==='cancelled'&&!image?'后继已停止，未生成新版本；前图保留':'新版本可比较、选用或回退，不覆盖原图'}</p>`:''}
        ${hasWork?`<div class="creative-card-actions"><button type="button" data-creative-art="${esc(candidate.artwork_id)}">查看 · 评价 / 圈改</button><button type="button" data-creative-select="${esc(candidate.artwork_id)}" ${!flags.canChoose?'disabled':''}>${selected&&selectionOrigin(record)!=='assistant_validation'?'当前选用':'选用此版本'}</button><button type="button" data-creative-revert="${esc(candidate.artwork_id)}" ${!flags.canChoose||selected?'disabled':''}>回到此版本</button></div>`:''}</article>`;
    }).join(''):'<p class="creative-pending">准备与排队不代表图片已经生成；实际候选会显示在这里。</p>';
  }
  function create({read,storage,onRecord=()=>{},onConnection=()=>{},schedule=setTimeout,unschedule=clearTimeout,idFactory=requestId,interval=2500}){
    let current=null,busy=false,timer=null,epoch=0;
    const saved=()=>{try{return JSON.parse(storage?.getItem(storageKey)||'null');}catch{return null;}};
    const save=value=>{try{storage?.setItem(storageKey,JSON.stringify(value));}catch{}};
    const stop=()=>{epoch++;if(timer!==null)unschedule(timer);timer=null;};
    function actionSeen(record,pending){
      if(!pending)return false;
      return record.last_action_request_id===pending.body.request_id||Boolean(record.action_receipts?.[pending.body.request_id])||Boolean(record.mutation_requests?.[pending.body.request_id]);
    }
    function accept(record,token){
      if(token!==epoch)return null;
      if(!record?.id||!record.status)throw new Error('创作接口未返回有效任务；未把空响应当作成功');
      if(current?.id===record.id&&Number.isFinite(current.revision)&&record.revision<current.revision)return current;
      current=record;const prior=saved()||{};
      const pending=actionSeen(record,prior.pending_action)?null:prior.pending_action;
      save({...prior,id:record.id,record,pending_action:pending,creation_unknown:false});onRecord(record);return record;
    }
    function arm(record,token){
      if(token!==epoch||!describe(record).working)return;
      if(timer!==null)unschedule(timer);
      timer=schedule(()=>{timer=null;refresh().catch(()=>{});},interval);
    }
    async function refresh(expectedId,expectedVersion){
      const id=expectedId||current?.id||saved()?.id;if(!id)return restore();
      const token=expectedVersion??epoch;
      try{const received=await read(`/api/creative-sessions/${encodeURIComponent(id)}`);
        if(expectedId&&received?.id!==expectedId)throw new Error('返回的创作编号与原记录不一致，未采用其它任务状态。');
        if(expectedId&&current?.id!==expectedId)return received;
        const record=accept(received,token);
        if(record){onConnection(null);arm(record,token);}return record;
      }catch(error){if(token===epoch){onConnection(error);if(error.status!==404&&describe(current).working)arm(current,token);}throw error;}
    }
    async function restore(id){
      stop();const token=epoch,prior=saved();current=null;
      try{
        let record;
        if(id||prior?.id)record=await read(`/api/creative-sessions/${encodeURIComponent(id||prior.id)}`);
        else if(prior?.request_id){
          const rows=await read(`/api/creative-sessions?request_id=${encodeURIComponent(prior.request_id)}&limit=1`);
          record=Array.isArray(rows)?rows[0]:null;
          if(!record)throw new Error('上次提交回执未知，暂未找到任务。刷新只查询，不重新派发；请稍后再次读取。');
        }else return null;
        const result=accept(record,token);if(result){onConnection(null);arm(result,token);}return result;
      }catch(error){if(token===epoch)onConnection(error);throw error;}
    }
    return {saved,current:()=>current,busy:()=>busy,observationVersion:()=>epoch,stop,refresh,restore,
      async start(value){
        if(busy)throw new Error('正在提交，请等待已有请求的回执');
        const body=input(value),prior=saved();
        if(prior?.creation_unknown){await restore();throw new Error('已有未核验提交；已仅读取它的状态。请先核对任务，不重复生成。');}
        stop();busy=true;const token=epoch,payload={request_id:idFactory(),...body};
        save({request_id:payload.request_id,input:body,creation_unknown:true});
        try{const result=accept(await read('/api/creative-sessions',{method:'POST',body:JSON.stringify(payload)}),token);
          if(result){onConnection(null);arm(result,token);}return result;
        }catch(error){if(error.status>=400&&error.status<500&&error.status!==408)save({request_id:payload.request_id,input:body,creation_unknown:false,rejected:true});onConnection(error);throw error;}
        finally{busy=false;}
      },
      async action(action,details={}){
        if(busy)throw new Error('正在提交，请等待已有请求的回执');
        if(!current?.id)throw new Error('请先读取对应的创作任务');
        const review=reviewState(current);
        if(action==='generate'&&(review.prose||review.relational||review.canonical)&&!describe(current).canGenerate)
          throw new Error('当前保存原件尚无可提交的当次完整条件；没有生成或恢复。');
        if(action==='resume'&&describe(current).canPrepareNative)
          throw new Error('这份暂停原稿需要明确重新准备条件；不恢复旧步骤或自动生图。');
        if(action==='resume'&&describe(current).resumeUnavailableReason)
          throw new Error(describe(current).resumeUnavailableReason+'；没有恢复或增加额度。');
        if(saved()?.pending_action){await refresh();if(saved()?.pending_action)throw new Error('上次操作结果尚未核验，仅读取已有状态；不会重派同一操作。');}
        if(action==='complete_comparison'){
          const flags=describe(current);
          if(!flags.canCompleteComparison
            ||details.comparison_turn_id!==flags.comparisonCompletion.capability?.turn_id)
            throw new Error('当前没有明确可补完的已保存两图比较；先重新读取，没有调用作者或生图。');
          if(Object.keys(details).some(name=>!['comparison_turn_id','initiator'].includes(name)))
            throw new Error('补完比较只使用已保存的两图与比较输入，不接受反馈、参数覆盖或生图要求。');
        }
        if(action==='prepare_native'){
          const flags=describe(current);
          if(!flags.canPrepareNative)throw new Error(flags.prepareNativeUnavailableReason+'；没有重新准备或生图。');
          const cap=flags.nativePreparation,budget=details.max_calls;
          if(cap.language_new_call_required||budget!==undefined){
            if(!Number.isInteger(budget)||budget<cap.minimum_max_calls||budget>cap.maximum_max_calls||budget>8)
              throw new Error(`本次准备须明确 ${cap.minimum_max_calls}–${cap.maximum_max_calls} 的模型调用总上限；没有新增调用或生图。`);
          }
        }
        if(['continue','local_edit','regenerate'].includes(action)){
          const flags=describe(current,{initiator:details.initiator}),guard={continue:['canContinue','continueUnavailableReason'],local_edit:['canLocalEdit','localEditUnavailableReason'],regenerate:['canRegenerate','regenerateUnavailableReason']}[action];
          if(!flags[guard[0]])throw new Error(flags[guard[1]]+'；没有提交新的操作。');
          if(current.status==='failed'&&action!=='regenerate'&&!current.candidates?.some(candidate=>candidate.artwork_id===details.artwork_id&&candidate.status==='completed'&&imageHref(candidate.image_url)))
            throw new Error('请选择本失败任务中已完成的真实候选；没有提交新的操作。');
        }
        const id=current.id,token=epoch,body={...details,request_id:idFactory(),expected_revision:current.revision,action};
        busy=true;save({...saved(),pending_action:{id,body,unknown:true}});
        try{const result=accept(await read(`/api/creative-sessions/${encodeURIComponent(id)}/action`,{method:'POST',body:JSON.stringify(body)}),token);
          if(result){save({...saved(),pending_action:null});onConnection(null);arm(result,token);}return result;
        }catch(error){if(error.status>=400&&error.status<500&&error.status!==408)save({...saved(),pending_action:null});onConnection(error);throw error;}
        finally{busy=false;}
      },
      async forArtwork(id){const rows=await read(`/api/creative-sessions?artwork_id=${encodeURIComponent(id)}&limit=1`);return Array.isArray(rows)?rows[0]||null:null;},
    };
  }
  function mount({document:doc,read,storage,showSession,loadArtworks,openArtwork,selectView,onRecord=()=>{}}){
    bindSamplingParameterSources(doc);
    const $=id=>doc.getElementById(id),panel=$('creativeSessionPanel'),dialog=$('creativeReviewDialog');
    $('sessionLive').append(panel);
    let cyclePanel=$('creativeResultCycle');
    if(!cyclePanel&&doc.createElement){
      cyclePanel=doc.createElement('details');cyclePanel.id='creativeResultCycle';cyclePanel.className='review-notes';cyclePanel.hidden=true;
      const anchor=$('creativeCandidates');
      if(anchor?.parentNode?.insertBefore)anchor.parentNode.insertBefore(cyclePanel,anchor);
      else panel.append(cyclePanel);
    }
    let directionPanel=$('creativeDirectionPlan');
    if(!directionPanel&&doc.createElement){
      directionPanel=doc.createElement('details');directionPanel.id='creativeDirectionPlan';directionPanel.className='review-notes preparation-history';directionPanel.hidden=true;
      const anchor=cyclePanel?.parentNode?cyclePanel:$('creativeCandidates');
      if(anchor?.parentNode?.insertBefore)anchor.parentNode.insertBefore(directionPanel,anchor);else panel.append(directionPanel);
    }
    let current=null,nativeBase=null,nativeDirty=false,nativeSession='',nativeBudgetSignature='',candidateSignature='',reviewSignature='',cycleSignature='',directionSignature='',recoveryBusy=false,recoveryFeedback=null;
    const notify=(text,bad=false)=>{ $('creativeConnection').textContent=text;$('creativeConnection').classList.toggle('bad',bad); };
    const controller=create({read,storage,onRecord:render,onConnection(error){
      notify(error?(error.status===404?'该创作任务不存在（404）。读取已停止，不会重新生成。':`连接或提交未确认：${error.message}。只重读状态，不自动重派。`):'');
    }});
    function syncControls(record,flags=describe(record,{pending_action:controller.saved()?.pending_action,busy:controller.busy()})){
      const recovery=receiptRecovery(record),inspectButton=$('creativeInspectJob'),feedback=$('creativeReceiptFeedback');
      if(inspectButton){inspectButton.hidden=!recovery.visible;inspectButton.disabled=recoveryBusy||controller.busy()||!recovery.allowed;inspectButton.title=recovery.reason||'只核对已记录的原任务，不接旧图、不重派，也不认证作品成功。';}
      if(feedback){const saved=recoveryFeedback?.sessionId===record.id&&(!record.active_job_id||record.active_job_id===recoveryFeedback.jobId)?recoveryFeedback:null;
        feedback.hidden=!saved&&(!recovery.visible||!recovery.reason);feedback.textContent=saved?.text||(recovery.visible?recovery.reason:'');feedback.classList.toggle('bad',Boolean(saved?.bad));}
      $('creativeRefresh').disabled=recoveryBusy;
      for(const button of panel.querySelectorAll('[data-creative-action]')){
        button.disabled=recoveryBusy||controller.busy()||!({pause:flags.canPause,resume:flags.canResume,stop:flags.canStop,regenerate:flags.canRegenerate,complete_comparison:flags.canCompleteComparison})[button.dataset.creativeAction];
        if(button.dataset.creativeAction==='complete_comparison'){
          const completion=flags.comparisonCompletion;
          button.hidden=!completion.newActionAvailable;
          button.textContent='补完本次比较';
          button.title=completion.capability
            ?`只比较已保存的两张原图；至多新增 ${completion.capability.maximum_new_calls} 次比较调用，不调用作者、不生成新图、不选用作品。`
            :'当前没有明确可补完的已保存比较';
        }
        if(button.dataset.creativeAction==='regenerate'){
          button.textContent=flags.userOperationBudget?'新回合 · 重新生成':'在本任务内重新生成';
          button.title=flags.regenerateUnavailableReason||(flags.userOperationBudget?'开始新的有界回合，原图、原句与历史保留；不重跑旧失败步骤。':'明确提出新的构思并生成候选，不恢复或复跑失败步骤。');
        }
        if(button.dataset.creativeAction==='resume')button.title=flags.resumeUnavailableReason||'继续原回合，不增加预算';
      }
      const completionNote=$('creativeComparisonCompletionNote'),completion=flags.comparisonCompletion;
      if(completionNote){
        completionNote.hidden=!completion.newActionAvailable&&!completion.paused;
        completionNote.textContent=completion.paused
          ?completion.resumeExisting?'本次补比较已暂停；“继续本回合”继续同一比较，不增加额度、不生成新图。'
            :record?.comparison_completion?.reason||'本次补比较已暂停，继续资格或调用回执尚未确认；先重新读取，不重派。'
          :completion.capability?`使用已保存的两张原图及本次比较输入，至多新增 ${completion.capability.maximum_new_calls} 次比较调用，0 张新图；原失败日志保留，完成比较不是作品接受或选图。`:'';
      }
      $('creativeReview').disabled=!reviewState(record).canReview;$('creativeGenerate').disabled=recoveryBusy||controller.busy()||!flags.canGenerate;
      $('creativeGenerate').hidden=!flags.canGenerate;$('creativeAwaiting').hidden=!flags.canGenerate;
    }
    function render(record){
      current=record;panel.hidden=false;onRecord(record);
      const flags=describe(record,{pending_action:controller.saved()?.pending_action,busy:controller.busy()});showSession(flags.title,flags.percent,flags.detail,'creative');
      $('cancelPreparation').classList.add('hidden');$('sessionResult').classList.add('hidden');
      $('creativeTaskId').textContent=`${record.id} · 修订 ${record.revision}${record.input_snapshot?.initiator==='assistant_validation'?' · 助手功能验收':''}`;syncControls(record,flags);
      $('creativeBudget').textContent=(record.remaining_budget?`${flags.userOperationBudget?'本回合剩余':'剩余预算'}：${Object.entries(record.remaining_budget).map(([k,v])=>`${({jobs:'图片',calls:'模型调用',active_seconds:'执行秒数'})[k]||k} ${k==='active_seconds'&&Number.isFinite(v)?Math.floor(v):v}`).join(' · ')}`:'')
        +(flags.userOperationBudget?`\n下一次明确继续、圈改或重新生成会开始新的有界回合：至多 ${flags.userOperationBudget.jobs} 图 / ${flags.userOperationBudget.calls} 次模型调用 / ${Math.floor(flags.userOperationBudget.active_seconds)} 秒。原图、原句与总记录保留；继续本回合不会增加预算。`:'');
      $('creativeSupportLimit').hidden=!record.support_limit;
      $('creativeSupportLimit').textContent=!record.support_limit?'':typeof record.support_limit==='string'?record.support_limit:record.support_limit?.detail||JSON.stringify(record.support_limit);
      const directionKey=key([record.direction_plan,record.candidates,record.status,record.generation_request?.author_proposal?.direction_assignment]);
      if(directionPanel&&directionKey!==directionSignature){
        directionSignature=directionKey;const html=directionPlanHtml(record);directionPanel.hidden=!html;
        const opened=new Set([...directionPanel.querySelectorAll('[data-direction-detail]')].filter(node=>node.open).map(node=>node.dataset.directionDetail));
        directionPanel.innerHTML=html;
        for(const node of directionPanel.querySelectorAll('[data-direction-detail]'))node.open=opened.has(node.dataset.directionDetail);
        for(const button of directionPanel.querySelectorAll('[data-creative-art]'))button.onclick=()=>openArtwork(button.dataset.creativeArt,{sessionId:record.id});
      }
      const cycleKey=key([record.result_cycle,record.input_snapshot?.result_guided,record.selected_artwork_id,record.selection_origin,
        record.continuation?.choice_rejection,record.continuation?.reference_choice_rejection]);
      if(cyclePanel&&cycleKey!==cycleSignature){
        cycleSignature=cycleKey;const html=resultCycleHtml(record);cyclePanel.hidden=!html;
        const openTurns=new Set([...cyclePanel.querySelectorAll('[data-result-turn]')].filter(node=>node.open).map(node=>node.dataset.resultTurn));
        cyclePanel.innerHTML=html;
        for(const node of cyclePanel.querySelectorAll('[data-result-turn]'))node.open=openTurns.has(node.dataset.resultTurn);
        for(const button of cyclePanel.querySelectorAll('[data-creative-art]'))button.onclick=()=>openArtwork(button.dataset.creativeArt,{sessionId:record.id});
      }
      const signature=key([record.candidates,record.direction_plan?.slots,record.selected_artwork_id,record.selection_origin,record.result_cycle?.cursor_artwork_id,flags.canChoose,controller.busy()]);
      if(signature!==candidateSignature){candidateSignature=signature;$('creativeCandidates').innerHTML=candidatesHtml(record);
        for(const b of panel.querySelectorAll('[data-creative-art]'))b.onclick=()=>openArtwork(b.dataset.creativeArt,{sessionId:record.id});
        for(const b of panel.querySelectorAll('[data-creative-select]'))b.onclick=()=>perform('select',{artwork_id:b.dataset.creativeSelect});
        for(const b of panel.querySelectorAll('[data-creative-revert]'))b.onclick=()=>perform('revert',{artwork_id:b.dataset.creativeRevert});
        if(record.candidates?.some(c=>c.artwork_id))Promise.resolve(loadArtworks()).catch(()=>{});
      }
      const newReview=key([record.id,record.status,record.plan,record.native_prompts,record.relational_picture,
        record.prose_picture_choice,record.translated_prose_picture_choice,record.prose_picture_receipt,
        record.saved_prose_preparation,
        record.request_reading,record.feedback,
        record.canonical_picture_choice,record.translated_canonical_picture_choice,record.picture_projection,
        record.generation_request,record.error,record.calls]);
      if(newReview!==reviewSignature){reviewSignature=newReview;populateReview(record);}
      $('creativeSupportMaterials').hidden=!record.materials?.some(m=>imageHref(m.image_url));
      $('creativeMaterialImages').innerHTML=(record.materials||[]).filter(m=>imageHref(m.image_url)).map(m=>{
        const localization=m.kind==='localization_mask_unverified'?'定位掩膜：白色区域为拟修改范围，模型定位未核准确性'
          :m.kind==='overlay_unverified'?'模型定位预览：叠图仅供查看，未核准确性':'';
        const scope=localization?` · ${esc(localization)} · 源作品：${esc(m.source_artwork_id||'未记录')}`:'';
        return `<figure><a href="${esc(imageHref(m.image_url))}" target="_blank" rel="noopener"><img src="${esc(imageHref(m.image_url))}" alt="实际视觉支撑 ${esc(m.id)}" loading="lazy"></a><figcaption>${esc(m.id)}${scope} · 非成品，不认证构图、身份或表达正确</figcaption></figure>`;
      }).join('');
      syncReviewFooter(record);
    }
    function populateReview(record){
      const review=reviewState(record),parts=record.relational_picture;
      const proseLanguageApplied=record.prose_picture_receipt?.language_application?.kind==='host_applies_only_original_cjk_drawing_fields';
      const protectedProposal=review.prose||review.relational||review.canonical;
      const nativeReadable=review.nativeReadable??review.complete;
      const direction=nativeReadable?reviewedDirection(record):null;
      if($('creativeReviewTitle'))$('creativeReviewTitle').textContent=direction?`${record.status==='needs_confirmation'?'当前待审':'保存的'}方向 ${direction.ordinal} · 方案与实际模型条件`:'方案与实际模型条件';
      $('creativeOriginal').textContent=record.input_snapshot?.sentence||'';
      if($('creativeRequestUnderstandingDetails'))$('creativeRequestUnderstandingDetails').textContent=JSON.stringify(requestReadingDetails(record),null,2);
      if($('creativeRequestUnderstandingNotice'))$('creativeRequestUnderstandingNotice').textContent=record.request_reading==null
        ?'该历史记录没有独立的原句理解步骤；查看不会自动补做，旧作者稿不被当成来源解释。'
        :`以下是对原句的独立解释，当前保存状态：${record.request_reading.status??'未记录'}。${record.request_reading.raw==null?'尚无完整原回应，空内容不表示已理解或可生成。':''}完整原文与随后有归属的反馈可以纠正它；模型返回、引用核对和格式合法不等于理解或画面通过。`;
      if($('creativePictureMethod'))$('creativePictureMethod').textContent=`画面组织：${pictureMethodLabel(pictureAuthoringMethod(record))}${record.input_snapshot?.picture_authoring_method?'':'（旧记录未注明方法，按原有结构化方式查看）'} · 保存的构思配置：${authoringProfileLabel(authoringProfile(record))}${record.input_snapshot?.authoring_profile?'':'（旧记录未注明，按 4B 查看）'}`;
      $('creativePlanText').textContent=protectedProposal&&!review.complete
        ?review.explicitNativeOverride?'以下模型作者原件为手改前的历史。当前正负文本是已保存的用户明确版本，不声称与旧构思语义同步。'
          :review.prose?record.prose_picture_choice?.scene_narrative||'完整画面描述尚未返回；已保存的调用和错误可在细节中查看。'
            :review.canonical?'以下为当前整体构思的模型作者原件，尚未关联当次完整生图条件；未认证理解、角色身份或画面效果。即使留有旧方案，也不作为本次完整稿。'
          :'以下为当前关系构思的原回应，尚未成为完整生图条件；未认证理解、角色身份或画面效果。即使留有旧方案，也不作为本次完整稿。'
        :review.prose?record.prose_picture_choice.scene_narrative:record.plan?.scene_text||record.plan?.scene_narrative||'尚未形成可审阅的方案';
      $('creativePlanLogic').textContent=review.prose
        ?(review.complete?'完整描述原件已关联当次编译条件；未认证理解、角色身份或画面效果。\n'
          :review.explicitNativeOverride?'完整描述保留为手改前历史；当前实际条件采用已保存的用户明确版本。\n'
            :'完整描述原件尚未关联当次完整编译条件，目前仅供查看。\n')
          +'动作、物体、空间与光色保留在完整描述中，结构字段未单独抽取；空字段不表示画面没有这些内容。\n'
          +(record.translated_prose_picture_choice!=null
            ?proseLanguageApplied?'语言模型实际返回与程序应用后的稿件分别保留；仅应用作用记录所列语言字段，其余字段由程序沿用作者原稿。\n'
              :'已保存历史语言稿，但该记录未注明程序应用范围；不能将它称为已应用的编译稿。\n':'' )
          +(record.status==='failed'?'任务已失败；原件与调用保留，不代表仍在运行。\n':'')
          +`模型作者解释（未认证）：\n${record.prose_picture_choice?.visual_logic||''}`
        :review.canonical
        ?(review.complete?'原件、转换稿与机械投影回执已关联当次完整条件；不认证理解、角色身份或艺术效果。\n'
          :review.explicitNativeOverride?'原件和机械投影仅为手改前历史，不证明当前原生文本兑现了此构思。\n'
            :'原件仅供阅读，尚未关联当次完整编译条件；不是生图结果或理解认证。\n')
          +(record.status==='failed'?'任务已失败；保存原件保留，不代表仍在运行。\n':'')
          +`模型作者解释（未认证）：\n${record.canonical_picture_choice?.visual_logic||record.translated_canonical_picture_choice?.visual_logic||''}`
        :review.relational
        ?(review.complete?'两份原回应已由当次编译条件的同一回执关联；这不认证理解或艺术效果。\n':'')
          +(record.status==='failed'?'任务已失败；以下是停处的保存阶段，不代表仍在运行。\n':'')
          +`当前关系构思状态：${parts?.status||'尚无记录'}`+(review.complete?`\n${record.plan?.visual_logic||''}`:'')
        :record.plan?.visual_logic||'';
      $('creativePlanDetails').textContent=JSON.stringify(review.prose?{
        '完整画面描述原件（模型作者，未认证）':record.prose_picture_choice??null,
        '语言调用原件（模型实际返回，不等同于应用后稿）':(record.calls||[]).map((call,index)=>({index,call}))
          .filter(({call})=>call?.purpose==='native_language').map(({index,call})=>({'调用序号':index,'原调用记录':call})),
        '程序应用后的语言稿（包含沿用字段，不是逐字模型返回）':proseLanguageApplied?record.translated_prose_picture_choice??null:null,
        '历史语言稿（应用范围未注明）':!proseLanguageApplied?record.translated_prose_picture_choice??null:null,
        '语言字段应用范围与来源（程序作用记录）':record.prose_picture_receipt?.language_application??null,
        '编译与语言应用收据（不认证理解或效果）':record.prose_picture_receipt??null,
        '结构字段状态':'未单独抽取；完整描述是画面内容原件，空结构不表示内容缺失',
        '当次完整编译条件关联':review.complete,'当次完整方案':review.complete?record.plan:null,
        '当前已保存的用户原生覆盖（旧构思仅为历史）':review.explicitNativeOverride,
        '实际显示或编译错误':record.error??null,'模型调用回执（原始，未认证）':record.calls||[],
      }:review.canonical?{
        '模型作者原件（canonical，未认证）':record.canonical_picture_choice??null,
        '语言转换稿（存在时仍未认证）':record.translated_canonical_picture_choice??null,
        '机械投影回执（不认证理解或效果）':record.picture_projection??null,
        '当次完整编译条件关联':review.complete,'当次完整方案':review.complete?record.plan:null,
        '当前已保存的用户原生覆盖（旧构思仅为历史）':review.explicitNativeOverride,
        '实际显示或编译错误':record.error??null,'模型调用回执（原始，未认证）':record.calls||[],
      }:review.relational?{
        '当前关系构思状态':parts?.status??null,'第一份原回应（core）':parts?.core??null,
        '第二份原回应（finish）':parts?.finish??null,'合并回执（不认证理解或效果）':parts?.receipt??null,
        '当次完整编译条件关联':review.complete,'当次完整方案':review.complete?record.plan:null,
      }:record.plan||{},null,2);
      const channel=target(protectedProposal&&nativeReadable?record.generation_request:record.input_snapshot?.config),prompts=record.native_prompts?.[channel];
      const availablePrompts=protectedProposal&&!nativeReadable?null:prompts;
      const nativeGrid=$('creativePositive')?.closest?.('.contract-grid');
      if(nativeGrid)nativeGrid.hidden=protectedProposal&&!nativeReadable;
      $('creativeNativeTarget').textContent=protectedProposal&&!nativeReadable?'未关联当前条件':channel;
      $('creativeNativeUnavailable').hidden=Boolean(availablePrompts);
      const nextBase=availablePrompts?{positive:availablePrompts.positive||'',negative:availablePrompts.negative||''}:null;
      if(!nativeDirty||nativeSession!==record.id||!describe(record).canGenerate){nativeBase=nextBase;nativeDirty=false;nativeSession=record.id;
        $('creativePositive').value=nextBase?.positive||'';$('creativeNegative').value=nextBase?.negative||'';
      }
      $('creativePositive').disabled=!availablePrompts;$('creativeNegative').disabled=!availablePrompts;
      $('creativePositive').readOnly=!describe(record).canGenerate;$('creativeNegative').readOnly=!describe(record).canGenerate;
      if($('creativeConditionProvenance')){
        const submitted=record.generation_request,proposal=submitted?.author_proposal;
        $('creativeConditionProvenance').textContent=JSON.stringify({
          '当次条件可查看':Boolean(availablePrompts),'实际目标':availablePrompts?channel:null,
          '原生编译信息（条目存在不认证画面关系）':availablePrompts?Object.fromEntries(Object.entries(prompts).filter(([name])=>!['positive','negative'].includes(name))):null,
          '保存的条件来源':proposal??null,
          '条件来源用途':availablePrompts?'对应当前可查看的保存条件；不认证语义或画面效果':'仅保留记录，尚未与当前完整构思关联；不能作为当前可生成条件',
          '本任务采样来源':record.input_snapshot?.parameter_sources??null,
          '当前已保存的用户原生覆盖':Boolean(record.native_user_override),
        },null,2);
      }
      syncReviewFooter(record);
      nativeNotice();
    }
    function syncReviewFooter(record){
      const flags=describe(record,{pending_action:controller.saved()?.pending_action,busy:controller.busy()}),cap=flags.nativePreparation;
      const prepare=flags.canPrepareNative,button=$('creativeReviewGenerate'),budget=$('creativePrepareNativeBudget'),field=$('creativePrepareNativeMaxCalls');
      button.hidden=!flags.canGenerate&&!prepare;button.disabled=recoveryBusy||controller.busy()||button.hidden;
      const direction=flags.canGenerate?reviewedDirection(record):null;
      button.textContent=prepare?(cap.language_new_call_required?'用完整画面原稿重新准备（新增 1 次语言调用，不生图）':'用这份完整画面原稿重新准备条件（不生图）')
        :direction?`确认当前方向 ${direction.ordinal} 的条件，生成这一候选`:'确认当前条件，生成候选';
      if(budget)budget.hidden=!prepare||!cap.language_new_call_required;
      if(field){
        field.disabled=recoveryBusy||controller.busy()||!prepare;
        if(prepare&&cap.language_new_call_required){
          const signature=key([record.id,cap.minimum_max_calls,cap.suggested_max_calls,cap.maximum_max_calls]);
          if(signature!==nativeBudgetSignature){nativeBudgetSignature=signature;field.value=String(cap.suggested_max_calls);}
          field.min=String(cap.minimum_max_calls);field.max=String(cap.maximum_max_calls);
        }
      }
      if($('creativePrepareNativeBudgetNote'))$('creativePrepareNativeBudgetNote').textContent=prepare&&cap.language_new_call_required
        ?`此次只为保存原稿新增 1 次语言调用，模型调用总上限须为 ${cap.minimum_max_calls}–${cap.maximum_max_calls}；不重新构思、不自动生图，不修改下一新任务的预算。`:'';
    }
    function nativeNotice(){
      const now={positive:$('creativePositive').value,negative:$('creativeNegative').value};
      nativeDirty=Boolean(nativeBase&&key(now)!==key(nativeBase));
      const review=reviewState(current);
      if((review.prose||review.relational||review.canonical)&&!(review.nativeReadable??review.complete)){
        const flags=describe(current,{pending_action:controller.saved()?.pending_action,busy:controller.busy()});
        $('creativeNativeNotice').textContent='当前原回应尚未关联本次完整编译条件；不填入原生文本，不把旧方案或部分稿当成可生成条件。'
          +(flags.canPrepareNative?`\n${flags.nativePreparation.language_new_call_required?'保留作者原稿，明确新增一次语言准备。':'复用保存的作者原稿及可用语言材料，无需新增模型调用。'}完成后仅供审阅，不自动生图；原失败与调用保留，不代表语义已核验。`
            :current?.saved_prose_preparation?.reason?`\n${current.saved_prose_preparation.reason}`:'');return;
      }
      $('creativeNativeNotice').textContent=(review.explicitNativeOverride?'当前正负是已保存的用户明确版本；旧构思仅作历史，不声称语义同步。\n':'')
        +($('creativePositive').readOnly?'这是当次保存的实际原生条件，目前仅供查看。新方案待确认时才可编辑，不会把未提交的输入当成已用条件。':nativeDirty?(now.negative.trim()?'将提交你明确修改的实际目标原生文本；旧方案不据此获得语义认证。':'你显式清空了负面条件；可能增加碎裂、错误肢体等风险。不会静默补回。'):'原生正负条件未修改，生成时由服务端使用保存的完整条件。');
      const direction=reviewedDirection(current);
      if(direction)$('creativeNativeNotice').textContent+=`\n这些条件仅对应当前方向 ${direction.ordinal}；确认与生成只作用于当前项，本次确认不准备或批准其它方向，既有状态按各项记录查看。`;
    }
    async function perform(action,details={},options={}){
      if(recoveryBusy){const error=new Error('正在核对原任务回执，请等待本次读取；没有提交新的操作。');notify(error.message,true);if(options.throwErrors)throw error;return null;}
      for(const b of panel.querySelectorAll('button'))b.disabled=true;
      $('creativeReviewGenerate').disabled=true;
      try{const result=await controller.action(action,details);if(result)render(result);return result;}
      catch(error){if(error.status===409)await controller.refresh().catch(()=>{});notify(error.message,true);if(options.throwErrors)throw error;return null;}
      finally{if(current)render(current);$('creativeRefresh').disabled=false;}
    }
    async function generate(){
      if(recoveryBusy)return null;
      if(!current)throw new Error('没有创作任务');
      const review=reviewState(current);
      if((review.prose||review.relational||review.canonical)&&!describe(current).canGenerate){notify('当前构思尚无可提交的当次完整条件；原回应仅供查看，没有生成或恢复。',true);return null;}
      if(nativeDirty&&!$('creativePositive').value.trim()){notify('实际目标正向条件不能为空。',true);return null;}
      const override=nativeDirty?{native_override:{positive:$('creativePositive').value,negative:$('creativeNegative').value}}:{};
      const record=await perform('generate',override);if(record&&dialog.open)dialog.close();return record;
    }
    async function reviewAction(){
      const flags=describe(current,{pending_action:controller.saved()?.pending_action,busy:controller.busy()});
      if(!flags.canPrepareNative)return generate();
      const details=flags.nativePreparation.language_new_call_required
        ?{max_calls:Number($('creativePrepareNativeMaxCalls')?.value)}:{};
      // This action prepares saved material only. A later, separate click on
      // the existing generation action is required once current conditions bind.
      return perform('prepare_native',details);
    }
    async function inspectOriginalJob(){
      const target=receiptRecovery(current);
      if(recoveryBusy||controller.busy()||!target.allowed)return null;
      const sessionId=current.id,jobId=target.jobId,errors=[];
      recoveryBusy=true;controller.stop();const version=controller.observationVersion();recoveryFeedback={sessionId,jobId,bad:false,
        text:`正在核对原图像任务 ${jobId}；尚未确认，不会接图或重派。`};render(current);
      try{
        try{await read(`/api/jobs/${encodeURIComponent(jobId)}/inspect-engine`,{method:'POST'});}
        catch(error){errors.push(`原图像任务核对接口未确认：${error.message}`);}
        try{await controller.refresh(sessionId,version);}
        catch(error){errors.push(`原创作状态读取未确认：${error.message}`);}
        recoveryFeedback={sessionId,jobId,bad:Boolean(errors.length),text:errors.length?errors.join('\n'):
          `原图像任务 ${jobId} 的核对请求已返回并重新读取原状态；终态不代表图片或作品成功，不接旧图、不重派。必要时停止旧创作，再明确开始新构想。`};
        return {sessionId,jobId,error:Boolean(errors.length)};
      }finally{recoveryBusy=false;if(current){
        if(controller.observationVersion()===version&&!panel.hidden)render(current);else syncControls(current);}}
    }
    if($('creativeInspectJob'))$('creativeInspectJob').onclick=inspectOriginalJob;
    for(const b of panel.querySelectorAll('[data-creative-action]'))b.onclick=()=>perform(b.dataset.creativeAction,
      b.dataset.creativeAction==='complete_comparison'
        ?{comparison_turn_id:describe(current).comparisonCompletion.capability?.turn_id}:{});
    $('creativeRefresh').onclick=()=>controller.refresh().catch(()=>{});
    $('creativeReview').onclick=()=>{if(reviewState(current).canReview){populateReview(current);dialog.showModal();}};
    $('creativeGenerate').onclick=generate;$('creativeReviewGenerate').onclick=reviewAction;
    $('creativePositive').oninput=nativeNotice;$('creativeNegative').oninput=nativeNotice;
    async function owningArtworkTask(id,sessionId){
      const found=sessionId?{id:sessionId}:await controller.forArtwork(id);
      if(!found)throw new Error('此作品没有关联创作任务；可以查看、评价或独立圈改，不会补造原描述。');
      if(!current||current.id!==found.id)await controller.restore(found.id);
      if(!artworkState(current,id).belongs)throw new Error('这张图不是该任务的真实候选；请重新打开作品核对归属。');
      selectView('create');panel.hidden=false;return current;
    }
    return {controller,hasSaved:()=>{const saved=controller.saved();return Boolean(saved&&!saved.rejected&&(saved.id||saved.request_id));},
      current:()=>current,review:()=>{if(reviewState(current).canReview){populateReview(current);dialog.showModal();}},
      async start(value){notify('');panel.hidden=false;showSession('提交创作构想',null,'保留原句、配置与资源，等待同一创作任务的真实回执。','creative');
        try{const result=await controller.start(value);if(result){render(result);if(result.status==='needs_confirmation'){populateReview(result);dialog.showModal();}}return result;}
        catch(error){const rejected=error.status>=400&&error.status<500&&error.status!==408;
          showSession(rejected?'未能开始创作':'提交待核验',null,`${error.message}\n没有重派；${rejected?'请纠正实际输入或配置。':'先重新读取已有任务，不能将未知回执视为成功或失败。'}`,'creative');throw error;}},
      async restore(id){const result=await controller.restore(id);if(result){render(result);}return result;},
      hide(){panel.hidden=true;controller.stop();},
      openArtworkTask:owningArtworkTask,
      async chooseArtwork(id,sessionId,{revert=false}={}){
        await owningArtworkTask(id,sessionId);
        const receipt={pending_action:controller.saved()?.pending_action,busy:controller.busy()};
        if(!artworkState(current,id,receipt).canChoose)throw new Error('当前不能选择或回到此版本；已有在途任务和回执不被绕过。');
        return perform(revert?'revert':'select',{artwork_id:id},{throwErrors:true});
      },
      async continueArtwork(id,feedback,sessionId){
        if(!feedback?.trim())throw new Error('请写下希望保留什么、调整什么，或说明继续创作的方向。');
        await owningArtworkTask(id,sessionId);
        return perform('continue',{artwork_id:id,feedback:feedback.trim()},{throwErrors:true});
      },
      async localEditArtwork(id,regionEdit,sessionId){
        if(!regionEdit?.instruction?.trim())throw new Error('请说明圈选区域怎样修改');
        await owningArtworkTask(id,sessionId);
        return perform('local_edit',{artwork_id:id,region_edit:structuredClone(regionEdit),feedback:regionEdit.instruction},{throwErrors:true});
      },forArtwork:controller.forArtwork,
    };
  }
  root.CreativeSession={storageKey,input,samplingParameterSources,samplingValuesForNewRequest,clearSamplingParameterSources,bindSamplingParameterSources,pictureAuthoringMethod,pictureMethodLabel,authoringMethodForNewRequest,authoringProfile,authoringProfileLabel,authoringProfileForNewRequest,requestReadingDetails,target,key,reviewState,describe,artworkState,statusLabel,operationLabel,imageHref,historyHtml,receiptRecovery,candidatesHtml,resultCycleHtml,directionPlanHtml,create,mount};
  if(typeof module!=='undefined')module.exports=root.CreativeSession;
})(globalThis);
