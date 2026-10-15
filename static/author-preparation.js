/* Preparation observation never submits image jobs or guesses missing results. */
(function(root){
  const storageKey='pipeline-studio.creation-preparation';
  const terminal=new Set(['ready','needs_confirmation','failed','cancelled']);
  const preparationSteps={source_interpretation:'理解原文条件',art_direction:'构思空间与画法',
    source_reconsideration:'重新理解原文',source_revision_art:'解释变化并重构画面'};
  const stable=value=>Array.isArray(value)?value.map(stable):value&&typeof value==='object'
    ?Object.fromEntries(Object.keys(value).sort().map(key=>[key,stable(value[key])])):value;
  const key=value=>JSON.stringify(stable(value));
  function input(value){
    if(!value?.sentence?.trim())throw new Error('请先描述你想要的画面');
    return {sentence:value.sentence,style_mode:value.style_mode||'standard_anime',
      knowledge_pack:value.knowledge_pack||'',knowledge_operator:value.knowledge_operator||'',
      knowledge_appearance:value.knowledge_appearance||'',config:structuredClone(value.config||{})};
  }
  function sameInput(first,second){return key(input(first))===key(input(second));}
  function proposal(compiled,edit){
    const saved=compiled?.author_proposal;if(!saved)throw new Error('缺少本次作者提案，请重新准备');
    const positions=JSON.parse(edit.positions);
    if(!positions||Array.isArray(positions)||typeof positions!=='object')throw new Error('空间安排应为 JSON 对象');
    const result={scene_narrative:edit.scene_narrative,visual_logic:edit.visual_logic,positions};
    if(Object.prototype.hasOwnProperty.call(saved,'layout')||edit.layout?.trim()){
      const layout=edit.layout===undefined?saved.layout||[]:JSON.parse(edit.layout);
      if(!Array.isArray(layout)||layout.length>30||layout.some(value=>typeof value!=='string'||!value.trim()||value.length>2500))throw new Error('布局方案应为不超过30条非空文本的 JSON 数组');
      result.layout=layout;
    }
    return result;
  }
  function proposalChanged(compiled,edit){
    const fields=['scene_narrative','visual_logic','positions'];
    if(Object.prototype.hasOwnProperty.call(compiled.author_proposal,'layout'))fields.push('layout');
    return key(proposal(compiled,edit))!==key(Object.fromEntries(fields.map(k=>[k,compiled.author_proposal[k]??(k==='layout'?[]:{})])));
  }
  function metadata(compiled,resolved){
    const changed=Object.entries(resolved.native||{}).some(([target,words])=>
      words.positive!==compiled.native_prompts?.[target]?.positive||words.negative!==(compiled.native_prompts?.[target]?.negative??''));
    return {...structuredClone(compiled.author_proposal),condition_status:changed?'historical_after_native_override':'compiled_author_proposal_unverified',
      scene_semantic_sync_verified:false};
  }
  function failedStages(record){
    return Object.entries(record.stage_evidence||{}).filter(([,evidence])=>evidence?.status==='failed')
      .map(([name])=>preparationSteps[name]||name);
  }
  function describe(record){
    if(record.status!=='failed')return record.detail||'';
    const failed=failedStages(record);
    return [record.detail,failed.length?`失败阶段：${failed.join('、')}`:'',
      record.error?`原始错误：${record.error}`:''].filter(Boolean).join('\n\n');
  }
  function stage(record){
    if(record.status==='failed'||record.stage==='failed'){
      const failed=failedStages(record);
      return failed.length?`准备失败 · ${failed.join('、')}`:'准备失败';
    }
    const phase=record.runtime_state?.phase;
    const steps={...preparationSteps};
    if(record.source_reconsideration?.conflict_transition_review?.automatic_adoption_permitted===false)
      steps.source_revision_art='核对解释变化';
    const step=steps[record.stage]||'形成创作提案';
    const names={startup:`启动本地模型 · ${step}`,generating:step,releasing:`释放本地模型 · ${step}`};
    return names[phase]||({...steps,queued:'等待构思资源',authoring:'形成创作提案',compiling:'准备模型条件',review:'等待审阅',failed:'准备失败',cancelled:'准备已取消',interrupted:'准备被中断'})[record.stage]||'准备中';
  }
  function receipt(record){
    const value=record?.generation_submissions?.[String(record.revision)];
    if(value)return {...value,job_id:value.job_id||null};
    if(record?.iteration_submitted)return {kind:'iteration',status:record.submitted_iteration_run_id?'submitted':'unknown',run_id:record.submitted_iteration_run_id||null};
    if(record?.generation_submitted)return {status:record.submitted_job_id?'submitted':'unknown',job_id:record.submitted_job_id||null};
    return null;
  }
  function validateIterationRequest(request){
    const conflicts=[];
    if(request.batch_size!==1)conflicts.push('数量必须为 1');
    if(request.highres_enabled)conflicts.push('高细节放大');
    if(request.face_detail_enabled)conflicts.push('附加脸部细化');
    if(request.hand_repair_enabled)conflicts.push('自动手部重绘');
    if(request.text_clean_enabled)conflicts.push('伪文字擦除');
    if(request.repair_source_file||request.image_handoff)conflicts.push('预先绑定的像素输入');
    if(conflicts.length)throw new Error(`按结果逐步创作当前不适用：${conflicts.join('、')}。请自行改为单张、标准输出并关闭附加细化后重新准备；没有自动降低质量或删除参数。`);
  }
  function outputConfig(config,{strategy,quality}){
    if(quality!=='iteration_base')return config;
    if(strategy!=='local_iteration')throw new Error('迭代基础输出只适用于“按结果逐步创作”，请明确选择相应生成方式或其他输出；没有自动换路。');
    // This visible, explicit output choice is consent to the base-image mode,
    // not a consequence of selecting the iteration strategy alone.
    return {...config,highres_enabled:false,face_detail_enabled:false,batch_size:1};
  }
  function iterationPayload(request,models){
    validateIterationRequest(request);
    // Model names come from the local installation catalogue. The aliases only
    // translate catalogue identifiers into the controller's existing protocol.
    const aliases={newbie:'newbie',anima_turbo:'anima',illustrious:'illustrious',z_image_turbo:'z_image_turbo'};
    if(!Array.isArray(models))throw new Error('本地模型目录尚未核验，不能启动迭代。');
    const allowed_models=[...new Set(models.filter(item=>item?.ready===true&&Object.hasOwn(aliases,item.id)).map(item=>aliases[item.id]))];
    if(!allowed_models.length)throw new Error('本地模型目录没有已就绪的可用模型，未启动迭代。');
    return {request:structuredClone(request),allowed_models,budget:{max_jobs:3,max_decisions:8,max_active_seconds:1800,max_rejected_candidates:2},autonomous_planning:true};
  }
  function frozenKey(value){
    const copy=structuredClone(value);
    if(copy.request&&copy.autonomous_planning===true)copy.request.output_label='';
    else copy.output_label='';
    return key(copy);
  }
  function create({read,storage,onRecord=()=>{},onConnection=()=>{},delay=ms=>new Promise(resolve=>setTimeout(resolve,ms)),uuid=()=>root.crypto.randomUUID()}){
    let sequence=0,current=null;
    const save=value=>{try{storage?.setItem(storageKey,JSON.stringify(value));}catch{}};
    const saved=()=>{try{return JSON.parse(storage?.getItem(storageKey)||'null');}catch{return null;}};
    function rememberRecord(record){
      const previous=saved(),pending=previous?.source_resume,body=pending?.body;
      const receipt=record.source_continuation_requests?.[body?.request_id];
      const acknowledged=pending?.unknown&&pending.id===record.id&&receipt
        &&receipt.request_id===body.request_id&&receipt.action===body.action
        &&receipt.expected_revision===body.expected_revision&&receipt.candidate_sha256===body.candidate_sha256
        &&(body.action!=='clarify'||receipt.clarification?.text===body.clarification);
      save({...previous,id:record.id,record,...(acknowledged?{source_resume:{...pending,unknown:false}}:{})});
    }
    async function watch(id,{token=++sequence}={}){
      let backoff=1500;
      while(token===sequence){
        let record;
        try{record=await read(`/api/creation-preparations/${encodeURIComponent(id)}`);}
        catch(error){
          if(token!==sequence)return null;
          onConnection(error);
          if(error.status===404)throw error;
          await delay(backoff);backoff=Math.min(10000,backoff*2);continue;
        }
        if(token!==sequence)return null;
        if(current?.id===record.id&&record.revision<current.revision)throw new Error('收到较旧的准备版本，未覆盖当前提案');
        current=record;rememberRecord(record);onConnection(null);onRecord(record);
        if(terminal.has(record.status))return record;
        await delay(1500);
      }
      return null;
    }
    return {saved,stop(){sequence++;},get current(){return current;},
      dispatch(record){const value=saved()?.meta?.dispatch;return value?.id===record?.id&&value?.revision===record?.revision?value:null;},
      rememberDispatch(record,value){
        const previous=saved()?.meta?.dispatch,same=previous?.id===record.id&&previous?.revision===record.revision;
        save({...saved(),meta:{...saved()?.meta,dispatch:{...(same?previous:{}),id:record.id,revision:record.revision,...value}}});
      },
      freezePayload(record,value){
        const previous=saved()?.meta?.dispatch;
        if(previous?.id===record.id&&previous?.revision===record.revision&&previous.payload){
          if(frozenKey(previous.payload)!==frozenKey(value))throw new Error('本版本已冻结另一份生成输入，请重建或重新准备，不能用旧回执提交新词。');
          return structuredClone(previous.payload);
        }
        const payload=structuredClone(value);
        this.rememberDispatch(record,{status:'prepared',payload});return payload;
      },
      async start(value,meta={}){
        const token=++sequence,snapshot=input(value),old=saved();
        // An explicit retry of an unknown response reuses its idempotency key.
        const requestId=old?.outcome_unknown&&sameInput(old.input,snapshot)?old.request_id:uuid();
        save({input:snapshot,request_id:requestId,meta,outcome_unknown:true});
        const record=await read('/api/creation-preparations',{method:'POST',body:JSON.stringify({...snapshot,request_id:requestId})});
        if(token!==sequence)return null;
        current=record;save({input:snapshot,request_id:requestId,meta,id:record.id,record,outcome_unknown:false});onRecord(record);
        return terminal.has(record.status)?record:watch(record.id,{token});
      },
      async restore(){const value=saved();return value?.id?watch(value.id):null;},
      async cancel(){
        const id=current?.id||saved()?.id;if(!id)return null;
        const record=await read(`/api/creation-preparations/${encodeURIComponent(id)}/cancel`,{method:'POST'});
        current=record;save({...saved(),record});onRecord(record);return record;
      },
      async refresh(){
        const id=current?.id||saved()?.id;if(!id)throw new Error('没有准备记录');
        const record=await read(`/api/creation-preparations/${encodeURIComponent(id)}`);
        current=record;rememberRecord(record);return record;
      },
      async resumeSource(body){
        const id=current?.id||saved()?.id;if(!id)throw new Error('没有准备记录');
        const previous=saved()?.source_resume;
        const comparable=value=>key(Object.fromEntries(Object.entries(value).filter(([k])=>k!=='request_id')));
        if(previous?.unknown&&previous.id===id&&comparable(previous.body)!==comparable(body))
          throw new Error('上次接续结果未确认，请先重新读取，不能改成另一份纠正重复发送。');
        const payload=previous?.unknown&&previous.id===id?structuredClone(previous.body):{...body,request_id:uuid()};
        const token=++sequence;
        save({...saved(),source_resume:{id,body:payload,unknown:true}});
        try{
          const record=await read(`/api/creation-preparations/${encodeURIComponent(id)}/source-review`,{method:'POST',body:JSON.stringify(payload)});
          if(token!==sequence)return null;
          current=record;save({...saved(),record,source_resume:{id,body:payload,unknown:false}});onRecord(record);
          return terminal.has(record.status)?record:watch(id,{token});
        }catch(error){
          if(error.status>=400&&error.status<500&&error.status!==408)
            save({...saved(),source_resume:{id,body:payload,unknown:false,rejected:true}});
          throw error;
        }
      },
      async rebuild(body){
        const id=current?.id||saved()?.id;if(!id)throw new Error('没有准备记录');
        const record=await read(`/api/creation-preparations/${encodeURIComponent(id)}/rebuild`,{method:'POST',body:JSON.stringify(body)});
        current=record;save({...saved(),record});onRecord(record);return record;
      },
    };
  }
  root.AuthorPreparation={storageKey,input,sameInput,key,proposal,proposalChanged,metadata,stage,describe,receipt,validateIterationRequest,outputConfig,iterationPayload,create};
  if(typeof module!=='undefined')module.exports=root.AuthorPreparation;
})(globalThis);
