/* UI bindings for one configuration model; experiment definitions are untouched. */
let creationCatalog=null,creationLoras=[],knowledgeRevision=0,editorsCompilation=null;
function numberControl(field){
  const s=field.schema,id=`setting-${field.key}`;
  if(s.type==='boolean')return `<label class="setting-check"><input id="${id}" data-setting="${field.key}" type="checkbox">${esc(field.label)}</label>`;
  return `<label>${esc(field.label)}<input id="${id}" data-setting="${field.key}" type="number" min="${s.minimum}" max="${s.maximum}" step="${field.step}"></label>`;
}
function professionalValues(){
  const values={render_pipeline:$('pipelineChoice').value,seed:$('seedChoice').value,sampler:$('samplerChoice').value,scheduler:$('schedulerChoice').value,lora_name:$('loraChoice').value||null};
  for(const field of document.querySelectorAll('[data-setting]'))values[field.dataset.setting]=field.type==='checkbox'?field.checked:field.value;
  values.composition_style_preset=$('sourceStyle').value==='inherit'?null:$('sourceStyle').value;
  values.character_style_preset=$('finalStyle').value==='inherit'?null:$('finalStyle').value;
  if(creationCatalog?.source_profiles?.[values.render_pipeline]&&$('sourceOverride').checked){
    values.source_parameters={...creationCatalog.source_profiles[values.render_pipeline]};
    for(const field of document.querySelectorAll('[data-source-setting]'))values.source_parameters[field.dataset.sourceSetting]=field.value;
  }
  return values;
}
function setValues(values){
  globalThis.CreativeSession?.clearSamplingParameterSources?.(Object.keys(values));
  for(const [key,value] of Object.entries(values)){
    const field=$(`setting-${key}`);if(!field)continue;
    if(field.type==='checkbox')field.checked=value;else field.value=value;
  }
  for(const [id,key] of [['pipelineChoice','render_pipeline'],['samplerChoice','sampler'],['schedulerChoice','scheduler'],['seedChoice','seed']])if(key in values)$(id).value=values[key]??'';
}
function updatePictureAuthoringControls(record=state.creativeSession){
  const section=$('creativePictureSettings'),note=$('creativePictureSettingsNote');
  if(section)section.hidden=$('generationStrategy').value!=='creative';
  const scope=$('creativeParameterScopeSettings');if(scope)scope.hidden=$('generationStrategy').value!=='creative';
  if(!note)return;
  const supported=creationCatalog?.capabilities?.creative_prose_picture===true;
  const current=record?.id?`正在查看任务 ${record.id}：${CreativeSession.pictureMethodLabel(CreativeSession.pictureAuthoringMethod(record))}${record.input_snapshot?.picture_authoring_method?'':'（旧记录未注明方法）'}。`:'';
  note.textContent=current+'此选择仅用于下一次新建创作，普通模式默认使用完整画面描述。'
    +(supported?'':' 当前后端尚未载入完整画面描述更新，请载入更新后刷新；专业模式可明确使用另两种画面组织方式，原路线与研究入口仍可使用。');
  const profileNote=$('creativeAuthoringProfileNote');
  if(profileNote)profileNote.textContent=(record?.id?`正在查看任务 ${record.id} 保存的构思配置：${CreativeSession.authoringProfileLabel(CreativeSession.authoringProfile(record))}${record.input_snapshot?.authoring_profile?'':'（旧记录未注明，按 4B 查看）'}。`:'')
    +'模型选择仅用于下一次新建创作，不改历史任务。普通模式保持 4B 默认；新任务的 7B 选择使用受限日常配置，旧 8K/mmap 配置不会转换，效果仍需作品验证。'
    +(creationCatalog?.capabilities?.creative_request_understanding===true?'':' 当前后端尚未声明独立原句理解能力，请载入更新后新建创作。');
}
function updateConfigContext(){
  if(!creationCatalog)return;
  updatePictureAuthoringControls();
  const creative=$('generationStrategy').value==='creative';
  const adaptiveChoice=creationCatalog.capabilities?.creative_initial_selection===true;
  for(const option of $('pipelineChoice').options){
    option.dataset.legacyLabel??=option.textContent;
    option.disabled=creative&&CreationConfig.bridges.has(option.value);
    option.textContent=creative&&option.value==='adaptive'
      ?(adaptiveChoice?'自动选择兼容起点':'自动选路 · 后端待更新'):option.dataset.legacyLabel;
  }
  const route=creationCatalog.routes.find(item=>item.id===$('pipelineChoice').value),newbie=route.profile.model_id==='newbie',bridge=CreationConfig.bridges.has(route.id)||route.id==='adaptive';
  $('bridgeSettings').classList.toggle('hidden',creative||!bridge);
  for(const option of $('samplerChoice').options)option.disabled=!route.samplers.includes(option.value);
  for(const option of $('schedulerChoice').options)option.disabled=!route.schedulers.includes(option.value);
  $('setting-steps').min=route.minimum_steps;
  const startingDescription=route.id==='adaptive'
    ?(adaptiveChoice?`自动选择兼容起点。当前参数以 ${route.profile.model_id} 为初始基准；明确覆盖保持，所声明的模型默认值随实际选择使用。`:'当前后端未声明自动选路能力，请载入更新后新建创作。')
    :`当前起点：${route.profile.model_id}，这里的参数用于这个起点。`;
  $('configSummary').textContent=creative
    ?`${CreationConfig.bridges.has(route.id)?'当前所选固定桥接不适用于创作任务，请明确选择一个起点。':startingDescription} 后继依据真实画面和评价决定，不预排固定模型链。固定路线及其源阶段参数在原路线／研究入口保留。高细节输出按所用模型放大后重采样，不因此限定起点；脸部细化仅 NewBie 支持，自动起点明确启用此项时会限制兼容起点。`
    :`${route.name}。${newbie?'采样参数用于 NewBie 阶段':'采样参数用于当前模型'}；高细节输出可按所用模型放大后重采样，脸部细化仅 NewBie 支持。${bridge?' 协作一次一张；明确选择路线后可分别设置源阶段参数。':''} 切换路线会重置采样参数，不会修改描述和画幅。`;
  updateSourceControls(route.id);
  for(const group of creationCatalog.groups)for(const field of group.fields){
    const control=$(`setting-${field.key}`);
    const faceOnly=field.key.startsWith('face_detail_');
    // An old checked face option remains reachable so the user can explicitly
    // cancel it; resolve refuses it for another model rather than clearing it.
    control.disabled=(faceOnly&&!newbie&&!(field.key==='face_detail_enabled'&&control.checked))||(field.key==='hybrid_denoise'&&(creative||!bridge))||(field.key==='cfg_rescale'&&!newbie)||(field.key==='shift'&&route.id==='illustrious')||(field.key==='batch_size'&&bridge);
    if(faceOnly)control.title=newbie?'脸部细化仅 NewBie 支持；与所有模型可用的输出放大独立。':'当前模型不支持 NewBie 脸部细化；已有勾选需明确取消，不会静默关闭。';
  }
  if(bridge)$('setting-batch_size').value='1';
  const selected=$('loraChoice').value;
  $('loraChoice').replaceChildren(new Option('不加载 LoRA',''));
  for(const item of creationLoras){const option=new Option(`${item.name} · ${item.family}`,item.name);option.disabled=!item.compatible_models?.includes(route.profile.model_id);$('loraChoice').append(option);}
  const stillCompatible=creationLoras.some(item=>item.name===selected&&item.compatible_models?.includes(route.profile.model_id));
  $('loraChoice').value=stillCompatible?selected:'';
  updateLoraNote();
  globalThis.CooperativeUI?.updateSettings();
}
function updateLoraNote(){const item=creationLoras.find(item=>item.name===$('loraChoice').value);$('loraNote').textContent=item?.evidence||'只允许选择元数据声明兼容的基础架构；灰色项目不兼容或尚未确认。角色和外观仍需验证。';}
function resetCreationSettings(){if(!creationCatalog)return;setValues(creationCatalog.defaults);$('sourceStyle').value='inherit';$('finalStyle').value='inherit';$('loraChoice').value='';$('generationStrategy').value='creative';if($('creative-picture-authoring-method'))$('creative-picture-authoring-method').value='prose';if($('creative-authoring-profile'))$('creative-authoring-profile').value='local_4b';if($('creativeParameterScope'))$('creativeParameterScope').value='initial';$('cooperativeSettingsNote').classList.add('hidden');sourceControlsRoute=null;updateConfigContext();}
let sourceControlsRoute=null;
function updateSourceControls(routeId){
  let panel=$('sourceSamplingSettings');
  if(!panel){panel=document.createElement('details');panel.id='sourceSamplingSettings';panel.className='setting-section';$('bridgeSettings').prepend(panel);}
  const profile=creationCatalog.source_profiles?.[routeId];panel.classList.toggle('hidden',!profile);
  if(!profile){sourceControlsRoute=null;return;}
  if(sourceControlsRoute===routeId)return;sourceControlsRoute=routeId;
  const route=creationCatalog.routes.find(item=>item.profile.model_id===profile.model_id),schema=creationCatalog.source_parameter_schema;
  const numeric=(key,label)=>`<label>${label}<input id="source-${key}" data-source-setting="${key}" type="number" min="${schema[key].minimum}" max="${schema[key].maximum}" step="${schema[key].type==='integer'?1:.1}" value="${profile[key]??schema[key].default}"></label>`;
  panel.innerHTML=`<summary>源阶段参数 · ${esc(route.name)}</summary><label class="setting-check"><input id="sourceOverride" type="checkbox">自行设置源阶段</label><p>不勾选时沿用路线预设。源阶段与后续阶段独立；留空 Seed 则使用本次任务 Seed。切换路线会重置本组。</p><div class="settings-grid">${numeric('steps','采样步数')}${numeric('cfg','提示词引导 CFG')}${numeric('shift','采样偏移 Shift')}<label>Seed（留空继承）<input id="source-seed" data-source-setting="seed" type="number" min="0" max="9007199254740991" step="1"></label><label>采样器<select id="source-sampler" data-source-setting="sampler">${route.samplers.map(value=>`<option>${value}</option>`).join('')}</select></label><label>调度器<select id="source-scheduler" data-source-setting="scheduler">${route.schedulers.map(value=>`<option>${value}</option>`).join('')}</select></label><label>源阶段 LoRA<select id="source-lora_name" data-source-setting="lora_name"><option value="">不加载 LoRA</option></select></label>${numeric('lora_strength','源阶段 LoRA 强度')}</div><p id="sourceLoraNote"></p>`;
  $('source-steps').min=route.minimum_steps;$('source-sampler').value=profile.sampler;$('source-scheduler').value=profile.scheduler;
  for(const item of creationLoras){const option=new Option(`${item.name} · ${item.family}`,item.name);option.disabled=!item.compatible_models?.includes(profile.model_id);$('source-lora_name').append(option);}
  const toggle=()=>{for(const control of panel.querySelectorAll('[data-source-setting]'))control.disabled=!$('sourceOverride').checked||(control.dataset.sourceSetting==='shift'&&profile.model_id==='illustrious')||(['lora_name','lora_strength'].includes(control.dataset.sourceSetting)&&profile.model_id!=='illustrious');};
  $('sourceLoraNote').textContent=profile.model_id==='illustrious'?'仅显示有架构兼容证据的资源；这不代表外观准确性已经验证。':'当前 Turbo 源阶段尚未实现 LoRA 加载，此项不可用。';
  $('sourceOverride').onchange=toggle;toggle();
}
function syncCreationEntryMode(){
  const field=$('generationStrategy'),professional=$('proMode').checked;
  const changed=typeof state.creationModeProfessional==='boolean'&&state.creationModeProfessional!==professional;
  if(professional){
    if(changed&&state.professionalGenerationStrategy)field.value=state.professionalGenerationStrategy;
    if(!field.value)field.value='creative';
    state.professionalGenerationStrategy=field.value;
  }else{
    if(changed)state.professionalGenerationStrategy=field.value||'creative';
    if(changed||!field.value||field.value==='single'||field.value==='cooperative')field.value='creative';
  }
  state.creationModeProfessional=professional;
  for(const value of ['single','cooperative']){
    const option=field.querySelector?.(`option[value="${value}"]`);if(option)option.disabled=!professional;
  }
}
function professionalModeChanged(){
  syncCreationEntryMode();
  const enabled=$('proMode').checked;
  $('professionalPanel').classList.toggle('hidden',!enabled);$('workbench').classList.toggle('is-professional',enabled);
  $('openPromptReview').classList.toggle('hidden',!enabled);$('openPromptReview').disabled=$('generationStrategy').value==='creative'?!CreativeSession.reviewState(state.creativeSession).canReview:!state.compiled;
  $('make').innerHTML=enabled?'解析并审阅 <span>↗</span>':'生成画面 <span>↗</span>';
  $('modeSummary').textContent=enabled?'自行选择资源与参数，生成前可以审阅和修改提示词。':'从一句描述开始，参数交给默认配置。';
  if(!enabled&&$('promptDialog').open)$('promptDialog').close();
  if(enabled&&state.compiled&&editorsCompilation!==state.compiled)populateCompiledEditors();
  updateConfigContext();
}
function openPromptReview(){
  if(!state.compiled||!$('proMode').checked)return;
  if(editorsCompilation!==state.compiled)populateCompiledEditors();
  $('contractPanel').classList.remove('hidden');$('promptMessage').classList.add('hidden');
  if(!$('promptDialog').open)$('promptDialog').showModal();
}
function selectPromptTab(name){
  document.querySelectorAll('[data-prompt-tab]').forEach(button=>{const active=button.dataset.promptTab===name;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));});
  document.querySelectorAll('.prompt-pane').forEach(pane=>pane.classList.toggle('active',pane.id===`${name}PromptPane`));
}
function renderParameterFields(catalog){
  const fields=new Map(catalog.groups.flatMap(group=>group.fields).map(field=>[field.key,field]));
  const take=keys=>keys.map(key=>{const field=fields.get(key);if(!field)return '';fields.delete(key);return numberControl(field);}).join('');
  $('samplingFields').innerHTML=`<div class="settings-grid">${take(['steps','cfg','shift','cfg_rescale'])}</div><div id="creativeParameterScopeSettings"><label>采样参数作用范围<select id="creativeParameterScope"><option value="initial" selected>仅起点（默认）</option><option value="all">整个创作</option></select></label><p>仅用于下一次新建创作：仅起点允许后继使用目标模型默认；整个创作保留你的明确采样值，不改历史任务。</p></div>`;
  $('loraStrengthSlot').innerHTML=take(['lora_strength']);
  const groups=[['输出尺寸与数量',['width','height','batch_size']],['输出细化（所有模型）',['highres_scale','highres_steps','highres_denoise']],['NewBie 脸部细化',['face_detail_enabled','face_detail_steps','face_detail_denoise']],['协作交接',['hybrid_denoise']],['生成后检查',['check_enabled']]];
  $('parameterGroups').innerHTML=groups.map(([name,keys])=>`<details class="setting-section"><summary>${name}</summary><div class="settings-grid">${take(keys)}</div></details>`).join('');
  if(fields.size)$('parameterGroups').insertAdjacentHTML('beforeend',`<details class="setting-section"><summary>其他设置</summary><div class="settings-grid">${[...fields.values()].map(numberControl).join('')}</div></details>`);
}
function populateCompiledEditors(){
  if(!state.compiled)return;
  editorsCompilation=state.compiled;
  $('contractJson').value=JSON.stringify(state.compiled.spec,null,2);$('contractXml').value=state.compiled.xml;$('contractNegative').value=state.compiled.negative_prompt;
  $('nativeEditors').innerHTML=Object.entries(state.compiled.native_prompts||{}).filter(([target])=>target!=='newbie').map(([target,prompt])=>`<details><summary>${esc(target)} 原生提示词</summary><label>正向<textarea data-native="${esc(target)}" data-channel="positive">${esc(prompt.positive||'')}</textarea></label><label>排除项<textarea data-native="${esc(target)}" data-channel="negative">${esc(prompt.negative||'')}</textarea></label></details>`).join('');
  const review=state.compiled.professional_review;
  $('professionalQuestions').innerHTML=(review?.questions||[]).map(q=>`<article><strong>${esc(q.question)}</strong><p>${esc(q.reason)}</p><p>${esc(q.action)}</p></article>`).join('')+(review?.assumptions||[]).map(text=>`<p>默认补充：${esc(text)}</p>`).join('');
  $('professionalQuestions').innerHTML+=(review?.compiler_issues||[]).map(q=>`<article><strong>${esc(q.title)}</strong><p>${esc(q.reason)}</p><p>${esc(q.action)}</p></article>`).join('');
  const removed=state.compiled.negative_policy?.removed_automatic_terms||[];
  $('compileWarnings').textContent=[...(state.compiled.warnings||[]),...(removed.length?[`已撤回与当前语义冲突的自动负面词：${removed.map(item=>item.token).join('、')}。你的排除项未被删除；这不是画面质量验证。`]:[])].join('\n');
  $('openPromptReview').disabled=false;
  $('promptDialog').querySelector('.review-notes').open=Boolean(review?.blocking||review?.compiler_issues?.length);
}
function readCompiledEditor(){
  const native={};for(const field of document.querySelectorAll('[data-native]')){native[field.dataset.native]??={};native[field.dataset.native][field.dataset.channel]=field.value;}
  return {spec:$('contractJson').value,xml:$('contractXml').value,negative:$('contractNegative').value,native};
}
function selectedKnowledge(){
  if(!$('proMode').checked||!$('knowledgePack').value)return {};
  if(state.knowledge?.id!==$('knowledgePack').value)throw new Error('知识库尚未加载成功，请重新选择。');
  if(!$('operator').value||!$('appearance').value)throw new Error('请选择知识库中的角色和外观。');
  return {knowledge_pack:$('knowledgePack').value,knowledge_operator:$('operator').value,knowledge_appearance:$('appearance').value};
}
async function selectKnowledgePack(){
  const revision=++knowledgeRevision,pack=$('knowledgePack').value;state.knowledge=null;
  $('knowledgePicker').classList.toggle('hidden',!pack);$('operator').replaceChildren();$('appearance').replaceChildren();$('knowledgeSource').removeAttribute('href');
  $('knowledgeNote').textContent=pack?'正在读取角色与外观…':'不使用知识库，仍可创作任意角色。';
  if(!pack)return;
  try{const value=await api(`/api/knowledge-packs/${encodeURIComponent(pack)}`);if(revision!==knowledgeRevision)return;state.knowledge=value;populateKnowledge();}
  catch(error){if(revision===knowledgeRevision)$('knowledgeNote').textContent=`知识库读取失败：${error.message}`;}
}
function populateKnowledge(){$('operator').replaceChildren(...(state.knowledge?.operators||[]).map(op=>new Option(op.name,op.id)));populateAppearances();}
function populateAppearances(){const op=(state.knowledge?.operators||[]).find(op=>op.id===$('operator').value);$('appearance').replaceChildren(...(op?.appearances||[]).map(ap=>new Option(`${ap.name}${ap.knowledge_status==='injectable'?' · 可注入事实':' · 仅参考'}`,ap.id)));updateKnowledgeNote();}
function updateKnowledgeNote(){
  const op=(state.knowledge?.operators||[]).find(op=>op.id===$('operator').value),ap=(op?.appearances||[]).find(ap=>ap.id===$('appearance').value);
  $('knowledgeNote').textContent=ap?`${op.name} / ${ap.name}。${ap.knowledge_status==='injectable'?'注入带核验记录的视觉事实，不代表模型能准确画出。':'只有参考索引，没有足够的外观事实，不能宣称锁定皮肤。'}`:'请选择角色与外观。';
  const source=ap?.source_url||op?.source_url;
  if(source&&/^https?:\/\//i.test(source))$('knowledgeSource').href=source;else $('knowledgeSource').removeAttribute('href');
}
async function initCreationControls(){
  $('proMode').onchange=professionalModeChanged;$('knowledgePack').onchange=selectKnowledgePack;$('operator').onchange=populateAppearances;$('appearance').onchange=updateKnowledgeNote;
  $('resetSettings').onclick=resetCreationSettings;$('loraChoice').onchange=updateLoraNote;
  if($('creative-picture-authoring-method'))$('creative-picture-authoring-method').onchange=()=>updatePictureAuthoringControls();
  if($('creative-authoring-profile'))$('creative-authoring-profile').onchange=()=>updatePictureAuthoringControls();
  const [catalog,packs,loras]=await Promise.allSettled([api('/api/creation-options'),api('/api/knowledge-packs'),api('/api/loras')]);
  if(catalog.status!=='fulfilled'){$('configSummary').textContent='无法读取创作配置，请确认服务代码已更新后刷新。';throw new Error('创作配置加载失败，未采用猜测的参数。');}
  creationCatalog=catalog.value;creationLoras=loras.status==='fulfilled'?loras.value:[];
  $('pipelineChoice').replaceChildren(...creationCatalog.routes.map(route=>new Option(route.name,route.id)));
  $('samplerChoice').replaceChildren(...creationCatalog.samplers.map(value=>new Option(value,value)));$('schedulerChoice').replaceChildren(...creationCatalog.schedulers.map(value=>new Option(value,value)));
  renderParameterFields(creationCatalog);
  for(const id of ['sourceStyle','finalStyle'])$(id).innerHTML='<option value="inherit">继承整体风格配方</option><option value="">不追加配方</option>'+state.styles.map(item=>`<option value="${esc(item.id)}">${esc(item.name)}</option>`).join('');
  $('knowledgePack').replaceChildren(new Option('不使用知识库',''),...(packs.status==='fulfilled'?packs.value:[]).map(pack=>new Option(pack.name,pack.id)));
  if(packs.status!=='fulfilled')$('knowledgeNote').textContent='知识库列表读取失败；可不使用知识库继续。';
  $('pipelineChoice').onchange=()=>{const route=creationCatalog.routes.find(item=>item.id===$('pipelineChoice').value);setValues(route.profile);updateConfigContext();};
  resetCreationSettings();
  professionalModeChanged();
}
