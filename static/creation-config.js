/* Canonical ordinary defaults + professional overrides. No DOM or requests. */
(function(root){
  const bridges=new Set(['anima_newbie','illustrious_newbie','z_newbie']);
  function resolve(catalog,{professional=false,values={},quality='high',style='',seed=42}={}){
    if(!catalog?.defaults)throw new Error('创作配置尚未加载，请重试连接。');
    const routeId=professional?(values.render_pipeline||catalog.defaults.render_pipeline):catalog.defaults.render_pipeline;
    const route=catalog.routes.find(item=>item.id===routeId);
    if(!route)throw new Error('所选路线不在当前配置中。');
    const config={...catalog.defaults,...route.profile,...(professional?values:{}),render_pipeline:routeId,model_id:route.profile.model_id};
    config.seed=config.seed===null||config.seed===''?seed:Number(config.seed);
    if(!Number.isSafeInteger(config.seed)||config.seed<0)throw new Error('Seed 请填写非负安全整数，或留空随机。');
    for(const group of catalog.groups)for(const field of group.fields){
      if(field.schema.type==='boolean'){config[field.key]=Boolean(config[field.key]);continue;}
      const number=Number(config[field.key]),{minimum,maximum,type}=field.schema;
      if(!Number.isFinite(number)||config[field.key]===''||number<minimum||number>maximum||(type==='integer'&&!Number.isInteger(number)))throw new Error(`${field.label}超出允许范围。`);
      config[field.key]=number;
    }
    if(config.width%16||config.height%16)throw new Error('宽度与高度必须是 16 的倍数。');
    if(!route.samplers.includes(config.sampler)||!route.schedulers.includes(config.scheduler)||config.steps<route.minimum_steps)throw new Error(`采样参数不满足当前 ${route.name} 路线的支持范围。`);
    if((bridges.has(routeId)||routeId==='adaptive')&&config.batch_size!==1)throw new Error('当前协作和自动路线一次仅支持一张图，请将数量设为 1。');
    const newbie=config.model_id==='newbie';
    config.highres_enabled=quality==='high';
    if(!newbie&&config.face_detail_enabled)throw new Error('脸部细化目前仅支持 NewBie；请明确取消此项，或选择 NewBie，不会静默关闭。');
    config.style_preset=style;
    config.hand_repair_enabled=false;config.text_clean_enabled=false;
    config.style_category='auto';config.use_default_negative_prompt=false;
    config.parameter_policy='explicit';
    config.source_parameters=null;
    if(professional&&values.source_parameters){
      const profile=catalog.source_profiles?.[routeId],source={...values.source_parameters};
      if(!profile||source.model_id!==profile.model_id)throw new Error('源阶段参数与当前协作路线不一致。');
      const sourceRoute=catalog.routes.find(item=>item.profile.model_id===profile.model_id);
      for(const key of ['steps','cfg','shift','lora_strength']){
        const schema=catalog.source_parameter_schema[key],value=source[key]??schema.default,number=Number(value);
        if(value===''||!Number.isFinite(number)||number<schema.minimum||number>schema.maximum||(schema.type==='integer'&&!Number.isInteger(number)))throw new Error(`源阶段 ${key} 超出允许范围。`);
        source[key]=number;
      }
      source.seed=source.seed===''||source.seed==null?null:Number(source.seed);
      if(source.seed!==null&&(!Number.isSafeInteger(source.seed)||source.seed<0))throw new Error('源阶段 Seed 无效。');
      if(source.steps<sourceRoute.minimum_steps||!sourceRoute.samplers.includes(source.sampler)||!sourceRoute.schedulers.includes(source.scheduler))throw new Error('源阶段采样参数不在当前模型支持范围。');
      source.lora_name=source.lora_name||null;
      if(source.lora_name&&source.model_id!=='illustrious')throw new Error('当前 Turbo 源阶段尚未启用 LoRA。');
      config.source_parameters=source;
    }
    return config;
  }
  function conditioning(config,resolved,data){
    const target={'anima':'anima','illustrious':'illustrious','z_image_turbo':'z_image'}[config.render_pipeline]||'newbie';
    const sourceTarget={'anima_newbie':'anima','illustrious_newbie':'illustrious','z_newbie':'z_image'}[config.render_pipeline];
    const selected=resolved.native[target];
    if(!selected?.positive?.trim())throw new Error(`缺少 ${target} 的原生提示词，请重新编译。`);
    const source=sourceTarget?resolved.native[sourceTarget]:null;
    if(sourceTarget&&!source?.positive?.trim())throw new Error(`缺少源阶段 ${sourceTarget} 的提示词。`);
    return {prompt:selected.positive,negative_prompt:selected.negative??'',native_prompts:resolved.native,
      anima_prompt:source?.positive||data.anima_prompt||resolved.spec.source_sentence,
      source_prompt:source?.positive||data.anima_prompt||resolved.spec.source_sentence,
      source_negative_prompt:source?source.negative??'':null};
  }
  function validateStyles(config,styles){
    const checks=bridges.has(config.render_pipeline)||config.render_pipeline==='adaptive'
      ? [[config.character_style_preset??config.style_preset,'newbie'],[config.composition_style_preset??config.style_preset,config.render_pipeline==='anima_newbie'?'anima':'tags']]
      : [[config.style_preset,config.render_pipeline==='newbie'?'newbie':config.render_pipeline==='anima'?'anima':'tags']];
    for(const [id,target] of checks){
      if(!id)continue;const style=styles.find(item=>item.id===id);
      if(!style)throw new Error(`找不到所选风格配方：${id}`);
      if(target!=='tags'&&!['both',target].includes(style.mode))throw new Error(`风格配方“${style.name}”没有 ${target} 格式，请选择对应模型格式的配方。`);
    }
  }
  root.CreationConfig={resolve,conditioning,validateStyles,bridges};
  if(typeof module!=='undefined')module.exports=root.CreationConfig;
})(globalThis);
