/* Stage comparison is an evidence view, not an automatic ranking or winner. */
(function(){
  const dialog=document.createElement('dialog');dialog.id='stageImagesDialog';
  dialog.innerHTML='<button class="stage-close">关闭</button><h2>阶段图片对比</h2><p>比较源阶段、后续处理与最终交付。保留某张图不会修改实验状态或认定它是最优路线。</p><button id="refreshStages">刷新阶段</button><p id="stageImageStatus" role="status"></p><div id="stageImageGrid"></div>';
  document.body.append(dialog);
  let activeJob=null,revision=0;
  dialog.querySelector('.stage-close').onclick=()=>dialog.close();
  dialog.addEventListener('close',()=>revision++);
  const status=dialog.querySelector('#stageImageStatus'),grid=dialog.querySelector('#stageImageGrid');
  const evidenceNames={retained:'运行时保留',cache_match:'历史缓存匹配',retained_cache_match:'已保留的历史匹配',delivered:'最终交付'};
  const sessionButton=document.createElement('button');sessionButton.id='sessionStages';sessionButton.hidden=true;sessionButton.textContent='查看阶段图片';
  document.getElementById('sessionLive').append(sessionButton);
  const artButton=document.createElement('button');artButton.id='artworkStages';artButton.textContent='比较阶段图片';
  document.getElementById('evidenceTool').prepend(artButton);
  artButton.onclick=()=>open(state.artwork?.job_id);
  function attachJob(id){sessionButton.hidden=!id;sessionButton.onclick=()=>open(id);}
  function node(tag,text){const el=document.createElement(tag);el.textContent=text;return el;}
  function showArtwork(id){dialog.close();document.getElementById('artDialog').close();openArtwork(id);}
  async function open(id){
    if(!id)return;
    activeJob=id;const ticket=++revision;
    if(!dialog.open)dialog.showModal();
    grid.replaceChildren();status.textContent=`正在读取任务 ${id} 的阶段记录…`;
    try{
      const result=await api(`/api/jobs/${encodeURIComponent(id)}/stages`);
      if(ticket!==revision)return;
      status.textContent=`${id} · ${result.status} · ${result.images.length} 张阶段/交付图片。${(result.warnings||[]).join(' ')}`;
      if(!result.images.length)grid.append(node('p','尚无可读取的阶段图片。旧任务缺少文件时不会猜测或重新生图。'));
      for(const item of result.images){
        const card=node('article','');card.className='stage-image-card';
        const header=node('header','');header.className='stage-card-meta';
        header.append(node('h3',item.label),node('small',[item.model,evidenceNames[item.evidence]||item.evidence,item.id].filter(Boolean).join(' · ')));
        if(item.model_files?.length)header.append(node('small',item.model_files.join(' / ')));card.append(header);
        const link=document.createElement('a');link.href=item.image_url;link.target='_blank';link.rel='noopener';
        const img=document.createElement('img');img.src=item.image_url;img.alt=item.label;link.append(img);card.append(link);
        if(item.sampling?.length){const details=node('details','');details.append(node('summary','已保存的采样参数'));for(const sample of item.sampling)details.append(node('p',`节点 ${sample.node_id} · Seed ${sample.seed} · ${sample.steps} 步 · CFG ${sample.cfg} · 重绘 ${sample.denoise} · ${sample.sampler_name}/${sample.scheduler}`));card.append(details);}
        if(item.warning)card.append(node('p',item.warning));
        card.append(node('small',item.file_path||''));
        const actions=node('div','');actions.className='stage-actions';
        const download=node('a','下载原图');download.href=item.image_url+'?download=true';actions.append(download);
        const copy=node('button','复制文件路径');copy.onclick=async()=>{try{await navigator.clipboard.writeText(item.file_path);copy.textContent='已复制';}catch{status.textContent='无法复制，请选择图片下方的完整路径。';}};actions.append(copy);
        const keep=node('button',item.artwork_id?'查看作品':'保留此阶段为作品');
        keep.onclick=async()=>{
          if(item.artwork_id){showArtwork(item.artwork_id);return;}
          keep.disabled=true;
          try{
            const art=await api(`/api/jobs/${encodeURIComponent(id)}/stages/${encodeURIComponent(item.id)}/retain`,{method:'POST'});
            if(ticket!==revision)return;
            item.artwork_id=art.id;keep.textContent='查看已保留作品';
            status.textContent='已保留到创作作品，可写评价或局部修改；原实验和最终图不变。';
          }catch(error){if(ticket===revision)status.textContent=error.message;}
          finally{keep.disabled=false;}
        };
        actions.append(keep);card.append(actions);grid.append(card);
      }
    }catch(error){if(ticket===revision)status.textContent=`读取失败：${error.message}。可点击刷新重试。`;}
  }
  dialog.querySelector('#refreshStages').onclick=()=>open(activeJob);
  window.StageImages={open,attachJob};
})();
