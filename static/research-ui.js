const researchUI = {page: 0, experimentId: '', timer: null, sequence: 0, compare: new Map(), signature: ''};
const statusLabels = {pending:'待运行', queued:'排队中', running:'运行中', paused:'暂停', completed:'完成', failed:'失败', cancelled:'已跳过'};

function resultFilters() {
  return Object.fromEntries(['query','status','character','pipeline','scene','seed'].map(key => [key, $(`filter-${key}`).value]));
}
function populateResultFilters(items) {
  for (const key of ['character','pipeline','scene','seed']) {
    const select = $(`filter-${key}`), previous = select.value;
    const values = [...new Set(items.map(item => ResearchResults.axis(item,key)))].filter(Boolean).sort();
    const signature = JSON.stringify(values);
    if (select.dataset.signature === signature) continue;
    select.dataset.signature = signature;
    select.replaceChildren(new Option('全部',''), ...values.map(value => new Option(value,value)));
    if (values.includes(previous)) select.value = previous;
  }
}
function renderResultCards() {
  const experiment = state.experiment;
  if (!experiment) return;
  const result = ResearchResults.pageItems(experiment.items, resultFilters(), researchUI.page);
  researchUI.page = result.page;
  $('resultCount').textContent = `符合条件 ${result.total} / ${experiment.total} 项 · 第 ${result.page+1} / ${result.pages} 页`;
  $('previousResults').disabled = result.page === 0;
  $('nextResults').disabled = result.page + 1 >= result.pages;
  const markup = result.items.map(item => {
    const method = ResearchResults.axis(item, 'method');
    const images = (item.images || []).map((url,index) => {
      const id = `${item.job_id}-${String(index+1).padStart(2,'0')}`;
      return `<div><button class="sample-image" data-art="${esc(id)}"><img loading="lazy" src="${esc(url)}" alt="作品 ${esc(id)}"></button><label class="compare-toggle"><input type="checkbox" data-compare="${esc(id)}" data-image="${esc(url)}" data-item="${esc(item.id)}" ${researchUI.compare.has(id)?'checked':''}>对比 · ${esc(id)}</label></div>`;
    }).join('');
    return `<article class="result-card">${images || '<div class="sample-placeholder">'+esc(statusLabels[item.status]||item.status)+'</div>'}<div><strong>${esc(ResearchResults.axis(item,'character'))} · ${esc(ResearchResults.axis(item,'model'))}</strong><small>${esc(ResearchResults.axis(item,'scene'))} · Seed ${esc(ResearchResults.axis(item,'seed'))}</small>${method?`<p class="result-method">方法：${esc(method)}</p>`:''}<small>实验项 ${esc(item.id)} · ${esc(statusLabels[item.status]||item.status)}</small>${item.axes?.study?`<p>对照：${esc(item.axes.recipe)} · 配对 ${esc(item.axes.pair_id)}</p>${item.request?.parent_artwork_id?`<button data-art="${esc(item.request.parent_artwork_id)}">查看上一步 ${esc(item.request.parent_artwork_id)}</button>`:''}`:''}${item.error?`<p class="sample-error">${esc(item.error)}</p>`:''}${item.job_id?`<button data-stages="${esc(item.job_id)}">查看阶段图片</button>`:''}${item.status==='pending'?`<button data-skip="${esc(item.id)}">跳过此项</button>`:''}${item.axes?.study==='loops'&&item.pixel_source?`<p>第 ${esc(item.axes.round)} 轮 · 目标强度 ${esc(item.axes.denoise)}</p><button data-path="${esc(item.id)}">对比从源图到这一步</button>`:''}<details><summary>原始描述与实际配置</summary><p>${esc(item.request?.semantic_spec?.source_sentence || item.request?.prompt || '')}</p><pre>${esc(JSON.stringify(item.request,null,2))}</pre></details></div></article>`;
  }).join('') || '<p class="empty-copy">没有符合当前筛选条件的实验项。</p>';
  // Do not replace focused controls or expanded evidence on unchanged polling.
  if (markup === researchUI.signature) return;
  researchUI.signature = markup;
  $('resultGrid').innerHTML = markup;
  $('resultGrid').querySelectorAll('[data-art]').forEach(button => button.onclick=()=>openArtwork(button.dataset.art));
  $('resultGrid').querySelectorAll('[data-stages]').forEach(button => button.onclick=()=>StageImages.open(button.dataset.stages));
  $('resultGrid').querySelectorAll('[data-path]').forEach(button => button.onclick=()=>{
    try{
      const chain=ResearchResults.ancestry(experiment.items,button.dataset.path);
      if(chain.length>4||chain.some(i=>i.status!=='completed'||i.images.length!==1))throw new Error('此路径尚未全部完成；请先单独查看已有版本。');
      researchUI.compare.clear();
      for(const item of chain){const id=item.job_id+'-01';researchUI.compare.set(id,{id,image:item.images[0],item:item.id});}
      researchUI.signature='';renderResultCards();
      $('openComparison').textContent=`并排对比（${chain.length}/4）`;$('openComparison').disabled=chain.length<2;
      openComparison();
    }catch(error){$('researchConnection').textContent=error.message;}
  });
  $('resultGrid').querySelectorAll('[data-compare]').forEach(input => input.onchange=()=> {
    if (input.checked && researchUI.compare.size >= 4) {input.checked=false; return;}
    if (input.checked) researchUI.compare.set(input.dataset.compare,{id:input.dataset.compare,image:input.dataset.image,item:input.dataset.item});
    else researchUI.compare.delete(input.dataset.compare);
    researchUI.signature='';
    $('openComparison').textContent=`并排对比（${researchUI.compare.size}/4）`;
    $('openComparison').disabled=researchUI.compare.size<2;
  });
  $('resultGrid').querySelectorAll('[data-skip]').forEach(button => button.onclick=async()=> {
    button.disabled=true;
    try {await api(`/api/experiments/${experiment.id}/items/${button.dataset.skip}/cancel`,{method:'POST'}); await loadExperiment();}
    catch(error){$('researchConnection').textContent=error.message;button.disabled=false;}
  });
}

async function refreshSelectedExperiment() {
  clearTimeout(researchUI.timer);
  const id=$('experimentSelect').value;
  if (!id) return;
  const sequence=++researchUI.sequence;
  try {
    const experiment=await api(`/api/experiments/${id}`);
    if(sequence!==researchUI.sequence || $('experimentSelect').value!==id) return;
    if(researchUI.experimentId!==id) {
      researchUI.experimentId=id;researchUI.page=0;researchUI.compare.clear();researchUI.signature='';
      $('openComparison').disabled=true;$('openComparison').textContent='并排对比（0/4）';
      for(const key of ['query','status','character','pipeline','scene','seed']) $(`filter-${key}`).value='';
    }
    state.experiment=experiment;
    state.experiments=state.experiments.map(x=>x.id===id?{...experiment,items:undefined}:x);
    $('runName').textContent=experiment.name;
    $('runState').textContent=statusLabels[experiment.state]||experiment.state;
    $('runProgress').textContent=`${experiment.finished} / ${experiment.total} · 失败 ${experiment.counts.failed} · 跳过 ${experiment.counts.cancelled}`;
    $('runEta').textContent=duration(experiment.eta_seconds);
    $('runReviews').textContent='在分析页查看作品评价统计';
    $('runMeter').style.width=`${Math.round(experiment.progress*100)}%`;
    $('researchConnection').textContent=`更新于 ${new Date().toLocaleTimeString()} · ${experiment.active_item_id?'当前项 '+experiment.active_item_id:'无执行中项目'}`;
    for(const button of document.querySelectorAll('[data-action]')) {
      const allowed={resume:experiment.state==='paused',pause:experiment.state==='running','retry-failed':experiment.counts.failed>0,interrupt:!!experiment.active_item_id};
      button.disabled=!allowed[button.dataset.action];
    }
    populateResultFilters(experiment.items);renderResultCards();renderStudy();
  } catch(error) {
    if(sequence===researchUI.sequence) $('researchConnection').textContent=`读取失败，将自动重试：${error.message}`;
  } finally {
    if(sequence===researchUI.sequence && state.view==='research') researchUI.timer=setTimeout(refreshSelectedExperiment,document.hidden?15000:3000);
  }
}

async function openComparison() {
  const selected=[...researchUI.compare.values()];
  $('comparisonGrid').innerHTML=selected.map(value=>`<article><img src="${esc(value.image)}" alt="${esc(value.id)}"><strong>${esc(value.id)}</strong><button data-review="${esc(value.id)}">打开版本、评价与局部修改</button><textarea data-comment="${esc(value.id)}" placeholder="直接描述这张图的优点与问题"></textarea><button data-save="${esc(value.id)}">保存评价</button><p data-feedback="${esc(value.id)}"></p></article>`).join('');
  $('comparisonGrid').querySelectorAll('[data-review]').forEach(button=>button.onclick=()=>openArtwork(button.dataset.review));
  for (const [index, article] of [...$('comparisonGrid').children].entries()) {
    const item=state.experiment.items.find(item=>item.id===selected[index].item);
    const context=document.createElement('p');
    context.textContent=item ? ['character','model','pipeline','scene','seed'].map(key=>`${key}: ${ResearchResults.axis(item,key)}`).join(' · ')+(ResearchResults.axis(item,'method')?` · 方法 ${ResearchResults.axis(item,'method')}`:'')+(item.axes?.study?` · 对照 ${item.axes.recipe} · 配对 ${item.axes.pair_id}`:'') : '原始配置未知';
    article.querySelector('strong').after(context);
  }
  $('comparisonGrid').querySelectorAll('[data-save]').forEach(button=>button.onclick=async()=>{
    const article=button.closest('article'),text=article.querySelector('textarea').value.trim();
    if(!text)return;
    button.disabled=true;
    try{await api(`/api/artworks/${button.dataset.save}/reviews`,{method:'POST',body:JSON.stringify({text,image_index:0})});article.querySelector('p').textContent='评价已保存';article.querySelector('textarea').value='';}
    catch(error){article.querySelector('p').textContent=error.message;}
    finally{button.disabled=false;}
  });
  $('comparisonDialog').showModal();
}

document.querySelectorAll('.result-filters input,.result-filters select').forEach(input=>input.addEventListener(input.tagName==='INPUT'?'input':'change',()=>{researchUI.page=0;renderResultCards();}));
$('previousResults').onclick=()=>{researchUI.page--;renderResultCards();};
$('nextResults').onclick=()=>{researchUI.page++;renderResultCards();};
$('reloadResults').onclick=refreshSelectedExperiment;
$('openComparison').onclick=openComparison;
$('closeComparison').onclick=()=>$('comparisonDialog').close();
