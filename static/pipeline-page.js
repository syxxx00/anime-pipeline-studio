/* A read-only viewer: every network call here is GET. */
(function(){
  const $=id=>document.getElementById(id);
  let sources=[],revision=0;
  async function read(url){
    const response=await fetch(url,{signal:AbortSignal.timeout(20000)});
    if(!response.ok)throw new Error(`${url}：HTTP ${response.status}`);
    return response.json();
  }
  function show(id){
    const source=sources.find(item=>item.id===id);
    if(!source)return;
    $('sourceTitle').textContent=source.title;
    $('sourceNote').textContent=source.note;
    $('raw').textContent=JSON.stringify(source.value,null,2);
    PipelineDiagram.mount($('diagram'),source.value);
  }
  function options(preferred){
    $('sourceSelect').replaceChildren(...sources.map(item=>new Option(item.title,item.id)));
    $('sourceSelect').disabled=!sources.length;
    if(sources.some(item=>item.id===preferred))$('sourceSelect').value=preferred;
    show($('sourceSelect').value);
  }
  async function load(){
    const sequence=++revision,selected=$('sourceSelect').value;
    $('status').textContent='正在读取定义与记录…';
    const job=new URLSearchParams(location.search).get('job');
    const requests=[read('/api/v3/graph/default'),read('/api/pipelines')];
    if(job)requests.push(read(`/api/jobs/${encodeURIComponent(job)}/workflow`));
    const results=await Promise.allSettled(requests);
    if(sequence!==revision)return;
    const next=sources.filter(item=>item.local),errors=[];
    if(results[0].status==='fulfilled')next.push({id:'definition',title:'协作管线 · 定义示例',value:results[0].value,note:'通用契约生成的协作定义，包含可选分支与循环。不是当前一句话入口的实际执行记录；节点中的模型用途是现有配置描述，不是已验证的能力结论。'});
    if(results[1].status==='fulfilled'&&Array.isArray(results[1].value)){
      for(const item of results[1].value)next.push({id:`template:${item.id}`,title:`模板 · ${item.name}`,value:item,note:`${item.summary||''} 这是示例参数构建的底层工作流，不代表全部风格配方或实验组合。`});
    }
    if(results[2]?.status==='fulfilled'){
      const value=results[2].value;
      next.unshift({id:`job:${job}`,title:`作品任务 · ${job}`,value,note:PipelineDiagram.evidenceNote(value)});
    }
    results.forEach(result=>{if(result.status==='rejected')errors.push(result.reason.message);});
    sources=next;options(selected|| (job?`job:${job}`:'definition'));
    $('status').textContent=errors.length?`部分来源不可用：${errors.join('；')}。可重试，或导入本地记录。`:'';
  }
  $('sourceSelect').onchange=()=>show($('sourceSelect').value);
  $('reloadSources').onclick=load;
  $('importWorkflow').onchange=async event=>{
    const file=event.target.files[0];if(!file)return;
    ++revision; // A pending network load must not replace the user's local file.
    try{
      const value=JSON.parse(await file.text());
      if(!PipelineDiagram.normalize(value).nodes.length)throw new Error('未发现节点。支持本项目管线图或 Comfy API 格式；不支持 Comfy 编辑器格式。');
      const id=`local:${file.name}`;
      sources=sources.filter(item=>item.id!==id);
      sources.push({id,title:`本地记录 · ${file.name}`,value,local:true,note:'仅在浏览器内读取，文件不会上传。来源真实性及是否实际执行，以该文件记录为准。'});
      options(id);$('status').textContent='';
    }catch(error){$('status').textContent=`导入失败：${error.message}`;}
  };
  load();
})();
