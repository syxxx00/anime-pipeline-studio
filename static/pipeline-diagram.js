/* Read-only adapters and SVG rendering for saved graphs and Comfy API workflows. */
(function(root){
  let renderSequence=0;
  const escape=value=>String(value??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  function normalize(value){
    if(!value||typeof value!=='object')throw new Error('工作流必须是 JSON 对象。');
    const graph=value.graph||value;
    if(Array.isArray(graph.nodes)&&Array.isArray(graph.edges))return{
      nodes:graph.nodes.map(n=>({id:String(n.id),label:n.label||n.id,kind:n.type||'node',group:'graph',detail:n})),
      edges:graph.edges.map(e=>({source:String(e.source),target:String(e.target),label:e.condition||'',detail:e})),
      note:'条件连线来自管线定义；此图不表示所有分支都会执行。'
    };
    const workflow=value.workflow||value;
    const stages=workflow.stages||{workflow};
    const nodes=[],edges=[];
    for(const [stage,entries] of Object.entries(stages)){
      const known=new Set(Object.keys(entries||{}).filter(id=>entries[id]?.class_type));
      for(const id of known){
        const node=entries[id],nodeId=`${stage}/${id}`;
        nodes.push({id:nodeId,label:node._meta?.title||node.class_type,kind:node.class_type,group:stage,detail:node});
        for(const [port,input] of Object.entries(node.inputs||{})){
          if(Array.isArray(input)&&input.length===2&&known.has(String(input[0]))&&Number.isInteger(input[1]))edges.push({source:`${stage}/${input[0]}`,target:nodeId,label:`${port} ← ${input[1]}`,detail:{port,output:input[1]}});
        }
      }
    }
    return{nodes,edges,note:workflow.stages?'阶段内连线来自实际节点输入。阶段间通过图片等方式传递的数据，未作为 Comfy 端口连线记录。':'连线来自节点输入与输出端口。'};
  }

  function layout(graph){
    const ids=new Set(graph.nodes.map(n=>n.id));
    const edges=graph.edges.filter(e=>ids.has(e.source)&&ids.has(e.target));
    const adjacency=new Map(graph.nodes.map(n=>[n.id,[]]));
    edges.forEach((edge,index)=>adjacency.get(edge.source).push({edge,index}));
    const visiting=new Set(),visited=new Set(),back=new Set();
    function walk(id){
      visiting.add(id);
      for(const {edge,index} of adjacency.get(id)){
        if(visiting.has(edge.target))back.add(index);
        else if(!visited.has(edge.target))walk(edge.target);
      }
      visiting.delete(id);visited.add(id);
    }
    graph.nodes.forEach(n=>{if(!visited.has(n.id))walk(n.id);});
    const rank=new Map(graph.nodes.map(n=>[n.id,0]));
    // Back edges are drawn separately; they never force infinite rank growth.
    for(let pass=0;pass<graph.nodes.length;pass++){
      let changed=false;
      edges.forEach((e,i)=>{if(!back.has(i)&&rank.get(e.target)<=rank.get(e.source)){rank.set(e.target,rank.get(e.source)+1);changed=true;}});
      if(!changed)break;
    }
    const groups=[...new Set(graph.nodes.map(n=>n.group))],positions=new Map(),groupBoxes=[];
    const wrapped=Math.max(0,...rank.values())>4;
    let top=60,maxX=640;
    for(const group of groups){
      const columns=new Map(),members=graph.nodes.filter(n=>n.group===group);
      for(const n of members)columns.set(rank.get(n.id),(columns.get(rank.get(n.id))||0)+1);
      const bands=new Map(),offsets=new Map(),bandCount=wrapped?Math.floor(Math.max(0,...columns.keys())/4)+1:1;
      let bandTop=top+42;
      for(let band=0;band<bandCount;band++){
        const rows=Math.max(1,...[...columns].filter(([r])=>!wrapped||Math.floor(r/4)===band).map(([,count])=>count));
        bands.set(band,bandTop);bandTop+=rows*134+62;
      }
      for(const n of members){
        const r=rank.get(n.id),row=offsets.get(r)||0,band=wrapped?Math.floor(r/4):0,reverse=wrapped&&band%2===1;
        offsets.set(r,row+1);
        const column=wrapped?(reverse?3-r%4:r%4):r,x=60+column*290,y=bands.get(band)+row*134;
        positions.set(n.id,{...n,x,y,width:226,height:82,band,bandTop:bands.get(band),reverse});maxX=Math.max(maxX,x+280);
      }
      const height=bandTop-top;
      groupBoxes.push({label:group,x:30,y:top,width:0,height});top+=height+34;
    }
    groupBoxes.forEach(g=>g.width=maxX-40);
    return{nodes:[...positions.values()],edges:edges.map((e,i)=>({...e,index:i,back:back.has(i)})),positions,groups:groupBoxes,wrapped,width:maxX+30+(wrapped?back.size*20:0),height:top+30};
  }

  function svg(graph){
    const diagram=layout(graph),markerId=`diagram-arrow-${++renderSequence}`;let backLane=0;
    const links=diagram.edges.map(e=>{
      const a=diagram.positions.get(e.source),b=diagram.positions.get(e.target),x=a.x+a.width,y=a.y+41,tx=b.x,ty=b.y+41;
      let d;
      if(diagram.wrapped&&e.back){const lane=diagram.width-12-backLane++*20,ax=a.x+a.width/2,bx=b.x+b.width/2;d=`M ${ax} ${a.y} L ${ax} ${a.bandTop-20} L ${lane} ${a.bandTop-20} L ${lane} ${b.bandTop-28} L ${bx} ${b.bandTop-28} L ${bx} ${b.y}`;}
      else if(diagram.wrapped&&a.band!==b.band){const sx=a.reverse?a.x:a.x+a.width,side=sx+(a.reverse?-1:1)*(16+e.index%3*9),bx=b.x+b.width/2,mid=b.bandTop-32;d=`M ${sx} ${y} L ${side} ${y} L ${side} ${mid} L ${bx} ${mid} L ${bx} ${b.y}`;}
      else if(diagram.wrapped&&a.reverse){const sx=a.x,ex=b.x+b.width;d=`M ${sx} ${y} C ${sx-32} ${y},${ex+32} ${ty},${ex} ${ty}`;}
      else if(e.back||tx<=x){const lane=26+backLane++*20;d=`M ${x} ${y} C ${x+35} ${y},${x+35} ${lane},${x} ${lane} L ${tx-26} ${lane} Q ${tx-40} ${lane},${tx-40} ${lane+16} L ${tx-40} ${ty-16} Q ${tx-40} ${ty},${tx} ${ty}`;}
      else d=`M ${x} ${y} C ${x+32} ${y},${tx-32} ${ty},${tx} ${ty}`;
      return `<path class="diagram-edge ${e.back?'return-edge':''}" data-edge="${e.index}" d="${d}" marker-end="url(#${markerId})"><title>${escape(e.source+' → '+e.target+' · '+e.label)}</title></path>`;
    }).join('');
    const groups=diagram.groups.map(g=>`<rect class="diagram-group" x="${g.x}" y="${g.y}" width="${g.width}" height="${g.height}" rx="10"/><text class="diagram-group-label" x="${g.x+15}" y="${g.y+24}">${escape(g.label==='graph'?'协作定义':g.label)}</text>`).join('');
    const nodes=diagram.nodes.map((n,i)=>`<g class="diagram-node" data-node="${i}" role="button" tabindex="0" aria-label="${escape(n.label)}" transform="translate(${n.x} ${n.y})"><rect width="${n.width}" height="${n.height}" rx="8"/><circle cx="0" cy="41" r="4"/><circle cx="226" cy="41" r="4"/><text x="14" y="23" class="diagram-id">${escape(n.id.slice(-32))}</text><text x="14" y="47" class="diagram-label">${escape(n.label.length>27?n.label.slice(0,25)+'…':n.label)}</text><text x="14" y="66" class="diagram-kind">${escape(n.kind.slice(0,31))}</text><title>${escape(n.label)}</title></g>`).join('');
    return{diagram,markup:`<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="管线节点与连线图" viewBox="0 0 ${diagram.width} ${diagram.height}"><defs><marker id="${markerId}" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#8da6bf"/></marker></defs>${groups}${links}${nodes}</svg>`};
  }

  function mount(container,value){
    const graph=normalize(value),rendered=svg(graph),diagram=rendered.diagram;
    container.innerHTML=`<div class="diagram-toolbar"><button data-command="fit">适应画布</button><button data-command="in" aria-label="放大">＋</button><button data-command="out" aria-label="缩小">−</button><span>${graph.nodes.length} 节点 · ${graph.edges.length} 连线 · 拖动平移，滚轮缩放</span></div><p class="diagram-note">${escape(graph.note)}</p><div class="diagram-layout"><div class="diagram-viewport">${rendered.markup}</div><aside class="diagram-inspector"><h3>选择节点</h3><p>查看实际参数、输入输出和连接条件。</p></aside></div>`;
    const canvas=container.querySelector('svg'),viewport=container.querySelector('.diagram-viewport'),inspector=container.querySelector('.diagram-inspector');
    let box={x:0,y:0,w:diagram.width,h:diagram.height},drag=null;
    const update=()=>canvas.setAttribute('viewBox',`${box.x} ${box.y} ${box.w} ${box.h}`);
    function zoom(factor){const w=Math.max(160,Math.min(diagram.width*6,box.w*factor)),h=box.h*w/box.w;box={x:box.x+(box.w-w)/2,y:box.y+(box.h-h)/2,w,h};update();}
    container.querySelector('[data-command="fit"]').onclick=()=>{box={x:0,y:0,w:diagram.width,h:diagram.height};update();};
    container.querySelector('[data-command="in"]').onclick=()=>zoom(.8);
    container.querySelector('[data-command="out"]').onclick=()=>zoom(1.25);
    viewport.addEventListener('wheel',event=>{event.preventDefault();zoom(event.deltaY>0?1.12:.89);},{passive:false});
    canvas.addEventListener('pointerdown',event=>{if(event.target.closest('[data-node]'))return;const m=canvas.getScreenCTM().inverse();drag={x:event.clientX,y:event.clientY,box:{...box},scaleX:m.a,scaleY:m.d};canvas.setPointerCapture(event.pointerId);});
    canvas.addEventListener('pointermove',event=>{if(!drag)return;box.x=drag.box.x-(event.clientX-drag.x)*drag.scaleX;box.y=drag.box.y-(event.clientY-drag.y)*drag.scaleY;update();});
    canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);
    for(const element of canvas.querySelectorAll('[data-node]')){
      const select=()=>{
        const node=diagram.nodes[Number(element.dataset.node)];
        canvas.querySelectorAll('.diagram-node').forEach(el=>el.classList.toggle('selected',el===element));
        const related=diagram.edges.filter(e=>e.source===node.id||e.target===node.id);
        canvas.querySelectorAll('[data-edge]').forEach(el=>el.classList.toggle('selected',related.some(e=>String(e.index)===el.dataset.edge)));
        inspector.innerHTML=`<h3>${escape(node.label)}</h3><code>${escape(node.id)}</code><h4>连接</h4>${related.map(e=>`<p>${escape(e.source)} → ${escape(e.target)}<br><b>${escape(e.label||'always')}</b></p>`).join('')||'<p>无连接</p>'}<h4>参数与证据</h4><pre>${escape(JSON.stringify(node.detail,null,2))}</pre>`;
      };
      element.onclick=select;element.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}};
    }
  }
  function evidenceNote(value){
    if(value.source==='reconstructed')return '没有找到保存的执行工作流；下图按当前代码和原任务参数重建，不是历史执行证据。';
    if(value.source)return `读取已保存的工作流：${value.source}。执行是否完成及检查结果请结合任务状态判断。`;
    return `提交前预览，尚未执行。${value.route_reason||''}`;
  }
  function mountEvidence(container,value){
    mount(container,value);
    const note=document.createElement('p');note.className='diagram-note';note.textContent=evidenceNote(value);container.prepend(note);
    const details=document.createElement('details'),summary=document.createElement('summary'),raw=document.createElement('pre');
    summary.textContent='请求、检查与原始工作流';raw.textContent=JSON.stringify(value,null,2);details.append(summary,raw);container.append(details);
  }
  root.PipelineDiagram={normalize,layout,svg,mount,evidenceNote,mountEvidence};
  if(typeof module!=='undefined')module.exports=root.PipelineDiagram;
})(globalThis);
