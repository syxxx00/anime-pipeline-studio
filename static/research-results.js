/* Experiment results keep their original ordering and IDs across refreshes. */
(function (root) {
  const axis = (item, key) => String(item.axes?.[key] ?? item.request?.[key] ?? '');
  function pageItems(items, filters = {}, page = 0, pageSize = 24) {
    const query = (filters.query || '').toLowerCase();
    const filtered = items.filter(item => {
      if (filters.status && item.status !== filters.status) return false;
      if (['character', 'pipeline', 'scene', 'seed'].some(key => filters[key] && axis(item, key) !== filters[key])) return false;
      const searchable = [item.id, item.job_id, ...Object.values(item.axes || {}), ...(item.files || []), item.request?.semantic_spec?.source_sentence || item.request?.prompt || ''].join(' ').toLowerCase();
      return !query || searchable.includes(query);
    });
    const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
    page = Math.min(Math.max(0, page), pages - 1);
    return {items: filtered.slice(page * pageSize, (page + 1) * pageSize), total: filtered.length, pages, page};
  }
  function ancestry(items, id) {
    const keys=new Map(items.filter(i=>i.key).map(i=>[i.key,i])),seen=new Set(),chain=[];
    let current=items.find(i=>i.id===id);
    while(current){
      if(seen.has(current.id))throw new Error('路径存在循环依赖，不能展示错误来源');
      seen.add(current.id);chain.unshift(current);
      if(!current.pixel_source)break;
      current=keys.get(current.pixel_source.source_key);
      if(!current)throw new Error('路径缺少父版本');
    }
    return chain;
  }
  root.ResearchResults = {axis, pageItems, ancestry};
  if (typeof module !== 'undefined') module.exports = root.ResearchResults;
})(globalThis);
