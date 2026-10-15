'use strict';
(() => {
  const $ = id => document.getElementById(id);
  let revision = null;
  let busy = false;
  const controls = new Map();
  const endpoint = '/api/model-connections';
  const descriptions = {
    image_service: '连接已有 ComfyUI Core 时关闭自动启动。Core 不需要桌面端；其节点和模型必须与所选适配器匹配。',
    authoring: '本地 GGUF 由本项目管理运行器；兼容服务由你自行启动，模型须提供结构化 JSON 与图像输入能力。',
    newbie: '使用带 tokenizer 的 Comfy 格式编码器。扩散权重与 VAE 即使同名，也分别放在各自分类目录。',
    anima: '此适配器使用 Anima Turbo，扩散权重、编码器和 VAE 需要配套。',
    illustrious: '配置兼容 Illustrious／SDXL 的 checkpoint；改变文件名不等于支持任意模型架构。',
    z_image_turbo: '使用与 Z-Image Turbo 配套的编码器和 VAE。',
    resources: '默认使用项目附带的概念资源。风格配方可选；作品与用户评价保存在本地。'
  };
  function message(text, error = false) {
    $('status').textContent = text;
    $('status').classList.toggle('error', error);
  }
  function lock(value) {
    busy = value;
    for (const id of ['save', 'validate', 'probe', 'reload', 'bind']) $(id).disabled = value || (!revision && id !== 'reload');
  }
  async function request(path = '', method = 'GET', payload) {
    const response = await fetch(endpoint + path, {method, headers: {'Content-Type': 'application/json'},
      body: payload === undefined ? undefined : JSON.stringify(payload)});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '配置请求失败，请检查输入。');
    return data;
  }
  function values() {
    const result = {};
    for (const [key, input] of controls) {
      result[key] = input.type === 'checkbox' ? input.checked : input.type === 'number' ? Number(input.value) : input.value;
    }
    return result;
  }
  function backendVisibility() {
    const external = controls.get('authoring_backend')?.value === 'openai_compatible';
    for (const [key, input] of controls) {
      if (key.startsWith('local_llm_')) input.closest('label').hidden = external;
      else if (key.startsWith('authoring_') && key !== 'authoring_backend') input.closest('label').hidden = !external;
    }
    $('local-registration').hidden = external;
  }
  function render(data) {
    revision = data.revision;
    controls.clear(); $('fields').replaceChildren(); $('sections').replaceChildren(); $('checks').replaceChildren();
    for (const group of data.groups) {
      const link = document.createElement('a'); link.href = '#group-' + group.id; link.textContent = group.title; $('sections').append(link);
      const section = document.createElement('section'); section.className = 'group'; section.id = 'group-' + group.id;
      const heading = document.createElement('h2'); heading.textContent = group.title; section.append(heading);
      const explanation = document.createElement('p'); explanation.className = 'group-description'; explanation.textContent = descriptions[group.id] || ''; section.append(explanation);
      const grid = document.createElement('div'); grid.className = 'grid'; section.append(grid);
      for (const field of group.fields) {
        const label = document.createElement('label'); label.className = 'field';
        const name = document.createElement('span'); name.textContent = field.label; label.append(name);
        const input = document.createElement(field.kind === 'backend' ? 'select' : 'input');
        input.name = field.key; input.id = 'setting-' + field.key;
        if (field.kind === 'backend') {
          for (const [value, text] of [['local_gguf', '本地 GGUF · llama-server'], ['openai_compatible', 'OpenAI 兼容服务 · 自行提供']]) {
            const option = document.createElement('option'); option.value = value; option.textContent = text; input.append(option);
          }
        } else if (field.kind === 'boolean') input.type = 'checkbox';
        else if (field.kind === 'integer') { input.type = 'number'; input.min = '1'; input.max = '65535'; input.required = true; }
        else { input.type = 'text'; input.spellcheck = false; input.maxLength = 4096; }
        if (input.type === 'checkbox') input.checked = Boolean(data.values[field.key]);
        else input.value = data.values[field.key] ?? '';
        if (['url', 'path', 'model_file', 'environment'].includes(field.kind)) input.required = true;
        const code = document.createElement('code'); code.textContent = field.key;
        label.append(input, code); grid.append(label); controls.set(field.key, input);
      }
      $('fields').append(section);
    }
    controls.get('authoring_backend').addEventListener('change', backendVisibility);
    backendVisibility();
    message(data.restart_required ? '已保存的配置与运行中进程不同。请停止后重新启动项目。' : '已读取配置。可先检查文件，再按需探测已有服务；这些操作不会加载模型。');
  }
  async function perform(action) {
    if (busy) return;
    lock(true);
    try { await action(); } catch (error) { message(error.message, true); }
    finally { lock(false); }
  }
  async function load() { render(await request()); }
  $('reload').addEventListener('click', () => perform(load));
  $('connections').addEventListener('submit', event => {
    event.preventDefault();
    perform(async () => {
      const result = await request('', 'PUT', {values: values(), expected_revision: revision});
      revision = result.revision; message(result.message); $('checks').replaceChildren();
    });
  });
  $('validate').addEventListener('click', () => perform(async () => {
    message('检查文件与配置中，不加载模型…'); $('checks').replaceChildren();
    const result = await request('/validate', 'POST', {values: values()});
    message(result.configuration_ready ? '所需配置与本地文件已就绪。实际推理需在创作台发起。' : '仍有需要准备的配置或本地文件，见下方明细。');
    for (const item of result.checks) {
      const li = document.createElement('li'); li.className = item.ok ? 'ok' : 'missing';
      li.textContent = (item.ok ? '✓ ' : item.required ? '待准备 · ' : '可选 · ') + item.message; $('checks').append(li);
    }
  }));
  $('probe').addEventListener('click', () => perform(async () => {
    message('仅请求指定服务的节点／模型目录，不进行推理…'); $('checks').replaceChildren();
    const result = await request('/probe', 'POST', {values: values()});
    for (const service of result.services) {
      const li = document.createElement('li'); li.className = service.reachable ? 'ok' : 'missing';
      li.textContent = service.service + ' · ' + service.message + (service.model_ids?.length ? ' 模型：' + service.model_ids.join('、') : ''); $('checks').append(li);
    }
    message('服务探测完成；尚未进行文字、视觉或图像推理。');
  }));
  $('bind').addEventListener('click', () => perform(async () => {
    message('检查原生语法解析并登记当前运行器，不加载 GGUF…');
    const result = await request('/bind-runtime', 'POST', {values: values()}); message(result.message);
  }));
  perform(load);
})();
