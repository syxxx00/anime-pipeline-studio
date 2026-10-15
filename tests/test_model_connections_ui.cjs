const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor(tag = 'div') {
    this.tag = tag; this.children = []; this.listeners = {}; this.hidden = false;
    this.value = ''; this.type = ''; this.checked = false;
    this.classList = {toggle() {}};
  }
  append(...items) { for (const item of items) { item.parent = this; this.children.push(item); } }
  replaceChildren(...items) { this.children = []; this.append(...items); }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  closest(tag) { return this.tag === tag ? this : this.parent?.closest(tag); }
}
const tick = () => new Promise(resolve => setImmediate(resolve));
async function fixture() {
  const ids = ['sections','connections','fields','save','validate','probe','reload','status','checks','local-registration','bind'];
  const elements = new Map(ids.map(id => [id, new Element()]));
  const calls = [];
  const values = {authoring_backend:'local_gguf', authoring_base_url:'', authoring_model:'',
    authoring_api_key_env:'AI_COMIC_AUTHORING_API_KEY', local_llm_server:'runtime/llama-server.exe',
    local_llm_port:8191, auto_start_engine:true};
  const groups = [{id:'authoring', title:'理解模型', fields:[
    ['authoring_backend','backend'], ['authoring_base_url','url_optional'], ['authoring_model','text'],
    ['authoring_api_key_env','environment'], ['local_llm_server','path_optional'], ['local_llm_port','integer'],
    ['auto_start_engine','boolean']].map(([key, kind]) => ({key,kind,label:key}))}];
  const response = {revision:'a'.repeat(64), values, groups, restart_required:false};
  const fetch = async (url, options = {}) => {
    const body = options.body && JSON.parse(options.body);
    calls.push({url, method:options.method, body});
    let data = response;
    if (options.method === 'PUT') data = {revision:'b'.repeat(64), message:'保存成功，请重启。'};
    if (url.endsWith('/validate')) data = {configuration_ready:false, checks:[{ok:false,required:true,message:'缺少权重'}]};
    if (url.endsWith('/probe')) data = {services:[{service:'authoring',reachable:true,message:'仅探测',model_ids:['<model>']}]};
    if (url.endsWith('/bind-runtime')) data = {message:'语法已登记'};
    return {ok:true, json:async () => data};
  };
  const document = {getElementById:id => elements.get(id), createElement:tag => new Element(tag)};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../static/model-connections.js'),'utf8'), {document,fetch});
  await tick(); await tick();
  const input = key => elements.get('fields').children.flatMap(s => s.children)
    .filter(e => e.className === 'grid').flatMap(g => g.children).flatMap(l => l.children).find(e => e.name === key);
  return {elements, calls, input};
}

test('initial read does not validate, probe, bind, save or load a model', async () => {
  const {elements,calls,input} = await fixture();
  assert.equal(calls.length,1); assert.equal(calls[0].method,'GET');
  assert.equal(input('authoring_base_url').closest('label').hidden,true);
  assert.equal(input('local_llm_server').closest('label').hidden,false);
  assert.equal(elements.get('save').disabled,false);
});
test('backend choice changes relevant controls, not the saved provider implicitly', async () => {
  const {elements,calls,input} = await fixture();
  input('authoring_backend').value = 'openai_compatible';
  input('authoring_backend').listeners.change();
  assert.equal(input('authoring_base_url').closest('label').hidden,false);
  assert.equal(input('local_llm_server').closest('label').hidden,true);
  assert.equal(elements.get('local-registration').hidden,true);
  assert.equal(calls.length,1);
});
test('save carries revision, integer ports, booleans and environment name only', async () => {
  const {elements,calls,input} = await fixture();
  input('local_llm_port').value = '8291'; input('auto_start_engine').checked = false;
  elements.get('connections').listeners.submit({preventDefault(){}}); await tick(); await tick();
  const saved = calls.at(-1);
  assert.equal(saved.method,'PUT'); assert.equal(saved.body.expected_revision,'a'.repeat(64));
  assert.equal(saved.body.values.local_llm_port,8291); assert.equal(saved.body.values.auto_start_engine,false);
  assert.equal(saved.body.values.authoring_api_key_env,'AI_COMIC_AUTHORING_API_KEY');
  assert.match(elements.get('status').textContent,/重启/);
});
test('file checks, service GET probe and grammar registration are separate explicit actions', async () => {
  const {elements,calls} = await fixture();
  for (const [button, suffix] of [['validate','/validate'],['probe','/probe'],['bind','/bind-runtime']]) {
    elements.get(button).listeners.click(); await tick(); await tick();
    assert.ok(calls.at(-1).url.endsWith(suffix)); assert.equal(calls.at(-1).method,'POST');
  }
  assert.equal(calls.length,4);
});
