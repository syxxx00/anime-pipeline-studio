(() => {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
  const latestReview = (value, artworkId) => [...(value.candidate_reviews || [])].reverse().find(review => review.artwork_id === artworkId);

  function describeArtwork(value, artworkId) {
    const review = latestReview(value, artworkId), current = artworkId === value.incumbent_artwork_id;
    if (artworkId === value.selected_artwork_id && value.selection_authority === 'user') return {group: 'chosen', label: '你选定的版本 · 后续派发已停止', review};
    if (review?.verdict === 'reject') return {group: 'rejected', label: '已拒收 · 不继续交接', review};
    if (review?.verdict === 'explore') {
      const active = (value.exploration_artwork_ids || []).includes(artworkId);
      return {group: active ? 'exploration' : 'observed', label: active ? '探索分支 · 非交付候选' : '探索记录 · 当前未开放继续', review};
    }
    if (artworkId === value.selected_artwork_id) return {group: 'retained', label: '系统选出的交付候选 · 待你确认', review};
    if (current && review?.verdict === 'accept') return {group: 'retained', label: '评审记录可交付 · 待你确认', review};
    if (current && review?.verdict === 'retain') return {group: 'retained', label: '当前保留候选 · 尚非定稿', review};
    if (current) return {group: 'retained', label: '迭代起点 · 未核验，不代表认可', review};
    if (['accept', 'retain'].includes(review?.verdict)) return {group: 'retained', label: '历史保留候选 · 非当前版本', review};
    if (review?.verdict === 'baseline') return {group: 'observed', label: '历史起点 · 不代表认可', review};
    if (review?.verdict === 'uncertain') return {group: 'observed', label: '比较不确定 · 尚未保留', review};
    return {group: 'observed', label: '观察版本 · 尚未保留', review};
  }

  function describeSource(value, artworkId) {
    return artworkId ? `${artworkId}（${describeArtwork(value, artworkId).label}）` : '无像素源图 · 重新生成';
  }

  function intentDisplay(value) {
    const modelReading = value.base_request?.creation_intent?.interpretation_method === 'owned_runtime_source_reading_unverified';
    return modelReading ? {modelReading, title: '模型对原文的理解（未核验）',
      notice: '这些解释和优先级分类来自阅读模型，可被原文和后续真实反馈纠正；分类、引用和建议观察项不新增用户要求、偏好或已确认事实。',
      classification: '阅读器分类', observable: '模型建议观察项', notes: '模型理解补充（未核验）', goalPrefix: '模型理解（未核验）'}
      : {modelReading, title: '保存的目标解释（来源未识别）',
        notice: '以下仅为保存的目标解释与原分类；来源方法未识别，不推定来自用户、模型或已获确认。',
        classification: '原记录分类', observable: '原记录观察项', notes: '保存的说明记录（来源未识别）', goalPrefix: '原记录目标解释'};
  }

  function intentGoalsHtml(value) {
    const display = intentDisplay(value);
    return (value.base_request?.creation_intent?.goals || []).map(goal => `<article><b>${esc(display.classification)}：${esc(goal.priority ?? '未记录')}</b>${esc(goal.description)}
      ${goal.observable ? `<p>${esc(display.observable)}：${esc(goal.observable)}</p>` : ''}
      ${goal.source_quote ? `<details><summary>引用原件</summary><p>${esc(goal.source_quote)}</p></details>` : ''}</article>`).join('');
  }

  function goalLabelsFor(value) {
    const display = intentDisplay(value);
    return new Map((value.base_request?.creation_intent?.goals || []).map(goal => [goal.id, `${display.goalPrefix}：${goal.description}`]));
  }

  // The same renderer is used by cards and comparison; assessment text remains evidence, not a quality guarantee.
  function assessmentHtml(review, goalLabels = new Map()) {
    if (!review) return '<p class="note">尚无本次比较结论；执行完成不等于达标。</p>';
    const axes = [['requirement_fit', '内容符合度'], ['preservation', '已有关系保留'], ['technical_quality', '画质'], ['style_preference', '表现偏好']];
    const words = {better: '改善', same: '相同', worse: '退化', unknown: '未知'};
    let html = `<p class="review-evidence">${esc(review.evidence || '未记录比较依据')}</p><p class="axes">${axes.map(([key, label]) => `${label}：${esc(words[review[key]] || '未知')}`).join(' · ')}</p>`;
    if (review.unresolved_requirements?.length) html += `<p><strong>仍未完成：</strong>${esc(review.unresolved_requirements.map(goal => goalLabels.get(goal) || goal).join('；'))}</p>`;
    else html += '<p class="note">评审未列出未完成项；不等于用户已经认可。</p>';
    if (review.regressions?.length) html += `<p class="loss"><strong>结果损失：</strong>${esc(review.regressions.join('；'))}</p>`;
    if (review.verdict === 'explore') {
      const exploration = review.exploration;
      if (!exploration) return html + '<p class="loss">未记录探索依据，不据此承诺继续或恢复。</p>';
      const entries = [['阶段收益', exploration.gains], ['暂时损失', exploration.temporary_losses],
        ['验证什么', exploration.success_criteria], ['止损条件', exploration.stop_conditions]];
      html += `<div class="exploration-facts">${entries.map(([label, values]) => `<p><strong>${label}：</strong>${esc(values?.length ? values.join('；') : '未记录')}</p>`).join('')}
        <p><strong>待验证假设：</strong>${esc(exploration.hypothesis || '未记录，不承诺恢复')}</p>
        <p><strong>本次评审建议的后续上限：</strong>${esc(exploration.max_followup_jobs ?? '未记录')} 次；不重置实际分支预算</p></div>`;
    }
    return html;
  }

  function branchBudgetHtml(branch, active) {
    if (!branch) return '';
    return `<p class="note"><strong>分支实际预算：</strong>总上限 ${esc(branch.max_followup_jobs ?? '未记录')} 次 · 已用 ${esc(branch.followup_jobs ?? '未记录')} 次 · 剩余 ${esc(branch.remaining_jobs ?? '未记录')} 次<br>
      <strong>分支起点：</strong>${esc(branch.root_artwork_id)}${active ? '' : '（本图当前未开放交接）'}</p>`;
  }

  function localEditAvailability(value, artworkId) {
    if (value.autonomous_planning === true) return {allowed: false, reason: '本任务启用了自动提案。圈选前请先暂停并明确点击“切为手动决策”；不会替你偷偷改变规划模式'};
    if (!(value.available_operations || []).includes('local_edit') || value.local_edit_capability?.source_sha256_required !== true) return {allowed: false, reason: '后端尚未加载当前局部修改能力，请先更新服务'};
    if (value.observation_capability?.available !== true || value.observation_capability.image_dependent_decisions_require_observation_id !== true) return {allowed: false, reason: '服务尚未接入原件观察校验，请更新服务后再修改'};
    if (!(value.allowed_models || []).includes('newbie')) return {allowed: false, reason: '本任务没有启用当前局部工具需要的 NewBie'};
    if (!['running', 'paused'].includes(value.status)) return {allowed: false, reason: '本任务已结束；不能在这里继续派发'};
    if ((Number.isFinite(value.remaining_jobs) && value.remaining_jobs <= 0) ||
        (Number.isFinite(value.budget?.max_jobs) && (value.steps || []).length >= value.budget.max_jobs)) return {allowed: false, reason: '本任务的生图预算已用完；不能继续圈选派发'};
    if (value.decision_queued || (value.steps || []).some(step => ['reserved', 'submitted'].includes(step.status))) return {allowed: false, reason: '有决策或任务正在执行，请等待完成'};
    const reviewed = new Set((value.candidate_reviews || []).map(review => review.artwork_id));
    if ((value.steps || []).some(step => step.artwork_id && !reviewed.has(step.artwork_id))) return {allowed: false, reason: '新图尚未比较，请先完成审阅，不能跳过结果继续加工'};
    if (!artworkId || !(value.artworks || value.artifacts || []).some(artwork => artwork.id === artworkId)) return {allowed: false, reason: '源版本不属于本任务'};
    if (artworkId !== value.incumbent_artwork_id && !(value.exploration_artwork_ids || []).includes(artworkId)) return {allowed: false, reason: '此版本不是当前保留或有预算的探索源'};
    return {allowed: true, reason: ''};
  }

  function localEditDecision(value, artworkId, options) {
    const availability = localEditAvailability(value, artworkId);
    if (!availability.allowed) throw new Error(availability.reason);
    const instruction = String(options.instruction || '').trim();
    if (!instruction || instruction.length > 3000) throw new Error('请用不超过3000字说明局部修改内容');
    if (!String(options.mask_data_url || '').startsWith('data:image/png;base64,')) throw new Error('需要当前图片的 PNG 圈选遮罩');
    if (!/^[a-f0-9]{64}$/i.test(String(options.source_sha256 || ''))) throw new Error('缺少圈选时源图校验，请重新打开图片');
    const steps = Number(options.steps), denoise = Number(options.denoise);
    if (!Number.isInteger(steps) || steps < 5 || steps > 40 || !Number.isFinite(denoise) || denoise < .15 || denoise > .75) throw new Error('局部参数超出支持范围');
    const goals = value.task_goals || value.base_request?.creation_intent?.goals || [];
    const goalIds = goals.filter(goal => goal.priority === 'required').map(goal => goal.id);
    if (!goalIds.length || !value.original_text || !value.original_sha256) throw new Error('缺少完整原句或关联目标，请重新读取任务');
    return {operation: 'local_edit', target: 'newbie', source_artwork_id: artworkId,
      reason: '用户针对当前源版本明确圈选并提出本次局部修改；原始任务仍保留，效果须另行比较。',
      objective: instruction, preserve: ['未圈选区域的原图像素', '原句规定的角色与场景关系；未成立的内容不因局部修改自动算完成'],
      acceptance: ['未圈选区域逐像素一致', `检查圈内是否实现：${instruction}`, '圈内画法、身份与相关动作没有未经判断的新退化'],
      goal_ids: goalIds, original_text: value.original_text, original_sha256: value.original_sha256,
      region_edit: {instruction, mask_data_url: options.mask_data_url, source_sha256: options.source_sha256, steps, denoise}};
  }

  function userSelectionAvailability(value, artworkId) {
    const capability = value.user_selection_capability;
    if (capability?.enabled !== true || capability.requires_artwork_sha256 !== true || capability.quality_certification !== false) return {allowed: false, reason: '后端尚未加载当前版本选择能力，请先更新服务'};
    if (value.user_selection_receipt || value.selection_authority === 'user') return {allowed: false, reason: '已经保存你的版本选择；新的创作请另建任务'};
    if (!['running', 'paused', 'completed', 'cancelled', 'exhausted'].includes(value.status)) return {allowed: false, reason: '任务状态未知，请重新读取'};
    if ((value.steps || []).some(step => ['reserved', 'submitted'].includes(step.status)) || (value.jobs || []).some(job => !['completed', 'failed', 'cancelled'].includes(job.status))) return {allowed: false, reason: '仍有在途任务，请先暂停后续派发，等待结果并核对队列后选择'};
    const artwork = (value.artworks || []).find(artwork => artwork.id === artworkId);
    if (!artwork) return {allowed: false, reason: '此版本不属于本任务'};
    if (!/^[a-f0-9]{64}$/.test(artwork.artwork_sha256 || '')) return {allowed: false, reason: '当前原图无法校验，请检查文件是否存在'};
    if (!Number.isInteger(value.revision) || value.revision < 0 || !Array.isArray(value.decisions) || !/^[a-f0-9]{64}$/.test(value.feedback_signature || '')) return {allowed: false, reason: '缺少当前任务或评价校验，请重新读取'};
    return {allowed: true, reason: ''};
  }

  function userSelectionPayload(value, artworkId, requestId) {
    const available = userSelectionAvailability(value, artworkId);
    if (!available.allowed) throw new Error(available.reason);
    if (!/^[a-zA-Z0-9_-]{8,128}$/.test(String(requestId || ''))) throw new Error('缺少本次选择的唯一标识');
    return {request_id: requestId, artwork_id: artworkId, artwork_sha256: value.artworks.find(artwork => artwork.id === artworkId).artwork_sha256,
      revision: value.revision, decision_count: value.decisions.length, feedback_signature: value.feedback_signature};
  }

  function userSelectionReceiptMatches(value, payload) {
    const receipt = value.user_selection_receipt;
    return value.status === 'completed' && value.stop_reason === 'user_selected' && value.selection_authority === 'user' && value.selected_artwork_id === payload.artwork_id &&
      receipt?.scope === 'version_choice_only_not_quality_certification' &&
      receipt?.request_id === payload.request_id && receipt.artwork_id === payload.artwork_id && receipt.artwork_sha256 === payload.artwork_sha256 &&
      ['request_id', 'artwork_id', 'artwork_sha256', 'revision', 'decision_count', 'feedback_signature'].every(key => receipt.request?.[key] === payload[key]);
  }

  function userSelectionContextMatches(value, source) {
    return value.id === source.runId && value.original_sha256 === source.originalHash && value.revision === source.payload.revision &&
      value.decisions?.length === source.payload.decision_count && value.feedback_signature === source.payload.feedback_signature &&
      value.artworks?.find(artwork => artwork.id === source.payload.artwork_id)?.artwork_sha256 === source.payload.artwork_sha256;
  }

  function observationRequest(context, artworkId, sourceHash, observationId) {
    if (context.observation_capability?.available !== true || context.observation_capability.image_dependent_decisions_require_observation_id !== true) throw new Error('当前服务没有接入原件观察校验');
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$/.test(String(observationId || ''))) throw new Error('缺少本次观察的唯一标识');
    if (!Number.isInteger(context.revision) || context.revision < 0 || !Number.isInteger(context.decision_count) || context.decision_count < 0 || !/^[a-f0-9]{64}$/.test(context.feedback_signature || '')) throw new Error('观察缺少当前版本或原文评价校验');
    if (!context.artifacts?.some(artwork => artwork.id === artworkId) || !/^[a-f0-9]{64}$/.test(sourceHash || '')) throw new Error('观察缺少本次圈选的源图及像素校验');
    return {observation_id: observationId, revision: context.revision, decision_count: context.decision_count,
      feedback_signature: context.feedback_signature, selected_artifacts: {local_edit_source: [artworkId]},
      expected_image_sha256: {[artworkId]: sourceHash}};
  }

  function observationReceiptMatches(archive, context, payload) {
    const manifest = archive?.manifest, original = archive?.api_request;
    return archive?.archive_integrity_verified === true && archive.currentness === 'current' &&
      manifest?.id === payload.observation_id && manifest.run_id === context.run_id &&
      manifest.original_sha256 === context.original_sha256 && manifest.revision === payload.revision &&
      manifest.feedback_signature === payload.feedback_signature && archive.context?.decision_count === payload.decision_count &&
      original?.observation_id === payload.observation_id && original.revision === payload.revision &&
      original.decision_count === payload.decision_count && original.feedback_signature === payload.feedback_signature &&
      JSON.stringify(original.selected_artifacts) === JSON.stringify(payload.selected_artifacts) &&
      JSON.stringify(original.expected_image_sha256) === JSON.stringify(payload.expected_image_sha256) &&
      JSON.stringify(manifest.roles) === JSON.stringify(payload.selected_artifacts) &&
      Object.entries(payload.expected_image_sha256).every(([artworkId, sha]) => manifest.images?.[artworkId]?.sha256 === sha);
  }

  function describePlanning(value) {
    if (!value.autonomous_planning) return null;
    const stage=value.planning?.stage||'idle';
    const labels={idle:'等待本地提案',reading_concepts:'读取当前概念原文',freezing_originals:'绑定本次真实原图',observing_pixels:'观察真实图像',locating_region:'根据实际原图定位修改区域',choosing_next_operation:'选择下一次操作',proposal_ready:'本地提案已返回 · 尚未质量认证',failed:'本地规划失败',cancelled:'本地规划已取消'};
    return {stage,label:labels[stage]||`本地规划 · ${stage}`,error:value.planning?.error||'',evidence:value.planning?.evidence_directory||'',quality_verified:false};
  }

  function planningModeAvailability(value) {
    const capability=value.planning_mode_capability;
    if(capability?.available!==true||capability.requires_paused!==true||capability.mode_change_resumes_run!==false)return {allowed:false,reason:'后端尚未提供暂停状态下的显式规划模式切换，请更新服务'};
    if(value.status!=='paused')return {allowed:false,reason:'请先暂停任务；切换模式不会替你暂停或继续'};
    if(value.decision_queued||(value.steps||[]).some(step=>['reserved','submitted'].includes(step.status))||
        (value.jobs||[]).some(job=>!['completed','failed','cancelled'].includes(job.status)))return {allowed:false,reason:'仍有在途任务或待派发决策，请等它们收住后再切换'};
    if(value.autonomous_planning===true&&['reading_concepts','freezing_originals','observing_pixels','locating_region','choosing_next_operation'].includes(value.planning?.stage))return {allowed:false,reason:'当前本地规划尚未停止，等待取消与资源清理完成后再切换'};
    if(typeof value.autonomous_planning!=='boolean'||!Number.isSafeInteger(value.revision)||value.revision<0)return {allowed:false,reason:'缺少当前真实模式与版本，未猜测切换状态'};
    if(value.autonomous_planning===false&&capability.automatic_available!==true)return {allowed:false,reason:'当前服务尚未绑定自动提案能力；手动决策与圈选仍可使用'};
    return {allowed:true,reason:''};
  }

  function planningModePayload(value) {
    const available=planningModeAvailability(value);if(!available.allowed)throw new Error(available.reason);
    return {autonomous_planning:!value.autonomous_planning,revision:value.revision};
  }

  function planningModeReceiptMatches(value,source) {
    return value.id===source.runId&&value.status==='paused'&&value.revision===source.payload.revision+1&&
      value.autonomous_planning===source.payload.autonomous_planning&&value.original_sha256===source.originalHash;
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = {describeArtwork, describeSource, assessmentHtml, branchBudgetHtml, localEditAvailability, localEditDecision,
    userSelectionAvailability, userSelectionPayload, userSelectionReceiptMatches, userSelectionContextMatches, observationRequest, observationReceiptMatches,
    describePlanning, planningModeAvailability, planningModePayload, planningModeReceiptMatches, intentDisplay, intentGoalsHtml, goalLabelsFor, esc};
  if (typeof document === 'undefined') return;
  const $ = id => document.getElementById(id);
  let id = new URLSearchParams(location.search).get('id') || '', run = null, timer = null, busy = false, graphKey = '', loadSequence = 0;
  let reviewArtworkId = '', reviewSaving = false;
  let editSource = null, editLoading = false, editSaving = false, editAttempted = false, editPoint = null, editDrawing = false, editPainted = false, editSequence = 0;
  let selectionSource = null, selectionLoading = false, selectionSaving = false, selectionSequence = 0, selectionImageUrl = '', selectionImageVerified = false;
  let planningAttempt = null;
  const compareIds = new Set(), reviewDrafts = new Map();
  const labels = {running: '执行中', paused: '已暂停', completed: '系统选择完成，待你确认', exhausted: '达到预算，非质量通过', cancelled: '已取消'};
  const reasons = {awaiting_user_review: '待你查看本轮结果', user: '由你暂停', user_feedback_changed: '评价已变化，旧决策停下，等待重新审阅', service_restart: '服务重启后安全暂停', service_shutdown: '服务停止后安全暂停', controller_error: '执行异常，查看记录', no_demonstrated_improvement: '没有证实改善，停止继续加工', candidate_comparison_uncertain: '比较结论不确定，待审阅'};
  reasons.user_selected = '由你选定，后续派发已停止';
  reasons.observation_stale = '原图或观察条件已变化，旧决策停止，等待重新审阅';
  const statusLabel = value => value.selection_authority === 'user' && value.status === 'completed' ? '你已选定版本' : labels[value.status] || value.status;
  const operations = {generate: '重新生成', handoff: '像素交接', local_edit: '局部修改', rollback: '回退', finish: '结束选择', pause: '暂停'};
  const stepStatuses = {reserved: '准备派发', submitted: '已派发', completed: '生成完成', failed: '执行失败'};
  async function read(path, options = {}) {
    const response = await fetch(path, options);
    if (!response.ok) {const error = new Error(`HTTP ${response.status}: ${await response.text()}`); error.status = response.status; throw error;}
    return response.json();
  }
  const artworkById = artworkId => run?.artworks?.find(artwork => artwork.id === artworkId);
  const selectionStorageKey = runId => `ai-comic-user-selection-v1:${runId}`;
  const pendingSelections = new Map();
  function pendingSelection(runId) {
    if (pendingSelections.has(runId)) return pendingSelections.get(runId);
    try {
      const value = JSON.parse(sessionStorage.getItem(selectionStorageKey(runId)) || 'null');
      if (value?.runId === runId && value.payload?.request_id && value.payload?.artwork_id) {pendingSelections.set(runId, value); return value;}
    } catch (_) { /* No writes or automatic retries during observation. */ }
    return null;
  }
  function savePendingSelection(source) {
    // Persist before the first POST so refresh cannot silently create another choice.
    sessionStorage.setItem(selectionStorageKey(source.runId), JSON.stringify(source));
    pendingSelections.set(source.runId, source);
  }
  function clearPendingSelection(runId) {
    try {sessionStorage.removeItem(selectionStorageKey(runId));} catch (_) { /* Server receipt remains authoritative. */ }
    pendingSelections.delete(runId);
  }

  const observationStorageKey = (runId, artworkId) => `ai-comic-edit-observation-v1:${runId}:${artworkId}`;
  async function prepareEditObservation(source, context) {
    const key = observationStorageKey(source.runId, source.artworkId);
    let pending = null;
    try {pending = JSON.parse(sessionStorage.getItem(key) || 'null');} catch (_) {throw new Error('无法读取本次观察记录；没有重新创建或派发');}
    let payload;
    if (pending) {
      payload = pending.payload;
      const expected = observationRequest(context, source.artworkId, source.sourceHash, payload?.observation_id);
      if (pending.runId !== source.runId || pending.originalHash !== source.originalHash || JSON.stringify(payload) !== JSON.stringify(expected)) {
        // The old observation was never a generation submission. Only after a
        // read confirms its state may another explicit gesture create a new one.
        const old = await read(`/api/iteration-runs/${encodeURIComponent(source.runId)}/observations/${encodeURIComponent(payload?.observation_id || '')}`);
        if (old.currentness === 'stale') sessionStorage.removeItem(key);
        throw new Error('先前观察的上下文已变化，本次没有派发；请重新打开源图。未知观察不会自动重建');
      }
    } else {
      payload = observationRequest(context, source.artworkId, source.sourceHash, `edit-${crypto.randomUUID()}`);
      // Save before POST. A lost receipt/refresh must read this same package,
      // not silently freeze another set of pixels or create a new identifier.
      sessionStorage.setItem(key, JSON.stringify({runId: source.runId, originalHash: source.originalHash, payload}));
    }
    $('editStatus').textContent = pending ? '正在重查同一次原件观察（只读取）…' : '正在保存这次实际原图、输入与原文评价…';
    let archive;
    if (pending) archive = await read(`/api/iteration-runs/${encodeURIComponent(source.runId)}/observations/${encodeURIComponent(payload.observation_id)}`);
    else {
      try {archive = await read(`/api/iteration-runs/${encodeURIComponent(source.runId)}/observations`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});}
      catch (error) {
        // Unknown writes are reconciled by reads, never by a second POST.
        try {archive = await read(`/api/iteration-runs/${encodeURIComponent(source.runId)}/observations/${encodeURIComponent(payload.observation_id)}`);}
        catch (_) {throw new Error(`原件保存回执未确认；保留同一观察标识，没有提交修改：${error.message}`);}
      }
    }
    if (!observationReceiptMatches(archive, context, payload)) {
      if (archive?.currentness === 'stale') sessionStorage.removeItem(key);
      throw new Error('原件观察不匹配或已过期，没有提交修改；请重新打开源图');
    }
    return payload.observation_id;
  }

  function humanReviewsHtml(artwork) {
    return artwork?.reviews?.length ? artwork.reviews.map(review => `<article class="human-review"><p>${esc(review.text)}</p><small>${esc(review.created_at || '')}</small></article>`).join('') : '<p class="note">尚无保存的原文评价。</p>';
  }
  function cardHtml(artwork, comparison = false) {
    const state = describeArtwork(run, artwork.id), review = state.review;
    const edit = localEditAvailability(run, artwork.id);
    const selection = userSelectionAvailability(run, artwork.id), pending = pendingSelection(run.id);
    const pendingHere = pending?.payload.artwork_id === artwork.id;
    const step = (run.steps || []).find(step => step.artwork_id === artwork.id);
    const branch = (run.exploration_branches || []).find(branch => branch.root_artwork_id === artwork.id || (branch.artwork_ids || []).includes(artwork.id));
    return `<article class="card ${state.group}" data-artwork="${esc(artwork.id)}"><h3>${esc(state.label)}</h3>
      <a class="image-link" href="${esc(artwork.image_url)}" target="_blank" rel="noopener"><img src="${esc(artwork.image_url)}" alt="版本 ${esc(artwork.id)}"></a>
      <strong class="artwork-id">${esc(artwork.id)}</strong>${step?.decision.source_artwork_id ? `<p class="source"><strong>源图：</strong>${esc(describeSource(run, step.decision.source_artwork_id))}</p>` : ''}
      ${assessmentHtml(review, goalLabelsFor(run))}
      ${branchBudgetHtml(branch, state.group === 'exploration')}
      <details class="user-reviews"><summary>你的原文评价 · ${artwork.reviews?.length || 0} 条</summary>${humanReviewsHtml(artwork)}</details>
      ${comparison ? '' : `<div class="card-actions"><label><input type="checkbox" data-compare="${esc(artwork.id)}" ${compareIds.has(artwork.id) ? 'checked' : ''}> 加入比较</label><button data-review="${esc(artwork.id)}">写评价</button><button data-edit="${esc(artwork.id)}" ${edit.allowed ? '' : 'disabled'} title="${esc(edit.reason || '圈选区域，说明本次怎样修改')}">圈选修改</button><button data-select="${esc(artwork.id)}" ${(pending ? pendingHere : selection.allowed) && !selectionSaving ? '' : 'disabled'} title="${esc(pending ? '上次选择回执未确认，先核对同一次请求' : selection.reason || '由你选择这版，不替代内容和质量核验')}">${pendingHere ? '核对选择回执' : '选定这版并停止'}</button></div>`}
      <details class="file-details"><summary>文件位置与详情</summary><p>${esc(artwork.file_path || '未记录文件位置')}</p><a href="/?view=library&artwork=${encodeURIComponent(artwork.id)}" target="_blank" rel="noopener">打开作品详情 ↗</a></details></article>`;
  }
  function renderCompareControls() {
    $('compare').disabled = compareIds.size !== 2;
    $('clearCompare').disabled = compareIds.size === 0;
    $('compareCount').textContent = compareIds.size ? `已选择 ${compareIds.size}/2 张` : '选择两张图片进行比较';
  }
  function render(value, fromPoll = false) {
    // A control/manual read is newer than a poll that was already in flight.
    // The old poll may finish, but must not undo a displayed durable choice.
    if (!fromPoll) {++loadSequence; clearTimeout(timer); if (id) timer = setTimeout(load, 2500);}
    run = value; $('empty').hidden = true; $('content').hidden = false;
    const pending = pendingSelection(value.id);
    if (pending && userSelectionReceiptMatches(value, pending.payload)) clearPendingSelection(value.id);
    $('selectionNotice').textContent = pending && !userSelectionReceiptMatches(value, pending.payload) ? '上次版本选择回执未确认。不会自动再次提交；可点击该图的“核对选择回执”。' : '';
    const active = (value.jobs || []).find(job => ['queued', 'running'].includes(job.status));
    const planning = describePlanning(value);
    $('state').textContent = `${value.id} · ${statusLabel(value)}`;
    $('phase').textContent = active ? `${active.status === 'queued' ? '等待执行' : '生成与检查'} · ${active.stage || '执行中'}` : planning && (value.status==='running'||planning.stage==='failed') ? planning.label : value.waiting_for_decision ? '等待审阅与下一步决策' : statusLabel(value);
    $('detail').textContent = active ? `任务 ${active.id} · ${Math.round((active.progress || 0) * 100)}%` : `已派发 ${(value.steps || []).length}/${value.budget.max_jobs} 个步骤${value.stop_reason ? ' · ' + (reasons[value.stop_reason] || value.stop_reason) : ''}${planning?.error?' · '+planning.error:''}${planning?.evidence?' · 提案记录 '+planning.evidence:''}`;
    $('controlMeaning').textContent = active ? '暂停只阻止后续派发，已进入执行的任务仍会完成并保留结果。' : planning ? '暂停会取消当前本地规划并阻止后续派发；继续后会读取当前条件与原图，再提出下一操作。提案和视觉描述尚未经过质量认证。' : '继续恢复后续派发资格；没有新的审阅决策时仍会等待，不会凭空生成图片。';
    if($('planningMeaning'))$('planningMeaning').textContent=planning?'本任务已明确启用本地结果驱动提案，根据当前概念、完整条件与真实图像选择操作，不是固定模型链。圈选前须先暂停、等在途操作结束，再由你明确切为手动决策。视觉描述与提案可能有误，不认证画面质量；原文评价仍保留，刷新只读取任务，不自动继续。':'本任务当前为手动决策，圈选不会暗中重新启用自动提案。若希望恢复自动提案，先暂停并明确切换模式，再另点继续。原文评价保持原话，不自动认可作品；检查、探索和成品不是同一件事。';
    $('pause').disabled = busy || value.status !== 'running'; $('resume').disabled = busy || value.status !== 'paused'; $('cancel').disabled = busy || ['completed', 'cancelled', 'exhausted'].includes(value.status);
    const modeAvailable=planningModeAvailability(value),modeUnknown=planningAttempt?.runId===value.id;
    $('planningMode').textContent=modeUnknown?'重查模式（只读取）':value.autonomous_planning?'切为手动决策':'重新启用自动提案';
    $('planningMode').disabled=busy||editLoading||editSaving||selectionSaving||(!modeUnknown&&!modeAvailable.allowed);
    $('planningMode').title=modeUnknown?'只读取实际状态，不重新提交切换':modeAvailable.reason;
    $('planningModeStatus').textContent=modeUnknown?'上次切换回执未确认；不会自动重发。请重查实际模式。':`${value.autonomous_planning?'自动提案':'手动决策'}${modeAvailable.allowed?' · 切换后仍保持暂停，另点继续才执行':' · '+modeAvailable.reason}`;
    $('original').textContent = value.original_text;
    $('digest').textContent = `原始输入 SHA256 · ${value.original_sha256}`;
    const intent = value.base_request.creation_intent || {};
    const display = intentDisplay(value);
    $('goalsTitle').textContent = display.title; $('intentAuthority').textContent = display.notice;
    $('intentNotesTitle').textContent = display.notes; $('goals').innerHTML = intentGoalsHtml(value);
    $('notes').innerHTML = (intent.interpretation_notes || []).map(note => `<li>${esc(note)}</li>`).join('');
    $('resultMeaning').textContent = value.selected_artwork_id ? value.selection_authority === 'user' ? `你选定了 ${value.selected_artwork_id}，后续派发已停止。这是版本选择，不表示角色、内容与质量全部达标；历史评审和原文评价保持不变。` : `系统选择了 ${value.selected_artwork_id}，仍不代表你的认可。机器选择与用户主动选定分别记录。` : '尚无已确认的交付结果。你可以选定已有版本并停止，也可以继续比较和评价；不把最后一张自动当作好图。';
    const known = new Set((value.artworks || []).map(artwork => artwork.id));
    for (const artworkId of compareIds) if (!known.has(artworkId)) compareIds.delete(artworkId);
    const groups = [
      ['chosen', '你选定的版本', '由你主动选择，不改写历史评审，也不认证全部要求已完成。'],
      ['retained', '起点与保留候选', '起点未核验；保留继续观察，不等于定稿。'],
      ['exploration', '探索分支', '有阶段收益也可能有暂时损失，不覆盖保留候选，不可直接交付。'],
      ['rejected', '拒收结果', '保留为比较证据，不以“以后会修好”为理由继续交接。'],
      ['observed', '其他观察版本', '尚未保留，或探索已关闭；仍可比较和评价。'],
    ];
    $('artworks').innerHTML = groups.map(([group, title, note]) => {
      const artworks = (value.artworks || []).filter(artwork => describeArtwork(value, artwork.id).group === group);
      return artworks.length ? `<div class="version-group" data-version-group="${group}"><h3>${title} <small>${artworks.length} 张</small></h3><p class="note">${note}</p><div class="cards">${artworks.map(artwork => cardHtml(artwork)).join('')}</div></div>` : '';
    }).join('') || '<p>还没有生成图片。</p>';
    renderCompareControls();
    $('steps').innerHTML = (value.steps || []).map((step, index) => {
      const decision = step.decision, outcome = latestReview(value, step.artwork_id);
      return `<article class="step"><h3>${index + 1}. ${esc(operations[decision.operation] || decision.operation)} → ${esc(decision.target)} · ${esc(stepStatuses[step.status] || step.status)}</h3>
        <p><strong>源图：</strong>${esc(describeSource(value, decision.source_artwork_id))}</p>
        <p><strong>本次尝试：</strong>${esc(decision.objective)}</p><p><strong>希望保留：</strong>${esc((decision.preserve || []).join('；') || '未指定像素保留')}</p>
        <p><strong>为什么试：</strong>${esc(decision.reason)}</p><p><strong>要检查什么：</strong>${esc((decision.acceptance || []).join('；'))}</p>
        ${step.artwork_id ? `<div class="step-outcome"><strong>结果 ${esc(step.artwork_id)}：</strong>${esc(describeArtwork(value, step.artwork_id).label)}${assessmentHtml(outcome, goalLabelsFor(value))}</div>` : ''}
        <small>原句校验 ${esc(decision.original_sha256)} · ${esc(step.job_id || '尚未派发')}</small>${step.error ? `<p class="loss">${esc(step.error)}</p>` : ''}</article>`;
    }).join('') || '<p>正在观察已有候选，还没有派发新步骤。</p>';
    $('records').textContent = JSON.stringify({original_text: value.original_text, original_sha256: value.original_sha256,
      incumbent_artwork_id: value.incumbent_artwork_id, selected_artwork_id: value.selected_artwork_id,
      selection_authority: value.selection_authority, user_selection_receipt: value.user_selection_receipt,
      exploration_artwork_ids: value.exploration_artwork_ids || [], exploration_branches: value.exploration_branches || [],
      decisions: value.decisions, reviews: value.candidate_reviews, user_reviews: (value.artworks || []).map(artwork => ({artwork_id: artwork.id, reviews: artwork.reviews || []})),
      events: value.events, autonomous_planning: Boolean(value.autonomous_planning), planning: value.planning,
      automatic_quality_certification: false, jobs: (value.jobs || []).map(job => ({id: job.id, status: job.status, request: job.request, checks: job.checks, error: job.error}))}, null, 2);
    const key = JSON.stringify(value.graph);
    if (key !== graphKey && value.graph && typeof PipelineDiagram !== 'undefined') {
      graphKey = key; PipelineDiagram.mount($('graph'), {graph: value.graph});
      const note = $('graph').querySelector('.diagram-note');
      if (note) note.textContent = '按本任务实际保存的源图、决策与步骤绘制；不是预设模型链。结合版本用途和比较结果查看。';
    }
    if ($('reviewDialog').open && artworkById(reviewArtworkId)) $('reviewHistory').innerHTML = humanReviewsHtml(artworkById(reviewArtworkId));
    if ($('editDialog').open && editSource) {
      const availability = localEditAvailability(value, editSource.artworkId);
      const stale = value.id !== editSource.runId || value.original_sha256 !== editSource.originalHash || value.feedback_signature !== editSource.feedbackSignature;
      $('submitEdit').disabled = editLoading || editSaving || editAttempted || stale || !availability.allowed || !editPainted;
      $('editDispatchMeaning').textContent = value.status === 'paused' ? '提交时会继续本任务并只排入这一次明确修改；不会复跑旧步骤。' : '只排入这一次明确修改，下一步仍需查看实际结果。';
      if (!editSaving && !editAttempted && (stale || !availability.allowed)) $('editStatus').textContent = stale ? '任务或原文评价已变化，请关闭并重新打开，结合最新内容决定修改。' : availability.reason;
    }
    if ($('selectionDialog').open && selectionSource) renderSelectionState(value);
  }
  async function load() {
    clearTimeout(timer); const selectedId = id, sequence = ++loadSequence;
    if (!selectedId) { $('content').hidden = true; $('empty').hidden = false; return; }
    try {
      const value = await read(`/api/iteration-runs/${encodeURIComponent(selectedId)}`);
      if (selectedId !== id || sequence !== loadSequence) return;
      render(value, true); $('connection').textContent = '';
    } catch (error) { if (selectedId === id && sequence === loadSequence) $('connection').textContent = error.message; }
    finally { if (selectedId === id && sequence === loadSequence) timer = setTimeout(load, 2500); }
  }
  async function list() {
    try {
      const rows = await read('/api/iteration-runs');
      $('runs').replaceChildren(new Option('选择任务', ''), ...rows.map(row => new Option(`${row.id} · ${statusLabel(row)} · ${row.original_text.slice(0, 28)}`, row.id)));
      $('runs').value = id;
      if (!id) $('empty').textContent = rows.length ? '选择已有任务查看。' : '尚无逐步创作任务。';
    } catch (error) { $('connection').textContent = error.message; }
  }
  function openReview(artworkId) {
    const artwork = artworkById(artworkId); if (!artwork) return;
    reviewArtworkId = artworkId;
    $('reviewArtwork').textContent = `${artworkId} · ${describeArtwork(run, artworkId).label}`;
    $('reviewText').value = reviewDrafts.get(artworkId) || '';
    $('reviewStatus').textContent = '';
    $('saveReview').disabled = reviewSaving;
    $('reviewHistory').innerHTML = humanReviewsHtml(artwork);
    $('reviewDialog').showModal(); $('reviewText').focus();
  }
  $('artworks').onclick = event => {
    const button = event.target.closest('[data-review]'); if (button) openReview(button.dataset.review);
    const editButton = event.target.closest('[data-edit]'); if (editButton && !editButton.disabled) openEdit(editButton.dataset.edit);
    const selectButton = event.target.closest('[data-select]'); if (selectButton && !selectButton.disabled) openSelection(selectButton.dataset.select);
  };
  $('artworks').onchange = event => {
    const input = event.target.closest('input[data-compare]'); if (!input) return;
    if (input.checked && compareIds.size >= 2 && !compareIds.has(input.dataset.compare)) {
      input.checked = false; $('compareCount').textContent = '最多比较两张；先取消一张即可更换。'; return;
    }
    if (input.checked) compareIds.add(input.dataset.compare); else compareIds.delete(input.dataset.compare);
    renderCompareControls();
  };
  $('compare').onclick = () => {
    const artworks = [...compareIds].map(artworkById).filter(Boolean); if (artworks.length !== 2) return;
    $('comparison').innerHTML = artworks.map(artwork => cardHtml(artwork, true)).join(''); $('compareDialog').showModal();
  };
  $('clearCompare').onclick = () => { compareIds.clear(); if (run) render(run); };
  $('closeCompare').onclick = () => $('compareDialog').close();
  $('closeReview').onclick = () => $('reviewDialog').close();
  function renderSelectionState(value) {
    const source = selectionSource; if (!source) return;
    const matched = userSelectionReceiptMatches(value, source.payload), pending = pendingSelection(source.runId);
    const available = userSelectionAvailability(value, source.payload.artwork_id);
    const stale = !userSelectionContextMatches(value, source);
    const retry = pending?.payload.request_id === source.payload.request_id;
    const imageChanged = value.artworks?.find(artwork => artwork.id === source.payload.artwork_id)?.artwork_sha256 !== source.payload.artwork_sha256;
    if (imageChanged) {
      selectionImageVerified = false; $('selectionImage').hidden = true;
      $('selectionArtwork').textContent = `${source.payload.artwork_id} · 旧观察原图已经变化或缺失，不能当作当前原图`;
    }
    const unsupported = value.user_selection_capability?.enabled !== true || value.user_selection_capability.requires_artwork_sha256 !== true || value.user_selection_capability.quality_certification !== false;
    const inFlight = (value.steps || []).some(step => ['reserved', 'submitted'].includes(step.status)) || (value.jobs || []).some(job => !['completed', 'failed', 'cancelled'].includes(job.status));
    $('confirmSelection').textContent = retry ? '重送同一次选择（相同请求标识）' : '确认选定这版并停止';
    $('confirmSelection').disabled = selectionLoading || selectionSaving || (!selectionImageVerified && !retry) || unsupported || inFlight || matched || !!value.user_selection_receipt || (!retry && (stale || !available.allowed));
    if (selectionSaving) return;
    if (matched) $('selectionStatus').textContent = `已保存选择：${source.payload.artwork_id}。后续派发已停止，原评审与评价保持不变。`;
    else if (retry) $('selectionStatus').textContent = imageChanged ? '未核对到回执，原图也已变化。可主动重送原请求核对：不会换图或创建新选择；服务若尚未保存，将拒绝已失效的原请求。' : '未核对到这次选择的回执。不会自动重发；你可重查回执，或主动重送同一次请求，不会新建另一项选择。';
    else if (stale || !available.allowed) $('selectionStatus').textContent = stale ? '任务、评价或图像已变化，请关闭并重新打开，结合当前版本决定。' : available.reason;
  }
  async function openSelection(artworkId) {
    if (!run || selectionSaving) return;
    const selectedId = id, sequence = ++selectionSequence;
    selectionSource = null; selectionLoading = true; selectionImageVerified = false;
    if (selectionImageUrl) URL.revokeObjectURL(selectionImageUrl);
    selectionImageUrl = ''; $('selectionImage').removeAttribute('src'); $('selectionImage').hidden = true;
    $('selectionArtwork').textContent = artworkId; $('confirmSelection').disabled = true;
    $('selectionStatus').textContent = '正在核对任务与当前原图…'; $('selectionDialog').showModal();
    try {
      const live = await read(`/api/iteration-runs/${encodeURIComponent(selectedId)}`);
      if (sequence !== selectionSequence || id !== selectedId || !$('selectionDialog').open) return;
      const pending = pendingSelection(selectedId);
      if (pending && pending.payload.artwork_id !== artworkId) throw new Error('已有另一版选择的回执未确认，请先核对原请求');
      const source = pending || {runId: selectedId, originalHash: live.original_sha256, payload: userSelectionPayload(live, artworkId, crypto.randomUUID())};
      selectionSource = source;
      if (pending && userSelectionReceiptMatches(live, source.payload)) {render(live); return;}
      const artwork = live.artworks.find(artwork => artwork.id === artworkId);
      if (!artwork) throw new Error('所选版本不在当前任务中');
      const url = new URL(artwork.image_url, location.origin);
      if (url.origin !== location.origin) throw new Error('只能核对本地服务中的原图');
      const response = await fetch(url.href); if (!response.ok) throw new Error(`原图 HTTP ${response.status}`);
      const blob = await response.blob(), bytes = await blob.arrayBuffer();
      if (!globalThis.crypto?.subtle) throw new Error('请使用本地安全地址，以便核对所选原图');
      const digest = await crypto.subtle.digest('SHA-256', bytes);
      const sha = [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('');
      if (sha !== source.payload.artwork_sha256) throw new Error('原图与所读记录不一致，没有提交选择；请重新打开当前文件');
      const objectUrl = URL.createObjectURL(blob), image = new Image();
      try {image.src = objectUrl; await image.decode();} catch (error) {URL.revokeObjectURL(objectUrl); throw error;}
      if (sequence !== selectionSequence || id !== selectedId || !$('selectionDialog').open) {URL.revokeObjectURL(objectUrl); return;}
      const current = await read(`/api/iteration-runs/${encodeURIComponent(selectedId)}`);
      if (sequence !== selectionSequence || id !== selectedId || !$('selectionDialog').open) {URL.revokeObjectURL(objectUrl); return;}
      if (!pending && !userSelectionContextMatches(current, source)) {URL.revokeObjectURL(objectUrl); throw new Error('读取原图期间任务或评价变化，请重新打开');}
      selectionImageUrl = objectUrl; selectionImageVerified = true; $('selectionImage').src = objectUrl; $('selectionImage').hidden = false;
      $('selectionArtwork').textContent = `${artworkId} · ${image.naturalWidth} × ${image.naturalHeight} · 已核对原图 SHA256`;
      selectionSource = source; $('selectionStatus').textContent = '选择只表示你决定保留这一版，不把它认证为全部要求已完成。';
      render(current);
    } catch (error) {if (sequence === selectionSequence) $('selectionStatus').textContent = `没有提交选择：${error.message}`;}
    finally {if (sequence === selectionSequence) {selectionLoading = false; if (run && selectionSource) renderSelectionState(run);}}
  }
  $('closeSelection').onclick = () => {if (!selectionSaving) $('selectionDialog').close();};
  $('selectionDialog').addEventListener('cancel', event => {if (selectionSaving) event.preventDefault();});
  $('selectionDialog').addEventListener('close', () => {
    selectionSequence++; selectionSource = null; selectionImageVerified = false;
    if (selectionImageUrl) URL.revokeObjectURL(selectionImageUrl);
    selectionImageUrl = ''; $('selectionImage').removeAttribute('src');
  });
  $('recheckSelection').onclick = async () => {
    const source = selectionSource; if (!source || selectionSaving) return;
    $('selectionStatus').textContent = '只读取当前回执，不提交选择…';
    try {
      const value = await read(`/api/iteration-runs/${encodeURIComponent(source.runId)}`);
      if (source.runId === id && selectionSource === source && $('selectionDialog').open) render(value);
    } catch (error) {if (source.runId === id && selectionSource === source && $('selectionDialog').open) $('selectionStatus').textContent = `回执仍未确认，不会自动重发：${error.message}`;}
  };
  $('confirmSelection').onclick = async () => {
    const source = selectionSource; if (!source || selectionLoading || selectionSaving || $('confirmSelection').disabled) return;
    selectionSaving = true; $('confirmSelection').disabled = true; $('closeSelection').disabled = true;
    let attempted = false;
    try {
      const current = await read(`/api/iteration-runs/${encodeURIComponent(source.runId)}`);
      if (source.runId !== id || selectionSource !== source || !$('selectionDialog').open) throw new Error('当前查看的任务已切换，没有提交');
      if (userSelectionReceiptMatches(current, source.payload)) {render(current); return;}
      const pending = pendingSelection(source.runId);
      if (!pending && (!userSelectionContextMatches(current, source) || !userSelectionAvailability(current, source.payload.artwork_id).allowed)) throw new Error('任务、评价或原图已变化，没有提交；请重新打开');
      if (pending && pending.payload.request_id !== source.payload.request_id) throw new Error('已有其他选择的回执未确认，未提交');
      savePendingSelection(source);
      $('selectionStatus').textContent = '正在保存这次版本选择并关闭后续派发…'; attempted = true;
      const saved = await read(`/api/iteration-runs/${encodeURIComponent(source.runId)}/select`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(source.payload)});
      if (!userSelectionReceiptMatches(saved, source.payload)) throw new Error('返回记录不对应本次请求，需要核对回执');
      clearPendingSelection(source.runId); if (id === source.runId) render(saved);
    } catch (error) {
      // An explicit 4xx rejects this request. Network loss/5xx never imply failure
      // or trigger another POST; recover only by reading the durable receipt.
      if (attempted && error.status >= 400 && error.status < 500) clearPendingSelection(source.runId);
      let observed = null;
      if (attempted) {try {observed = await read(`/api/iteration-runs/${encodeURIComponent(source.runId)}`);} catch (_) { /* Keep the pending request. */ }}
      if (observed && id === source.runId) render(observed);
      if (!observed || !userSelectionReceiptMatches(observed, source.payload)) $('selectionStatus').textContent = pendingSelection(source.runId) ? `选择回执未确认，不会自动再次提交。请重查回执；如主动重送，会保留同一请求标识。${error.message}` : `未保存选择：${error.message}`;
    } finally {
      selectionSaving = false; $('closeSelection').disabled = false;
      if (run && $('selectionDialog').open && selectionSource) renderSelectionState(run);
      load();
    }
  };
  async function openEdit(artworkId) {
    if (!run || editSaving) return;
    const selectedId = id, sequence = ++editSequence;
    editSource = null; editLoading = true; editAttempted = false; editPainted = false; editDrawing = false; editPoint = null;
    $('editInstruction').value = ''; $('submitEdit').disabled = true; $('editStatus').textContent = '正在读取源版本与任务状态…';
    $('editArtwork').textContent = artworkId; $('editDialog').showModal();
    for (const canvas of [$('editImage'), $('editMask')]) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    try {
      const live = await read(`/api/iteration-runs/${encodeURIComponent(selectedId)}`);
      if (sequence !== editSequence || id !== selectedId || !$('editDialog').open) return;
      const availability = localEditAvailability(live, artworkId); if (!availability.allowed) throw new Error(availability.reason);
      const artwork = live.artworks.find(artwork => artwork.id === artworkId);
      const sourceUrl = new URL(artwork.image_url, location.href);
      if (sourceUrl.origin !== location.origin) throw new Error('源图不是本机服务的实际图片');
      const imageResponse = await fetch(sourceUrl.href, {cache: 'no-store'}); if (!imageResponse.ok) throw new Error(`源图 HTTP ${imageResponse.status}`);
      const blob = await imageResponse.blob(), bytes = await blob.arrayBuffer();
      if (!globalThis.crypto?.subtle) throw new Error('请使用本地安全地址打开网页，以便校验圈选源图');
      const digest = await crypto.subtle.digest('SHA-256', bytes), sourceHash = [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('');
      if (artwork.artwork_sha256 && artwork.artwork_sha256 !== sourceHash) throw new Error('源图像素已变化，请刷新后重新圈选');
      const image = new Image(), objectUrl = URL.createObjectURL(blob);
      try {image.src = objectUrl; await image.decode();} finally {URL.revokeObjectURL(objectUrl);}
      if (sequence !== editSequence || id !== selectedId || !$('editDialog').open) return;
      for (const canvas of [$('editImage'), $('editMask')]) {canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;}
      $('editImage').getContext('2d').drawImage(image, 0, 0);
      editSource = {runId: selectedId, artworkId, originalHash: live.original_sha256, feedbackSignature: live.feedback_signature, sourceHash};
      $('editArtwork').textContent = `${artworkId} · ${image.naturalWidth} × ${image.naturalHeight} · 当前局部执行：NewBie`;
      $('editStatus').textContent = '在左边涂出允许修改的区域，再说明你的意图。';
      render(live);
    } catch (error) {if (sequence === editSequence) $('editStatus').textContent = `无法准备局部修改：${error.message}`;}
    finally {if (sequence === editSequence) {editLoading = false; if (run) render(run);}}
  }
  $('closeEdit').onclick = () => {if (!editSaving) $('editDialog').close();};
  $('editDialog').addEventListener('close', () => {editSequence++; editSource = null; editDrawing = false; editPoint = null;});
  $('editDialog').addEventListener('cancel', event => {if (editSaving) event.preventDefault();});
  $('editBrush').oninput = () => $('editBrushSize').textContent = $('editBrush').value;
  const maskCanvas = $('editMask');
  function paintEditMask(event) {
    if (!editDrawing || !editSource || editLoading || editSaving || editAttempted) return;
    const rect = maskCanvas.getBoundingClientRect(), x = (event.clientX - rect.left) * maskCanvas.width / rect.width,
      y = (event.clientY - rect.top) * maskCanvas.height / rect.height, radius = Number($('editBrush').value) / 2;
    const ctx = maskCanvas.getContext('2d'); ctx.strokeStyle = ctx.fillStyle = '#fff'; ctx.lineWidth = radius * 2; ctx.lineCap = ctx.lineJoin = 'round';
    ctx.beginPath(); if (editPoint) {ctx.moveTo(...editPoint); ctx.lineTo(x, y); ctx.stroke();} else {ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fill();}
    editPoint = [x, y]; editPainted = true; if (run) $('submitEdit').disabled = !localEditAvailability(run, editSource.artworkId).allowed || run.feedback_signature !== editSource.feedbackSignature;
  }
  maskCanvas.onpointerdown = event => {if (editSource && !editSaving && !editAttempted) {editDrawing = true; editPoint = null; maskCanvas.setPointerCapture(event.pointerId); paintEditMask(event);}};
  maskCanvas.onpointermove = paintEditMask;
  maskCanvas.onpointerup = maskCanvas.onpointercancel = () => {editDrawing = false; editPoint = null;};
  $('clearEditMask').onclick = () => {if (!editSaving && !editAttempted) {maskCanvas.getContext('2d').clearRect(0, 0, maskCanvas.width, maskCanvas.height); editPainted = false; editPoint = null; $('submitEdit').disabled = true;}};
  $('editForm').onsubmit = async event => {
    event.preventDefault(); if (!editSource || editLoading || editSaving || editAttempted || !editPainted) return;
    const source = {...editSource}; editSaving = true; $('submitEdit').disabled = true; $('closeEdit').disabled = true;
    try {
      let live = await read(`/api/iteration-runs/${encodeURIComponent(source.runId)}`);
      if (source.runId !== id || live.original_sha256 !== source.originalHash || live.feedback_signature !== source.feedbackSignature) throw new Error('任务或原文评价已变化，请重新打开并审阅');
      const options = {instruction: $('editInstruction').value, mask_data_url: maskCanvas.toDataURL('image/png'), source_sha256: source.sourceHash, steps: $('editSteps').value, denoise: $('editDenoise').value};
      localEditDecision(live, source.artworkId, options);
      if (live.status === 'paused') {
        $('editStatus').textContent = '正在继续本任务，为本次修改取得派发资格…';
        await read(`/api/iteration-runs/${encodeURIComponent(source.runId)}/resume`, {method: 'POST'});
      }
      const context = await read(`/api/iteration-runs/${encodeURIComponent(source.runId)}/context`);
      if (source.runId !== id || !$('editDialog').open || context.original_sha256 !== source.originalHash || context.feedback_signature !== source.feedbackSignature) throw new Error('上下文或评价已变化，没有提交修改；请重新审阅');
      const decision = localEditDecision({...live, ...context, status: 'running'}, source.artworkId, options);
      const observationId = await prepareEditObservation(source, context);
      const current = await read(`/api/iteration-runs/${encodeURIComponent(source.runId)}/context`);
      if (source.runId !== id || !$('editDialog').open || current.original_sha256 !== source.originalHash || current.feedback_signature !== source.feedbackSignature || current.revision !== context.revision || current.decision_count !== context.decision_count) throw new Error('保存原件期间上下文已变化，没有提交修改；请重新审阅');
      decision.observation_id = observationId;
      $('editStatus').textContent = '正在提交这一项局部修改…'; editAttempted = true;
      await read(`/api/iteration-runs/${encodeURIComponent(source.runId)}/decisions`, {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({revision: context.revision, decision_count: context.decision_count, feedback_signature: context.feedback_signature, observation_id: observationId, decision})});
      sessionStorage.removeItem(observationStorageKey(source.runId, source.artworkId));
      $('editDialog').close(); await load();
    } catch (error) {$('editStatus').textContent = editAttempted ? `本次提交未确认，不会自动重试。请刷新任务查看结果：${error.message}` : `没有提交修改：${error.message}`;}
    finally {editSaving = false; $('closeEdit').disabled = false; if (run && $('editDialog').open) render(run);}
  };
  $('reviewText').oninput = () => reviewDrafts.set(reviewArtworkId, $('reviewText').value);
  $('reviewForm').onsubmit = async event => {
    event.preventDefault(); const artworkId = reviewArtworkId, text = $('reviewText').value.trim();
    if (!text || reviewSaving) return;
    reviewSaving = true; $('saveReview').disabled = true; $('reviewStatus').textContent = '正在保存原文…';
    try {
      const saved = await read(`/api/artworks/${encodeURIComponent(artworkId)}/reviews`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text})});
      const artwork = artworkById(artworkId);
      if (artwork && !(artwork.reviews || []).some(review => review.id === saved.id)) (artwork.reviews ||= []).push(saved);
      // Do not erase another artwork's draft if the user switched dialogs during the request.
      if (reviewDrafts.get(artworkId)?.trim() === text) reviewDrafts.delete(artworkId);
      if (reviewArtworkId === artworkId) {
        if ($('reviewText').value.trim() === text) $('reviewText').value = '';
        $('reviewStatus').textContent = '已保存原文，供后续审阅；不代表认可整张图。';
        $('reviewHistory').innerHTML = humanReviewsHtml(artwork);
      }
      if (run) render(run);
    } catch (error) { if (reviewArtworkId === artworkId) $('reviewStatus').textContent = `未保存：${error.message}`; }
    finally {reviewSaving = false; $('saveReview').disabled = false; load();}
  };
  $('runs').onchange = () => {
    id = $('runs').value; run = null; compareIds.clear(); $('reviewDialog').close(); $('compareDialog').close(); $('editDialog').close(); $('selectionDialog').close();
    history.replaceState(null, '', id ? `/iteration?id=${encodeURIComponent(id)}` : '/iteration'); graphKey = ''; load();
  };
  $('refresh').onclick = () => { list(); load(); };
  $('planningMode').onclick = async () => {
    if(!id||busy||editLoading||editSaving||selectionSaving)return;
    const selectedId=id;
    // Uncertain POSTs are checked only by this explicit read gesture. Poll and
    // refresh never repeat a mode mutation or restart the task.
    if(planningAttempt?.runId===selectedId){
      busy=true;if(run)render(run);
      try{
        const current=await read(`/api/iteration-runs/${encodeURIComponent(selectedId)}`);
        if(selectedId!==id)return;
        if(current.id!==selectedId||current.original_sha256!==planningAttempt.originalHash||typeof current.autonomous_planning!=='boolean'||!Number.isSafeInteger(current.revision)||current.revision<0)throw new Error('服务端任务、原句、模式或版本不一致，回执仍待核对');
        const matched=planningModeReceiptMatches(current,planningAttempt);planningAttempt=null;render(current);
        $('planningModeStatus').textContent=matched?'已核对切换后的实际模式；任务仍暂停，没有自动继续。':`已读取当前实际模式：${current.autonomous_planning?'自动提案':'手动决策'}。不冒称原请求成功，也没有再次切换。`;
      }catch(error){if(selectedId===id)$('planningModeStatus').textContent=`模式仍未确认，不会重发：${error.message}`;}
      finally{busy=false;load();}
      return;
    }
    let source;
    try{source={runId:selectedId,originalHash:run.original_sha256,payload:planningModePayload(run)};}
    catch(error){$('planningModeStatus').textContent=error.message;return;}
    busy=true;if(run)render(run);let attempted=false;
    try{
      const current=await read(`/api/iteration-runs/${encodeURIComponent(selectedId)}`);
      if(selectedId!==id||current.original_sha256!==source.originalHash||current.revision!==source.payload.revision||current.autonomous_planning===source.payload.autonomous_planning)throw new Error('任务、模式或版本已变化，请查看当前状态后重新选择；没有切换');
      const available=planningModeAvailability(current);if(!available.allowed)throw new Error(available.reason);
      planningAttempt=source;attempted=true;
      const saved=await read(`/api/iteration-runs/${encodeURIComponent(selectedId)}/planning`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(source.payload)});
      if(!planningModeReceiptMatches(saved,source))throw new Error('返回状态不对应本次切换，需要读取核对');
      planningAttempt=null;if(selectedId===id){render(saved);$('planningModeStatus').textContent=`已切为${saved.autonomous_planning?'自动提案':'手动决策'}，任务仍暂停；需要你另点继续，不会自动启动。`;}
    }catch(error){
      if(attempted&&error.status>=400&&error.status<500&&error.status!==408)planningAttempt=null;
      if(selectedId===id)$('planningModeStatus').textContent=planningAttempt?`切换回执未确认，不会重发；请点“重查模式（只读取）”。${error.message}`:`没有切换：${error.message}`;
    }finally{busy=false;load();}
  };
  for (const action of ['pause', 'resume', 'cancel']) $(action).onclick = async () => {
    if (!id || busy) return;
    if (action === 'cancel' && !confirm('取消后续执行？已生成的版本会保留。')) return;
    busy = true; const selectedId = id; if (run) render(run);
    try {const value = await read(`/api/iteration-runs/${encodeURIComponent(selectedId)}/${action}`, {method: 'POST'}); if (selectedId === id) render(value);}
    catch (error) {if (selectedId === id) $('connection').textContent = error.message;}
    finally {busy = false; load();}
  };
  list(); load();
})();
