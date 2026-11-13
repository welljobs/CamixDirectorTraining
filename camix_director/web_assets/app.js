'use strict';
const $ = id => document.getElementById(id);
const state = {snapshot:null,selectedJob:'',config:null,pack:null,previewUrl:null,audioUrl:null,browseTarget:null,browseMode:'directory',browsePath:'data',polling:false,reportPath:null};
const fragment = new URLSearchParams(location.hash.slice(1));
const freshToken = fragment.get('token');
if (freshToken) { sessionStorage.setItem('camix-local-token',freshToken); history.replaceState(null,'',location.pathname); }
const token = sessionStorage.getItem('camix-local-token') || '';
const terminal = new Set(['completed','failed','cancelled','interrupted']);
const statusNames = {queued:'准备中',running:'运行中',completed:'已完成',failed:'失败',cancelled:'已停止',interrupted:'连接中断'};
const kindNames = {train:'模型训练',evaluate:'独立评估',prepare:'素材准备',download_ami:'AMI 示例准备',export_coreml:'Core ML 导出',smoke:'合成样例生成',inspect:'数据质检',quality:'完整质量检查',freeze:'冻结数据版本',replay:'离线闭环回放',raw_demo:'合成原片演示',gate:'研究候选验收检查',proxy:'浏览器代理制作'};
const phaseNames = {early_stopped:'验证指标触发早停，最佳权重已保存',loading:'正在载入与校验数据',training:'正在更新模型参数',validation:'正在计算验证指标',epoch_complete:'本轮完成',completed:'训练完成',cancelled:'任务已停止'};
const bytes = n => Number.isFinite(n) ? (n<1073741824?`${(n/1048576).toFixed(1)} MiB`:`${(n/1073741824).toFixed(1)} GiB`) : '未读取';
const percent = n => Number.isFinite(n) ? `${(n*100).toFixed(1)}%` : '—';
const loss = n => Number.isFinite(n) ? n.toFixed(4) : '—';
const esc = s => String(s ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function error(message) { $('global-error').textContent=String(message); $('global-error').hidden=false; }
function clearError() { $('global-error').hidden=true; }
let toastTimer;
function toast(message) { $('toast').textContent=message;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,4500); }
async function api(path,body) {
  const response=await fetch(path,{method:body?'POST':'GET',headers:{'X-Camix-Token':token,...(body?{'Content-Type':'application/json'}:{})},...(body?{body:JSON.stringify(body)}:{})});
  if (!response.ok) { const result=await response.json().catch(()=>({error:`HTTP ${response.status}`})); throw new Error(result.error); }
  return response;
}
async function json(path,body) { return (await api(path,body)).json(); }
function config() { const value=JSON.parse($('config-json').value);if (!value || Array.isArray(value) || typeof value!=='object') throw new Error('模型配置应为 JSON 对象');return value; }
function option(select,value,label) { const element=document.createElement('option');element.value=value;element.textContent=label;select.append(element);return element; }
function replaceOptions(select,items,placeholder) {
  const previous=select.value;select.replaceChildren();
  if (!items.length) option(select,'',placeholder || '暂无选项');
  for (const item of items) option(select,item.value,item.label);
  if (items.some(x=>x.value===previous)) select.value=previous;
}
function tab(name) {
  for (const button of document.querySelectorAll('[data-tab]')) button.classList.toggle('active',button.dataset.tab===name);
  for (const panel of document.querySelectorAll('.tab-panel')) panel.hidden=panel.id!==`tab-${name}`;
  const titles={train:['模型训练','让下一次训练，清晰可见。','配置多机位模型，观察学习过程，保留每一次实验。'],data:['数据与预览','从可靠的数据，开始学习。','检查机位与时钟，区分真实监督、弱标签与合成样例。'],evaluate:['评估与导出','先验证，再迈出下一步。','分离测试场次，检验模态贡献，导出可追溯的模型。'],annotate:['审片与标注','把导演判断，变成可信的数据。','多机位统一时间轴，点击标注，审核后生成版本化数据。'],quality:['数据质量与版本','数据可靠，实验才有依据。','检查泄漏和标签覆盖，冻结不可变的数据与标注版本。'],experiments:['实验与回放','让每个模型，都有清晰的来路。','对照实验、查看血缘，在离线闭环中观察模型建议。']};
  document.querySelector('.content-grid').classList.toggle('studio-wide',['annotate','experiments'].includes(name));
  $('breadcrumb').textContent=titles[name][0];$('page-title').textContent=titles[name][1];$('page-subtitle').textContent=titles[name][2];
}
function renderCatalog(catalog) {
  if (!$('preset').options.length) {
    for (const item of catalog.configs) option($('preset'),item.name,item.name.startsWith('pilot')?'轻量 · 2机位 / 64px / 8步记忆':item.name.startsWith('standard')?'扩容起点 · 4机位（未验收）':item.name);
    const first=catalog.configs.find(x=>x.name==='pilot-config') || catalog.configs[0];
    if (first) { $('preset').value=first.name;$('config-json').value=JSON.stringify(first.config,null,2); }
  }
  $('datasets').replaceChildren();
  for (const dataset of catalog.datasets) option($('datasets'),dataset.path,dataset.path);
  if (!$('train-data').value) { const target=catalog.datasets.find(x=>x.path==='data/ami_pilot/prepared') || catalog.datasets.find(x=>x.splits.train&&x.splits.validation);if(target)$('train-data').value=target.path; }
  if (!$('eval-data').value) { const target=catalog.datasets.find(x=>x.path==='data/ami_fresh_holdout/prepared') || catalog.datasets.find(x=>x.splits.test);if(target)$('eval-data').value=target.path; }
  const previousCheckpoint=$('checkpoint').value;
  replaceOptions($('checkpoint'),catalog.checkpoints.map(x=>({value:x.path,label:`${x.name} · ${x.qualification}`})),'暂无训练权重');
  if(!previousCheckpoint&&catalog.recommended_checkpoint)$('checkpoint').value=catalog.recommended_checkpoint;
  updateReportOptions();
  const packs=catalog.datasets.flatMap(x=>x.packs);
  replaceOptions($('pack'),packs.map(x=>({value:x.path,label:`${x.metadata.session_id} · ${x.metadata.split} · ${x.metadata.source}`})),'暂无数据包');
  if (state.pack?.path!==$('pack').value) loadPack();
  $('dataset-cards').innerHTML=catalog.datasets.length?catalog.datasets.map(d=>`<div class="dataset-entry"><code>${esc(d.path)}</code><p>TRAIN ${d.splits.train}　VALIDATION ${d.splits.validation}　TEST ${d.splits.test} · 展开约 ${bytes(d.raw_bytes)}</p><button class="button small ghost use-dataset" data-path="${esc(d.path)}">用于训练配置</button></div>`).join(''):'<p class="muted">暂无数据。可在下方生成合成样例检查链路，或准备真实素材。</p>';
  for(const button of document.querySelectorAll('.use-dataset'))button.onclick=()=>{$('train-data').value=button.dataset.path;updateDataNote();tab('train');};
  updateDataNote();
  window.dispatchEvent(new CustomEvent('camix-catalog',{detail:catalog}));
}
function updateDataNote() {
  const data=state.snapshot?.catalog.datasets.find(x=>x.path===$('train-data').value);
  const readiness=$('dataset-readiness');
  if(readiness)readiness.textContent=data?(data.packs.some(p=>p.metadata.source==='synthetic')?'这批数据含合成素材：需明确勾选合成确认，仅验证链路。':data.packs.some(p=>p.metadata.source==='real'&&!p.metadata.human_reviewed)?'这批数据含弱标签：开始前请勾选下方“我确认使用弱监督标签”。不需要把人审字段改成true。':'请检查审核证据和独立场次划分；有已审核标志不代表已经通过产品验收。'):'先选择数据目录，再检查标签与配置。';
  $('train-data-note').textContent=data?`${data.packs.length} 个片段 · 解压约 ${bytes(data.raw_bytes)}。按需读取并缓存到磁盘；批次和模型占用设备内存。`:'需同时包含 train 和 validation，按原始场次隔离；外部素材请先准备为数据包。';
}
function updateReportOptions() {
  const cp=state.snapshot?.catalog.checkpoints.find(x=>x.path===$('checkpoint').value);
  replaceOptions($('existing-report'),(cp?.reports||[]).map(p=>({value:p,label:p.includes('fresh')?'新留出评估 · '+p.split('/').slice(-2,-1)[0]:p.split('/').slice(-2).join('/')})),'该权重暂无已有评估报告');
}
function renderSystem(snapshot) {
  const s=snapshot.system;
  $('os-value').textContent=`${s.os} · ${s.machine}`;$('ram-value').textContent=bytes(s.ram_bytes);$('disk-value').textContent=bytes(snapshot.free_disk_bytes);
  const enabled=s.devices.filter(x=>x.available&&x.id!=='cpu');$('gpu-value').textContent=s.gpu_name || (enabled.map(x=>x.id.toUpperCase()).join(' / ') || 'CPU 可用');
  $('platform-label').textContent=`Python ${s.python} · PyTorch ${s.torch}`;$('workspace-path').textContent='工作区：'+snapshot.workspace;
  if (!$('device').options.length)for(const device of s.devices){const item=option($('device'),device.id,device.label+(device.available?'':' · 不可用'));item.disabled=!device.available;}
  $('export-coreml').disabled=!s.coreml_export_available;
  $('export-note').textContent=s.coreml_export_available?'当前为 Mac：可进行 Core ML 转换与 CPU 数值验证。结果不代表 iPad 实时性能。':'当前平台仅启用训练与评估。请下载 .pt 权重，复制到 Mac 后导出 Core ML。';
}
function drawChart(history) {
  const svg=$('loss-chart');svg.replaceChildren();const ns='http://www.w3.org/2000/svg';
  const add=(name,attrs,text)=>{const node=document.createElementNS(ns,name);for(const [key,value]of Object.entries(attrs))node.setAttribute(key,value);if(text)node.textContent=text;svg.append(node);return node;};
  add('path',{d:'M30 15V125H350 M30 70H350 M30 15H350',class:'grid-line'});
  if(!history.length){add('text',{x:110,y:78,class:'chart-empty'},'完成首轮后显示曲线');return;}
  const values=history.flatMap(x=>[x.train_loss,x.validation_loss]).filter(Number.isFinite);const maximum=Math.max(...values,.001)*1.1;
  for(const [key,color]of [['train_loss','#138a81'],['validation_loss','#4e7dd5']]) {
    const points=history.filter(x=>Number.isFinite(x[key])).map((x,i)=>`${30+i/Math.max(history.length-1,1)*320},${125-x[key]/maximum*110}`).join(' ');
    add('polyline',{points,fill:'none',stroke:color,'stroke-width':2,'stroke-linejoin':'round'});
  }
  add('text',{x:2,y:19,class:'chart-tick'},maximum.toFixed(2));add('text',{x:13,y:127,class:'chart-tick'},'0');
  add('text',{x:29,y:143,class:'chart-tick'},'1');add('text',{x:335,y:143,class:'chart-tick'},String(history.at(-1).epoch));
}
function renderJob(job) {
  if(!job) {$('stop-job').disabled=true;$('task-strip').hidden=true;return;}
  $('task-strip').hidden=false;$('task-strip-title').textContent=(kindNames[job.kind]||job.kind)+' · '+(statusNames[job.status]||job.status);$('task-strip-output').textContent=job.output_relative||job.id;
  $('job-status').textContent=job.stop_requested&&!terminal.has(job.status)?'正在停止':statusNames[job.status]||job.status;
  $('job-status').className='pill '+job.status;$('job-title').textContent=kindNames[job.kind]||job.kind;
  const progress=job.progress;const history=progress?.history || job.training_report?.history || [];
  const last=history.at(-1);
  $('job-phase').textContent=job.connection_warning || (job.status==='failed'?job.error:null) || (progress?`${phaseNames[progress.phase]||progress.phase}${progress.epochs?` · 第 ${progress.epoch}/${progress.epochs} 轮`:''}${progress.batches?` · 批次 ${progress.batch}/${progress.batches}`:''}`:terminal.has(job.status)?'任务已结束。请查看结果与完整日志。':'独立进程正在工作，输出会自动刷新。');
  const elapsed=progress?.elapsed_seconds || (job.started_at?(Date.parse(job.finished_at||new Date().toISOString())-Date.parse(job.started_at))/1000:0);
  $('job-elapsed').textContent=elapsed>0?`${Math.floor(elapsed/60)}分 ${Math.floor(elapsed%60)}秒`:job.id;
  let percentage=job.status==='completed'?100:0;
  if(progress&&progress.epochs){percentage=(Math.max(0,progress.epoch-1)+(progress.phase==='training'?progress.batch/Math.max(1,progress.batches)*.8:progress.phase==='validation'?.8+progress.batch/Math.max(1,progress.batches)*.2:progress.phase==='epoch_complete'?1:0))/progress.epochs*100;}
  if(job.status==='completed')percentage=100;$('job-progress').value=Math.max(0,Math.min(100,percentage));
  $('train-loss').textContent=loss(last?.train_loss ?? progress?.train_loss);$('val-loss').textContent=loss(last?.validation_loss);$('accuracy').textContent=percent(last?.validation_pair_accuracy);
  const resource=progress?.resources;
  $('resource-summary').textContent=resource?`进程CPU ${resource.worker_cpu_percent?.toFixed(0)||'—'}% · RSS峰值 ${bytes(resource.rss_peak_bytes)} · GPU ${bytes(resource.gpu_allocated_bytes)}${progress.learning_rate?' · LR '+progress.learning_rate:''}${progress.windows_per_second?' · '+progress.windows_per_second.toFixed(1)+'窗口/s':''}${Number.isFinite(progress.eta_seconds)?' · 估算剩余 '+Math.ceil(progress.eta_seconds/60)+' 分钟':''}`:'资源指标将在训练时显示；GPU吞吐未在本机CUDA验证';
  drawChart(history);$('job-output').textContent=job.output_relative||'—';$('stop-job').disabled=terminal.has(job.status)||!!job.stop_requested;
  $('job-checkpoint').hidden=!job.checkpoint;$('job-checkpoint').dataset.path=job.checkpoint||'';
  if(job.evaluation)renderEvaluation(job.evaluation);
  window.dispatchEvent(new CustomEvent('camix-job',{detail:job}));
  if(job.export_report)$('export-result').innerHTML=`<div class="notice"><strong>${job.export_report.parity_passed?'数值对齐通过':'数值对齐未通过'}</strong><p>最大绝对差 ${esc(job.export_report.maximum_absolute_difference)} · 结果位于 ${esc(job.output_relative)}。请从该目录复制 .mlpackage。</p></div>`;
}
function renderEvaluation(report) {
  const labels={full:'完整模型',memory_reset_each_step:'每步清空记忆',audio_muted:'静音音频',video_content_muted:'抹掉画面内容',one_camera_disconnected:'单机位断开',all_cameras_disconnected:'全部视频断开（只检查数值安全）'};
  const rows=(report.evaluations||[]).map(r=>`<tr><td>${esc(r.session_id)}<br>${esc(labels[r.ablation]||r.ablation)}</td><td>${percent(r.pair_accuracy)}</td><td>${percent(r.balanced_keep_take_accuracy)}</td><td>${esc(r.pairs)}${r.finite_output_ratio!==undefined?'<br>有限输出 '+percent(r.finite_output_ratio):''}</td></tr>`);
  for(const b of report.baselines||[])rows.push(`<tr><td>${esc(b.session_id)}<br>简单 PCM-RMS</td><td>${percent(b.pair_accuracy)}</td><td>${percent(b.balanced_keep_take_accuracy)}</td><td>${esc(b.pairs)}</td></tr>`);
  $('evaluation-result').innerHTML=`${report.summary?`<div class="notice">独立场次 ${report.summary.independent_sessions} · 场次宏平均 ${percent(report.summary.macro_session_pair_accuracy)} · ${esc(report.summary.confidence_warning)}</div>`:''}<table class="eval-table"><thead><tr><th>条件 / 场次</th><th>排序准确率</th><th>平衡准确率</th><th>比较数</th></tr></thead><tbody>${rows.join('')}</tbody></table><p class="muted">弱标签排序，不是专业导播准确率；相邻时刻相关。Mac CPU 单步耗时不是 iPad 端到端延时。</p>`;
}
async function refresh(force=false) {
  if(state.polling)return;state.polling=true;
  try {
    const snapshot=await json('/api/state'+(force?'?refresh=1':''));state.snapshot=snapshot;
    $('connection').textContent='● 本地已连接';$('connection').className='pill';renderSystem(snapshot);renderCatalog(snapshot.catalog);
    if(!state.selectedJob&&snapshot.jobs.length)state.selectedJob=snapshot.jobs[0].id;
    replaceOptions($('job-picker'),snapshot.jobs.map(j=>({value:j.id,label:`${kindNames[j.kind]||j.kind} · ${statusNames[j.status]||j.status} · ${j.id}`})),'尚无网页任务');
    if(state.selectedJob)$('job-picker').value=state.selectedJob;
    const current=snapshot.jobs.find(j=>j.id===state.selectedJob);renderJob(current);
    const running=snapshot.jobs.some(j=>!terminal.has(j.status));
    for(const id of ['start-train','make-smoke','download-ami'])$(id).disabled=running;
    $('export-coreml').disabled=running||!snapshot.system.coreml_export_available;
    if(current){const log=await json('/api/logs?id='+encodeURIComponent(current.id));const pre=$('logs');if(pre.textContent!==log.text){pre.textContent=log.text||'任务已排队，等待日志…';if($('auto-scroll').checked)pre.scrollTop=pre.scrollHeight;}}
  } catch(e) {$('connection').textContent='连接异常';$('connection').className='pill failed';error(e.message);}
  finally {state.polling=false;}
}
async function start(payload) {
  clearError();const job=await json('/api/jobs',payload);state.selectedJob=job.id;toast('任务已启动，可从顶部任务条查看进度与日志');await refresh(true);return job;
}
async function download(path) {
  if(!path)throw new Error('请先选择模型或已生成的文件');
  const response=await api('/api/artifact?path='+encodeURIComponent(path));const url=URL.createObjectURL(await response.blob());
  const link=document.createElement('a');link.href=url;link.download=path.split('/').at(-1);document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),5000);
}
async function loadPack() {
  const selected=state.snapshot?.catalog.datasets.flatMap(d=>d.packs).find(p=>p.path===$('pack').value);state.pack=selected||null;
  if(!selected){$('preview').hidden=true;$('preview-empty').hidden=false;return;}
  $('pack-meta').textContent=JSON.stringify(selected.metadata,null,2);$('tick').max=String(selected.steps-1);$('tick').value='0';
  const n=selected.metadata.config?.max_cameras||2;replaceOptions($('audio-camera'),Array.from({length:n},(_,i)=>({value:String(i),label:`机位 ${i+1}`})));
  await loadPreview();
}
async function loadPreview() {
  if(!state.pack)return;const tick=Number($('tick').value);$('tick-label').textContent=`${tick} / ${state.pack.steps-1}`;
  try {const r=await api(`/api/preview?path=${encodeURIComponent(state.pack.path)}&tick=${tick}`);const blob=await r.blob();if(state.previewUrl)URL.revokeObjectURL(state.previewUrl);state.previewUrl=URL.createObjectURL(blob);$('preview').src=state.previewUrl;$('preview').hidden=false;$('preview-empty').hidden=true;}
  catch(e){$('preview').hidden=true;$('preview-empty').hidden=false;$('preview-empty').textContent=e.message;}
}
async function browse(path='data') {
  const result=await json('/api/browse?path='+encodeURIComponent(path));state.browsePath=result.path;$('browser-path').textContent=result.path||'/';
  $('browser-entries').innerHTML=result.entries.map(x=>`<button class="browser-entry" data-path="${esc(x.path)}" data-directory="${x.directory}">${x.directory?'▸':'·'} ${esc(x.name)}</button>`).join('');
  for(const button of document.querySelectorAll('.browser-entry'))button.onclick=()=>guard(async()=>{if(button.dataset.directory==='true')await browse(button.dataset.path);else if(state.browseMode==='file'){ $(state.browseTarget).value=button.dataset.path;$('browser-dialog').close();}});
  $('choose-directory').hidden=state.browseMode!=='directory';
}
async function guard(work) {try{await work();}catch(e){error(e.message);}}
for(const button of document.querySelectorAll('[data-tab]'))button.onclick=()=>tab(button.dataset.tab);
$('refresh').onclick=()=>refresh(true);$('refresh-data').onclick=()=>refresh(true);$('train-data').oninput=updateDataNote;
$('preset').onchange=()=>{const preset=state.snapshot?.catalog.configs.find(x=>x.name===$('preset').value);if(preset)$('config-json').value=JSON.stringify(preset.config,null,2);};
$('train-form').onsubmit=event=>{event.preventDefault();guard(async()=>{
  const payload={kind:'train',data:$('train-data').value.trim(),config:config(),device:$('device').value,epochs:Number($('epochs').value),batch_size:Number($('batch').value),lr:Number($('lr').value),sequence_steps:Number($('sequence').value),burn_in:Number($('burn').value),seed:Number($('seed').value),allow_weak:$('weak').checked,allow_synthetic:$('synthetic').checked,balance_actions:$('balance').checked,ack_memory:$('ack-memory').checked,mode:$('train-mode').value,parent_checkpoint:$('train-parent').value,scheduler:$('scheduler').value,patience:Number($('patience').value),accumulation_steps:Number($('accumulation').value),checkpoint_steps:Number($('checkpoint-steps').value),precision:$('precision').value,num_workers:Number($('num-workers').value)};
  const check=await json('/api/studio/preflight',payload);const panel=$('preflight-results');panel.hidden=false;
  panel.className=check.passed?'quality-success':'quality-errors';
  panel.innerHTML=check.passed?'快速字段检查通过。启动时还会验证样本内容；正式实验建议先完成质量检查并冻结版本。':`<strong>请先处理以下问题（尚未启动训练）</strong><ul>${check.errors.map(e=>`<li><button type="button" class="preflight-link" data-field="${esc(e.field)}">${esc(e.message)}</button></li>`).join('')}</ul>`;
  for(const link of panel.querySelectorAll('[data-field]'))link.onclick=()=>{const target=$(link.dataset.field);target?.scrollIntoView({block:'center',behavior:'smooth'});target?.focus();};
  if(!check.passed){clearError();panel.scrollIntoView({block:'center',behavior:'smooth'});return;}
  await start(payload);
});};
$('eval-form').onsubmit=event=>{event.preventDefault();guard(()=>start({kind:'evaluate',data:$('eval-data').value.trim(),checkpoint:$('checkpoint').value,split:$('split').value,allow_weak:$('eval-weak').checked}));};
$('prepare-form').onsubmit=event=>{event.preventDefault();guard(()=>start({kind:'prepare',manifest:$('manifest').value.trim(),output:$('prepare-output').value.trim(),config:config()}));};
$('download-ami').onclick=()=>guard(()=>start({kind:'download_ami',config:config(),accept_license:$('ami-license').checked,seconds:45}));
$('make-smoke').onclick=()=>guard(()=>start({kind:'smoke',config:config(),ack_synthetic:$('ack-smoke').checked}));
$('export-coreml').onclick=()=>guard(()=>start({kind:'export_coreml',checkpoint:$('checkpoint').value}));
$('checkpoint').onchange=updateReportOptions;
$('load-report').onclick=()=>guard(async()=>{const path=$('existing-report').value;if(!path)throw new Error('该模型还没有已有评估报告，可先运行评估');renderEvaluation(await (await api('/api/artifact?path='+encodeURIComponent(path))).json());});
$('download-checkpoint').onclick=()=>guard(()=>download($('checkpoint').value));
$('job-checkpoint').onclick=event=>{event.preventDefault();guard(()=>download(event.currentTarget.dataset.path));};
$('job-picker').onchange=()=>{state.selectedJob=$('job-picker').value;refresh();};
$('stop-job').onclick=()=>{if(!state.selectedJob)return;state.stopTarget=state.selectedJob;$('stop-dialog').showModal();};
$('cancel-stop').onclick=()=>$('stop-dialog').close();
$('confirm-stop').onclick=()=>guard(async()=>{const id=state.stopTarget;$('stop-dialog').close();await json('/api/stop',{id});toast('已请求停止；请等待任务退出');refresh();});
$('pack').onchange=()=>guard(loadPack);let previewTimer;$('tick').oninput=()=>{clearTimeout(previewTimer);previewTimer=setTimeout(loadPreview,180);};
$('listen').onclick=()=>guard(async()=>{if(!state.pack)throw new Error('请先选择数据包');const r=await api(`/api/audio?path=${encodeURIComponent(state.pack.path)}&tick=${$('tick').value}&camera=${$('audio-camera').value}`);if(state.audioUrl)URL.revokeObjectURL(state.audioUrl);state.audioUrl=URL.createObjectURL(await r.blob());$('audio').src=state.audioUrl;$('audio').hidden=false;await $('audio').play();});
for(const button of document.querySelectorAll('.browse'))button.onclick=()=>guard(async()=>{state.browseTarget=button.dataset.target;state.browseMode=button.dataset.mode;await browse('data');$('browser-dialog').showModal();});
$('browser-up').onclick=()=>guard(()=>browse(state.browsePath.split('/').slice(0,-1).join('/')||'.'));
$('close-browser').onclick=()=>$('browser-dialog').close();$('choose-directory').onclick=()=>{ $(state.browseTarget).value=state.browsePath;$('browser-dialog').close();updateDataNote();};
refresh();setInterval(()=>refresh(),1800);

window.CamixConsole={start,tab,refresh};

$("task-strip-logs").onclick=()=>tab("train");
