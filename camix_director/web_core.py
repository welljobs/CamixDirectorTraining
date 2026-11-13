"""Workspace-scoped local GUI backend; all compute runs in a separate process."""
import ctypes
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile

import numpy as np
from .config import ModelConfig
from .runtime import atomic_json

CODE_ROOT = Path(__file__).resolve().parent.parent
TERMINAL = {'completed', 'failed', 'cancelled', 'interrupted'}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def scoped(root, value, exists=False):
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError('请选择本工程内的有效路径')
    root = Path(root).resolve()
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError('路径不能离开工作区；请先把素材放入工作区 data 目录')
    if exists and not candidate.exists():
        raise ValueError('路径不存在：' + value)
    return candidate


def relative(root, path):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def read_json(path, default=None, limit=4*1024*1024):
    try:
        if Path(path).stat().st_size > limit:
            return default
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def pack_info(path, root):
    """Read metadata only; do not decompress all RGB/PCM on the HTTP thread."""
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo('metadata.npy')
        if member.file_size > 1024*1024:
            raise ValueError('metadata exceeds 1 MiB')
        metadata = json.loads(str(np.load(io.BytesIO(archive.read(member)), allow_pickle=False).item()))
        raw_bytes = sum(x.file_size for x in archive.infolist())
        with archive.open('frames.npy') as source:
            version = np.lib.format.read_magic(source)
            shape, _, _ = np.lib.format._read_array_header(source, version)
    return dict(path=relative(root, path), name=Path(path).name, metadata=metadata,
                steps=int(shape[0]), modified_at=Path(path).stat().st_mtime, compressed_bytes=Path(path).stat().st_size, raw_bytes=raw_bytes)


def ram_bytes():
    if os.name == 'nt':
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(x, ctypes.c_ulonglong) for x in
                ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
        status = Memory();status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.total
    else:
        try:
            return os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        except (ValueError, OSError):
            if platform.system() == 'Darwin':
                try:
                    result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True, timeout=3)
                    return int(result.stdout.strip())
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
    return None


def system_info():
    import torch
    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
    return dict(os=platform.system(), release=platform.release(), machine=platform.machine(),
        python=platform.python_version(), torch=torch.__version__, cpu_threads=os.cpu_count(), ram_bytes=ram_bytes(),
        devices=[dict(id='cpu', label='CPU · 通用', available=True),
                 dict(id='cuda', label='CUDA · NVIDIA GPU', available=cuda),
                 dict(id='mps', label='MPS · Apple GPU', available=mps)],
        gpu_name=torch.cuda.get_device_name(0) if cuda else None, cuda_runtime=torch.version.cuda,
        coreml_export_available=platform.system() == 'Darwin',
        qualification='experimental_only', production_approved=False)


def integer(data, key, default, minimum, maximum):
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f'{key} 必须是 {minimum}～{maximum} 的整数')
    return value


def flag(data, key):
    value = data.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f'{key} 必须是布尔值')
    return value


def checked_config(payload):
    if not isinstance(payload, dict):
        raise ValueError('模型配置必须是 JSON 对象')
    config = ModelConfig(**payload)
    limits = {'max_cameras': (1, 8), 'image_size': (16, 512), 'audio_samples': (256, 64000),
              'hidden_dim': (8, 1024), 'heads': (1, 16), 'layers': (1, 16), 'memory_tokens': (1, 256),
              'sample_rate': (8000, 48000)}
    for name, (lo, hi) in limits.items():
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
            raise ValueError(f'网页模式下 {name} 的范围是 {lo}～{hi}')
    if not math.isfinite(config.decision_hz) or not 0 < config.decision_hz <= 30:
        raise ValueError('decision_hz 必须在 (0,30]')
    if not isinstance(config.pretrained_visual, bool):
        raise ValueError('pretrained_visual 必须是布尔值')
    return config


class JobManager:
    def __init__(self, workspace, code_root=CODE_ROOT, system=None):
        self.root = Path(workspace).resolve();self.code_root = Path(code_root).resolve()
        self.state = self.root / '.web';self.jobs_dir = self.state / 'jobs';self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.system = system or system_info();self.lock = threading.RLock();self.active = None
        self.catalog_cache = None;self.catalog_at = 0

    def catalog(self, force=False):
        if self.catalog_cache and not force and time.monotonic()-self.catalog_at < 8:
            return self.catalog_cache
        configs=[]
        for path in sorted((self.code_root/'examples').glob('*config.json')):
            value=read_json(path)
            if value:
                configs.append(dict(name=path.stem,config=value))
        datasets=[];errors=[];groups={}
        data_dir=self.root/'data'
        # One-level/short-episode datasets are expected; cap scanning for large projects.
        for index,path in enumerate(sorted(data_dir.rglob('*.npz')) if data_dir.exists() else []):
            if index>=500:
                errors.append('仅列出前500个数据包；请使用单独工作区或分片目录');break
            try:
                scoped(self.root,str(path),True)
                info=pack_info(path,self.root)
                groups.setdefault(relative(self.root,path.parent),[]).append(info)
            except Exception as error:
                errors.append(f'{path.name}: {error}')
        for path,packs in groups.items():
            datasets.append(dict(path=path,packs=packs,raw_bytes=sum(x['raw_bytes'] for x in packs),
                splits={split:sum(x['metadata'].get('split')==split for x in packs) for split in ('train','validation','test')}))
        from .storage import dataset_paths
        for manifest_path in sorted((self.root/'datasets/versions').glob('*/dataset-manifest.json')):
            manifest=read_json(manifest_path,{}) or {}
            try:
                packs=[pack_info(p,self.root) for p in dataset_paths(manifest_path.parent,verify=False)]
                datasets.append(dict(path=relative(self.root,manifest_path.parent),version_id=manifest.get('id'),name=manifest.get('name'),packs=packs,raw_bytes=sum(p['raw_bytes'] for p in packs),splits={part:sum(p['metadata']['split']==part for p in packs) for part in ('train','validation','test')}))
            except Exception as error:errors.append(f'Frozen dataset: {error}')
        checkpoints=[]
        for path in sorted([p for name in ('best.pt','last.pt') for p in (self.root/'runs').rglob(name)] if (self.root/'runs').exists() else [],key=lambda p:p.stat().st_mtime,reverse=True)[:150]:
            try:
                scoped(self.root,str(path),True)
                report=read_json(path.parent/'training-report.json',{}) or {}
                checkpoints.append(dict(path=relative(self.root,path),name=path.parent.name+(' · 完整恢复' if path.name=='last.pt' else ' · 最佳权重'),bytes=path.stat().st_size,
                    qualification=report.get('qualification',report.get('status','unqualified')),production_approved=False,
                    reports=[relative(self.root,r) for r in sorted(path.parent.rglob('evaluation.json'))[:12]]))
            except ValueError:
                continue
        recommended=None
        pointer=self.root/'runs/LATEST_AMI_RUN.txt'
        if pointer.is_file():
            try:
                selected=scoped(self.root,pointer.read_text(encoding='utf-8').strip())/'best.pt'
                if selected.is_file():recommended=relative(self.root,selected)
            except (OSError,ValueError):pass
        self.catalog_cache=dict(configs=configs,datasets=datasets,checkpoints=checkpoints,errors=errors,recommended_checkpoint=recommended)
        self.catalog_at=time.monotonic();return self.catalog_cache

    def browse(self,value='data'):
        path=scoped(self.root,value,True)
        if not path.is_dir():raise ValueError('请选择目录')
        entries=[]
        for item in sorted(path.iterdir(),key=lambda p:(not p.is_dir(),p.name.lower())):
            if item.name.startswith('.') or item.name in ('node_modules','downloads','__pycache__'):continue
            try:scoped(self.root,str(item),True)
            except ValueError:continue
            if item.is_dir() or item.suffix.lower() in ('.json','.jsonl','.npz','.pt','.mp4','.avi','.wav'):
                entries.append(dict(name=item.name,path=relative(self.root,item),directory=item.is_dir()))
            if len(entries)>=300:break
        return dict(path=relative(self.root,path),entries=entries)

    def _spec(self,payload,jobdir):
        kind=payload.get('kind')
        if isinstance(kind,str) and kind.startswith('v2_'):
            from .v2_api import job_spec
            return job_spec(self,payload,jobdir)
        if kind not in ('train','evaluate','prepare','download_ami','export_coreml','smoke','inspect','quality','freeze','replay','raw_demo','gate','proxy'):
            raise ValueError('未知任务类型')
        uid=jobdir.name
        result=dict(id=uid,kind=kind,workspace=str(self.root),created_at=stamp(),status='queued',output=None)
        config=None;args=[]
        if kind in ('train','prepare','download_ami','smoke','inspect'):
            config=checked_config(payload.get('config'))
            config_path=jobdir/'config.json';atomic_json(config_path,config.to_dict())
            args=['--config',str(config_path)]
        if kind in ('train','evaluate','inspect','quality','freeze','replay'):
            data=scoped(self.root,payload.get('data'),True)
            if not data.is_dir():raise ValueError('数据路径必须是包含 .npz 的目录')
            from .storage import dataset_paths
            packs=[pack_info(p,self.root) for p in dataset_paths(data,verify=False)]
            if not packs:raise ValueError('该目录没有 .npz 数据包；请先准备素材')
            weak=flag(payload,'allow_weak');synthetic=flag(payload,'allow_synthetic')
            if kind=='train':
                if not any(p['metadata'].get('split')=='train' for p in packs) or not any(p['metadata'].get('split')=='validation' for p in packs):
                    raise ValueError('训练目录必须同时含 train 和 validation，按原始场次隔离')
                for pack in packs:
                    m=pack['metadata']
                    if m.get('preview_only'):raise ValueError('未标注回放包不能训练，请先完成偏好审核')
                    if m.get('source')=='synthetic' and not synthetic:raise ValueError('合成数据仅能用于烟雾测试；请明确勾选合成数据确认')
                    if m.get('source')=='real' and not m.get('human_reviewed') and not weak:raise ValueError('标签未人工审核；只有确认弱监督后才能训练')
                    for key in ('max_cameras','image_size','sample_rate','audio_samples','decision_hz'):
                        if m.get('config',{}).get(key)!=getattr(config,key):raise ValueError(f'数据与配置 {key} 不匹配；请重新准备数据')
        if kind=='train':
            device=payload.get('device','cpu')
            if not any(x['id']==device and x['available'] for x in self.system['devices']):raise ValueError('所选训练设备不可用；可选择 CPU')
            epochs=integer(payload,'epochs',30,1,10000);sequence=integer(payload,'sequence_steps',16,2,512)
            burn=integer(payload,'burn_in',4,0,511);batch=integer(payload,'batch_size',2,1,64)
            seed=integer(payload,'seed',42,0,2147483647)
            if burn>=sequence:raise ValueError('burn-in 必须小于序列长度')
            lr=payload.get('lr',.001)
            if isinstance(lr,bool) or not isinstance(lr,(int,float)) or not math.isfinite(lr) or not 1e-7<=lr<=.1:raise ValueError('学习率范围为 1e-7～0.1')
            out=self.root/'runs'/('web_'+uid)
            args+=['--data',str(data),'--output',str(out),'--epochs',str(epochs),'--sequence-steps',str(sequence),'--burn-in',str(burn),'--batch-size',str(batch),'--device',device,'--lr',str(lr),'--seed',str(seed),'--stop-file',str(jobdir/'stop.request'),'--progress-file',str(jobdir/'progress.json')]
            if weak:args.append('--allow-weak-supervision')
            if synthetic:args.append('--allow-synthetic')
            if flag(payload,'balance_actions'):args.append('--balance-actions')
            args+=['--cache-root',str(self.root/'.cache/sequences')]
            mode=payload.get('mode','new')
            if mode not in ('new','fine_tune','resume'):raise ValueError('Invalid training mode')
            if mode!='new':
                parent=scoped(self.root,payload.get('parent_checkpoint'),True)
                if parent.suffix!='.pt':raise ValueError('请选择 .pt 父检查点')
                args+=['--init-checkpoint' if mode=='fine_tune' else '--resume-checkpoint',str(parent)]
            scheduler=payload.get('scheduler','none');precision=payload.get('precision','fp32')
            if scheduler not in ('none','cosine','plateau') or precision not in ('fp32','amp_fp16'):raise ValueError('Invalid scheduler/precision')
            args+=['--scheduler',scheduler,'--precision',precision,'--patience',str(integer(payload,'patience',0,0,1000)),
                   '--accumulation-steps',str(integer(payload,'accumulation_steps',1,1,64)),
                   '--checkpoint-steps',str(integer(payload,'checkpoint_steps',10,1,10000)),
                   '--num-workers',str(integer(payload,'num_workers',0,0,4))]
            result.update(module='camix_director.train',output=str(out))
        elif kind=='evaluate':
            cp=scoped(self.root,payload.get('checkpoint'),True)
            if cp.suffix!='.pt':raise ValueError('请选择 .pt 检查点')
            split=payload.get('split','test')
            if split not in ('validation','test'):raise ValueError('只能评估 validation 或 test')
            out=self.root/'runs'/'web_reports'/uid
            args=['--checkpoint',str(cp),'--data',str(data),'--output',str(out),'--split',split]
            if weak:args.append('--allow-weak-supervision')
            result.update(module='camix_director.evaluate',output=str(out))
        elif kind=='prepare':
            manifest=scoped(self.root,payload.get('manifest'),True)
            if manifest.suffix!='.json':raise ValueError('素材清单必须是 JSON')
            # The GUI does not let a manifest quietly escape its authorized workspace.
            content=read_json(manifest)
            if not isinstance(content,dict):raise ValueError('素材清单无效')
            for cam in content.get('cameras',[]):
                for name in ('video','audio'):
                    if cam.get(name):scoped(self.root,str(manifest.parent/cam[name]),True)
            for name in ('program_events','annotations'):
                if content.get(name):scoped(self.root,str(manifest.parent/content[name]),True)
            out=scoped(self.root,payload.get('output'))
            if not out.is_relative_to(self.root/'data') or out.suffix!='.npz':raise ValueError('输出必须位于 data 目录且以 .npz 结尾')
            if out.exists():raise ValueError('输出已存在，不会覆盖数据')
            args=[str(manifest)]+args+['--output',str(out)]
            result.update(module='camix_director.prepare',output=str(out))
        elif kind=='download_ami':
            if not flag(payload,'accept_license'):raise ValueError('请先确认 AMI 信号／转写 CC BY 4.0 许可与弱监督限制')
            seconds=integer(payload,'seconds',45,15,120)
            # A separate directory prevents changing the delivered research data.
            out=self.root/'data'/('ami_web_'+uid)
            args+=['--root',str(out),'--seconds',str(seconds),'--accept-ami-cc-by']
            result.update(module='camix_director.ami',output=str(out))
        elif kind in ('quality','freeze'):
            out=self.root/'runs'/'web_reports'/uid
            args=[kind,'--data',str(data),'--output',str(out),'--cache-root',str(self.root/'.cache/sequences'),'--stop-file',str(jobdir/'stop.request')]
            if payload.get('selected_packs') is not None:
                selected=payload['selected_packs']
                if not isinstance(selected,list) or not 1<=len(selected)<=1000:raise ValueError('请选择1～1000个数据包')
                paths=[]
                for value in selected:
                    path=scoped(self.root,value,True)
                    if path.suffix!='.npz' or not any(path.is_relative_to(self.root/name) for name in ('data','datasets/objects')):raise ValueError('Only authorized prepared data packs')
                    paths.append(str(path))
                atomic_json(jobdir/'selection.json',list(dict.fromkeys(paths)));args+=['--selection-file',str(jobdir/'selection.json')]
            if kind=='freeze':
                purpose=payload.get('purpose','train')
                if purpose not in ('train','evaluation'):raise ValueError('Invalid freeze purpose')
                args+=['--workspace',str(self.root),'--name',str(payload.get('name',''))[:100],'--purpose',purpose]
                if weak:args.append('--allow-weak-supervision')
                if synthetic:args.append('--allow-synthetic')
            result.update(module='camix_director.dataset_tools',output=str(out))
        elif kind=='replay':
            checkpoint=scoped(self.root,payload.get('checkpoint'),True)
            if checkpoint.suffix!='.pt':raise ValueError('Choose .pt checkpoint')
            out=self.root/'runs'/'web_reports'/uid
            hold=payload.get('min_hold',2.);margin=payload.get('gain_margin',0.)
            if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in (hold,margin)):raise ValueError('Invalid experimental replay guardrails')
            args=['--checkpoint',str(checkpoint),'--data',str(data),'--output',str(out),'--min-hold',str(hold),'--gain-margin',str(margin),'--stop-file',str(jobdir/'stop.request')]
            if weak:args.append('--allow-weak-supervision')
            result.update(module='camix_director.replay',output=str(out))
        elif kind=='proxy':
            from .annotations import AnnotationStore
            project=AnnotationStore(self.root).get(payload.get('project_id'))
            index=integer(payload,'camera_index',0,0,len(project['session']['cameras'])-1)
            out=self.root/'data/proxies'/uid
            args=['--workspace',str(self.root),'--project',project['id'],'--camera',str(index),'--output',str(out),'--stop-file',str(jobdir/'stop.request')]
            result.update(module='camix_director.media_proxy',output=str(out))
        elif kind=='gate':
            checkpoint=scoped(self.root,payload.get('checkpoint'),True)
            evaluation=scoped(self.root,payload.get('evaluation'),True)
            if checkpoint.suffix!='.pt' or evaluation.suffix!='.json':raise ValueError('请选择检查点和评估JSON')
            out=self.root/'runs'/'web_reports'/uid
            args=['--checkpoint',str(checkpoint),'--evaluation',str(evaluation),'--output',str(out)]
            if payload.get('replay'):
                replay_path=scoped(self.root,payload['replay'],True)
                if replay_path.suffix!='.json':raise ValueError('Invalid replay JSON')
                args+=['--replay',str(replay_path)]
            result.update(module='camix_director.gates',output=str(out))
        elif kind=='export_coreml':
            if not self.system['coreml_export_available']:raise ValueError('Windows 可训练／评估／下载 .pt；Core ML 导出与数值验证需转到 Mac')
            cp=scoped(self.root,payload.get('checkpoint'),True)
            if cp.suffix!='.pt':raise ValueError('请选择 .pt 检查点')
            out=self.root/'runs'/'web_exports'/uid
            args=['--checkpoint',str(cp),'--output',str(out),'--precision','fp32','--parity-steps','12']
            result.update(module='camix_director.export',output=str(out))
        elif kind=='raw_demo':
            if not flag(payload,'ack_synthetic'):raise ValueError('请确认合成原片只用于学习标注流程')
            out=self.root/'data'/('synthetic_media_'+uid)
            args=['--root',str(out),'--seconds','12','--accept-synthetic']
            result.update(module='camix_director.raw_demo',output=str(out))
        elif kind=='smoke':
            if not flag(payload,'ack_synthetic'):raise ValueError('请确认合成数据只用于验证链路，不代表导播效果')
            out=self.root/'data'/('synthetic_web_'+uid)
            args+=['--output',str(out),'--steps','24','--sessions','3','--acknowledge-synthetic']
            result.update(module='camix_director.synthetic',output=str(out))
        elif kind=='inspect':
            out=self.root/'runs'/'web_reports'/uid
            args+=['--data',str(data),'--out',str(out)]
            if weak:args.append('--allow-weak-supervision')
            result.update(module='camix_director.inspect_data',output=str(out))
        result['args']=args
        return result

    def start(self,payload):
        if not isinstance(payload,dict):raise ValueError('请求必须是 JSON 对象')
        with self.lock:
            self.refresh()
            if self.active and self.active[1].poll() is None:
                finishing=read_json(self.jobs_dir/self.active[0]/'job.json',{}) or {}
                if finishing.get('status') in TERMINAL:
                    try:self.active[1].wait(timeout=5)
                    except subprocess.TimeoutExpired:raise ValueError('任务已完成，进程仍在释放资源；请稍后再启动')
                    self.refresh()
                else:raise ValueError('已有任务正在运行，请先停止或等待完成')
            uid=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
            jobdir=self.jobs_dir/uid;jobdir.mkdir()
            try:
                spec=self._spec(payload,jobdir)
            except Exception:
                # Only remove the freshly-created spec/config; never data or results.
                for file in jobdir.iterdir():file.unlink()
                jobdir.rmdir();raise
            atomic_json(jobdir/'spec.json',spec)
            atomic_json(jobdir/'job.json',{k:v for k,v in spec.items() if k not in ('args',)})
            env=dict(os.environ);env.update(PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',PYTHONPATH=str(self.code_root))
            log=(jobdir/'console.log').open('wb')
            kwargs={'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}
            try:
                process=subprocess.Popen([sys.executable,'-u','-m','camix_director.web_worker','--job-dir',str(jobdir)],
                    cwd=self.code_root,env=env,stdout=log,stderr=subprocess.STDOUT,**kwargs)
            except Exception as error:
                atomic_json(jobdir/'job.json',dict(spec,status='failed',error=str(error),finished_at=stamp()));raise
            finally:log.close()
            self.active=(uid,process);self.catalog_at=0
            return self.get(uid)

    def refresh(self):
        if not self.active:return
        uid,process=self.active
        code=process.poll()
        if code is None:return
        jobdir=self.jobs_dir/uid;job=read_json(jobdir/'job.json',{}) or {}
        if job.get('status') not in TERMINAL:
            job.update(status='cancelled' if (jobdir/'stop.request').exists() else 'failed',exit_code=code,finished_at=stamp(),error='任务进程已退出；查看日志。已有完整检查点保留。')
            atomic_json(jobdir/'job.json',job)
        self.active=None;self.catalog_at=0

    def get(self,uid):
        if not isinstance(uid,str) or not uid or any(c not in '0123456789_abcdef' for c in uid):raise ValueError('任务ID无效')
        directory=scoped(self.jobs_dir,uid,True);job=read_json(directory/'job.json',{}) or {}
        job.pop('args',None)
        progress=read_json(directory/'progress.json')
        if progress:job['progress']=progress
        if job.get('output'):
            output=scoped(self.root,job['output'])
            job['output_relative']=relative(self.root,output)
            report=read_json(output/'training-report.json')
            if report:job['training_report']=report
            evaluation=read_json(output/'evaluation.json')
            if evaluation:job['evaluation']=evaluation
            export=read_json(output/'export-report.json')
            if export:job['export_report']=export
            for key,name in [('quality_report','quality-report.json'),('frozen_dataset','frozen-dataset.json'),('replay_report','replay-report.json'),('gate_report','gate-report.json'),('proxy_report','proxy-report.json')]:
                value=read_json(output/name)
                if value:job[key]=value
            if (output/'best.pt').is_file():job['checkpoint']=relative(self.root,output/'best.pt')
        if job.get('status') not in TERMINAL and (not self.active or self.active[0]!=uid):
            heartbeat=read_json(directory/'heartbeat.json',{}) or {}
            if time.time()-heartbeat.get('timestamp',0)<15:
                job['control']='cooperative_after_reconnect'
                job['connection_warning']='已重新连接工作进程；可请求协作停止，强制终止只对当前服务拥有的进程可用。'
            else:
                job['connection_warning']='工作进程心跳中断；不按旧PID杀进程，不自动重跑。可从完整 last.pt 新建恢复实验。'
                job['status']='interrupted'

        return job

    def jobs(self):
        with self.lock:
            self.refresh()
            return [self.get(path.name) for path in sorted(self.jobs_dir.iterdir(),reverse=True)[:40] if path.is_dir()]

    def logs(self,uid):
        self.get(uid)
        path=self.jobs_dir/uid/'console.log'
        if not path.exists():return ''
        with path.open('rb') as source:
            source.seek(0,2);source.seek(max(0,source.tell()-100_000))
            return source.read().decode('utf-8',errors='replace')

    def _terminate(self,process):
        if process.poll() is not None:return
        if os.name=='nt':
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=15)
        else:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:return
            try:process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass

    def stop(self,uid):
        with self.lock:
            self.refresh()
            directory=self.jobs_dir/uid
            if not self.active or self.active[0]!=uid:
                job=self.get(uid)
                if job.get('control')!='cooperative_after_reconnect':raise ValueError('该任务没有可验证的活跃心跳，不按旧PID终止')
                (directory/'stop.request').write_text('Cooperative stop after server reconnect',encoding='utf-8')
                return self.get(uid)
            process=self.active[1]
            (directory/'stop.request').write_text('User requested stop\n',encoding='utf-8')
            job=read_json(directory/'job.json',{}) or {};job['stop_requested']=True
            atomic_json(directory/'job.json',job)
            def escalate():
                try:process.wait(timeout=12)
                except subprocess.TimeoutExpired:self._terminate(process)
            threading.Thread(target=escalate,daemon=True).start()
            return self.get(uid)

    def close(self):
        with self.lock:
            self.refresh()
            if not self.active:return
            uid,process=self.active;self.stop(uid)
        try:process.wait(timeout=14)
        except subprocess.TimeoutExpired:self._terminate(process)
        self.refresh()
