"""Studio API extensions: annotation, preflight, versions, lineage and comparison."""
from pathlib import Path
import json
from .annotations import AnnotationStore
from .web_core import scoped,relative,read_json
from .preflight import check_training
from .storage import metadata_only
from .media_studio import probe,waveform


class Studio:
    def __init__(self,manager):self.manager=manager;self.root=manager.root;self.annotations=AnnotationStore(self.root)
    def get(self,route,query):
        first=lambda key,default='':query.get(key,[default])[0]
        if route=='/api/studio/media-list':
            paths=[];manifests=[]
            for p in sorted((self.root/'data').rglob('*')):
                if not p.is_file() or not p.resolve().is_relative_to(self.root/'data'):continue
                if p.suffix.lower() in ('.mp4','.m4v','.mov','.webm','.wav','.mp3','.ogg'):paths.append(relative(self.root,p))
                if p.name=='session.json':manifests.append(relative(self.root,p))
                if len(paths)>=1000:break
            return dict(media=paths,manifests=manifests)
        if route=='/api/studio/annotations':return dict(projects=self.annotations.list())
        if route=='/api/studio/annotation':return self.annotations.get(first('id'))
        if route=='/api/studio/probe':return probe(scoped(self.root,first('path'),True))
        if route=='/api/studio/waveform':return waveform(scoped(self.root,first('path'),True),float(first('start','0')),float(first('duration','60')))
        if route=='/api/studio/versions':
            return dict(versions=[dict(id=v['id'],name=v.get('name'),purpose=v['purpose'],path=relative(self.root,p.parent),sessions=v['quality']['sessions'],splits=v['quality']['splits']) for p in sorted((self.root/'datasets/versions').glob('*/dataset-manifest.json')) if (v:=read_json(p))])
        if route=='/api/studio/checkpoint':
            cp=scoped(self.root,first('path'),True)
            if cp.suffix!='.pt':raise ValueError('请选择 .pt')
            report=read_json(cp.parent/'training-report.json',{}) or {};state=read_json(cp.parent/'checkpoint-info.json',{}) or {}
            return dict(path=relative(self.root,cp),full_resume=cp.name=='last.pt' and state.get('format')=='camix_training_state_v2',config=state.get('config',report.get('config')),settings=state.get('settings',report.get('training',{})),position=state.get('position'),model_card=read_json(cp.parent/'model-card.json'),lineage=read_json(cp.parent/'run-manifest.json'))
        if route=='/api/studio/reports':
            return dict(evaluations=[relative(self.root,p) for p in sorted((self.root/'runs').rglob('evaluation.json'))[:100]],replays=[relative(self.root,p) for p in sorted((self.root/'runs').rglob('replay-report.json'))[:100]])
        if route=='/api/studio/experiments':
            rows=[]
            for p in sorted((self.root/'runs').rglob('training-report.json'),key=lambda x:x.stat().st_mtime,reverse=True)[:100]:
                r=read_json(p,{}) or {};labels=read_json(p.parent/'experiment-labels.json',{}) or {};manifest=read_json(p.parent/'run-manifest.json',{}) or {};h=r.get('history',[])
                rows.append(dict(path=relative(self.root,p.parent),name=labels.get('name',p.parent.name),notes=labels.get('notes',''),status=r.get('status'),qualification=r.get('qualification',r.get('status')),epochs=len(h),last=h[-1] if h else None,best_validation_loss=min((x['validation_loss'] for x in h),default=None),dataset_fingerprint=manifest.get('dataset_fingerprint'),parent=manifest.get('parent'),settings=manifest.get('settings',r.get('training')),has_resume=(p.parent/'last.pt').exists(),model_card=read_json(p.parent/'model-card.json')))
            return dict(experiments=rows)
        if route=='/api/studio/report':
            p=scoped(self.root,first('path'),True)
            if not p.is_relative_to(self.root/'runs') or p.suffix!='.json':raise ValueError('Only run JSON reports are readable')
            return read_json(p)
        raise ValueError('Unknown studio endpoint')
    def post(self,route,payload):
        if not isinstance(payload,dict):raise ValueError('请求必须是JSON对象')
        if route=='/api/studio/preflight':
            result=check_training(self.root,payload)
            device=payload.get('device','cpu')
            if not any(d['id']==device and d['available'] for d in self.manager.system['devices']):result['errors'].append(dict(field='device',message='所选设备不可用；可先选择CPU。'))
            if payload.get('precision','fp32')!='fp32' and device!='cuda':result['errors'].append(dict(field='precision',message='当前AMP FP16只开放给已检测到的CUDA，CPU/MPS请选择FP32。'))
            if payload.get('mode')=='resume':
                try:
                    parent=scoped(self.root,payload.get('parent_checkpoint'),True);info=read_json(parent.parent/'checkpoint-info.json',{}) or {}
                    if parent.name!='last.pt' or info.get('format')!='camix_training_state_v2':raise ValueError('需要完整last.pt，只有best.pt请使用微调。')
                    for key,field in [('epochs','epochs'),('batch_size','batch'),('sequence_steps','sequence'),('burn_in','burn'),('seed','seed'),('lr','lr'),('scheduler','scheduler'),('patience','patience'),('accumulation_steps','accumulation'),('precision','precision'),('device','device'),('balance_actions','balance'),('allow_synthetic','synthetic'),('allow_weak','weak')]:
                        if payload.get(key,info['settings'].get(key))!=info['settings'].get(key):result['errors'].append(dict(field=field,message=f'完整恢复要求 {key} 保持检查点设置。改变设置请微调。'))
                except Exception as error:result['errors'].append(dict(field='train-parent',message=str(error)))
            result['passed']=not result['errors'];return result
        if route=='/api/studio/annotation/create':return self.annotations.create(payload['session'])
        if route=='/api/studio/annotation/save':return self.annotations.save(payload['id'],payload['revision'],payload['session'])
        if route=='/api/studio/annotation/preview':return self.annotations.export_preview(payload['id'],payload['revision'],payload.get('alignment_confirmed'))
        if route=='/api/studio/annotation/review':return self.annotations.review(payload['id'],payload['revision'],payload.get('reviewer'),payload.get('alignment_confirmed'),payload.get('labels_confirmed'),payload.get('notes',''))
        if route=='/api/studio/annotation/apply-proxy':
            from .media_proxy import proxy_signature
            project=self.annotations.get(payload['id']);report=read_json(scoped(self.root,payload['report'],True))
            if report.get('project_id')!=project['id']:raise ValueError('Proxy belongs to another project')
            camera=project['session']['cameras'][report['camera_index']]
            if report['source_signature']!=proxy_signature(project['session'],camera):raise ValueError('Source offsets/media changed since proxy creation')
            camera.update(preview_video=report['preview_video'],preview_audio=report['preview_audio'],proxy_signature=report['source_signature'])
            return self.annotations.save(project['id'],payload['revision'],project['session'])
        if route=='/api/studio/annotation/import':
            manifest=scoped(self.root,payload.get('manifest'),True);m=read_json(manifest)
            if not m:raise ValueError('Invalid manifest')
            session={k:m.get(k) for k in ('session_id','split','duration_seconds','initial_program')};session.update(start_seconds=m.get('start_seconds',0),source=m.get('source','real'),scenario_tags=m.get('scenario_tags',[]),cameras=m['cameras'])
            for camera in session['cameras']:
                for key in ('video','audio'):
                    if camera.get(key):camera[key]=relative(self.root,scoped(self.root,str(manifest.parent/camera[key]),True))
            for field,key in [('labels','annotations'),('program_events','program_events')]:
                p=scoped(self.root,str(manifest.parent/m[key]),True) if m.get(key) else None
                session[field]=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()] if p else []
            return self.annotations.create(session)
        if route=='/api/studio/experiment-note':
            from .runtime import atomic_json
            from datetime import datetime,timezone
            p=scoped(self.root,payload.get('run'),True)
            if not p.is_relative_to(self.root/'runs') or not (p/'training-report.json').is_file():raise ValueError('Invalid experiment')
            value=dict(name=str(payload.get('name','')).strip()[:120] or p.name,notes=str(payload.get('notes',''))[:10000],updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(p/'experiment-labels.json',value)
            return value
        if route=='/api/studio/compare':
            paths=payload.get('runs',[])
            if not 1<=len(paths)<=6:raise ValueError('选择1～6个实验')
            rows=[]
            for value in paths:
                p=scoped(self.root,value,True)
                if not p.is_relative_to(self.root/'runs'):raise ValueError('Only experiments in runs')
                r=read_json(p/'training-report.json');m=read_json(p/'run-manifest.json',{}) or {}
                if not r:raise ValueError('Missing training report')
                rows.append(dict(path=relative(self.root,p),history=r['history'],settings=m.get('settings',r.get('training')),config=r.get('config'),dataset_fingerprint=m.get('dataset_fingerprint'),parent=m.get('parent')))
            identities={r['dataset_fingerprint'] for r in rows}
            return dict(experiments=rows,comparable_dataset=len(identities)==1 and None not in identities,warning='数据/标注版本不同的损失不能直接排名；弱标签指标不是导演质量。')
        raise ValueError('Unknown studio operation')
