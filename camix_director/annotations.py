"""Local annotation projects: optimistic concurrency, revisions and explicit review."""
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import uuid
from .lineage import canonical_hash
from .runtime import atomic_json,file_lock
from .storage import digest_file


def now():return datetime.now(timezone.utc).isoformat()


def review_digest(manifest,labels,events):
    fields=('session_id','split','source','start_seconds','duration_seconds','initial_program','scenario_tags')
    base={k:manifest.get(k,0 if k=='start_seconds' else [] if k=='scenario_tags' else 'real' if k=='source' else None) for k in fields}
    base['cameras']=[{k:c.get(k,0 if k.endswith(('offset','delay')) else None) for k in ('id','video_offset','audio_offset','video_delay','audio_delay','video_sha256','audio_sha256')} for c in manifest['cameras']]
    base.update(labels=labels,program_events=events)
    return canonical_hash(base)


class AnnotationStore:
    def __init__(self,workspace):self.root=Path(workspace).resolve();self.store=self.root/'.web/annotations';self.store.mkdir(parents=True,exist_ok=True)
    def _path(self,value,exists=True):
        if not isinstance(value,str) or not value:raise ValueError('缺少媒体路径')
        path=(self.root/value).resolve()
        if not path.is_relative_to(self.root/'data'):raise ValueError('媒体与清单必须位于工作区 data 内')
        if exists and not path.is_file():raise ValueError('文件不存在：'+value)
        return path
    def directory(self,identifier):
        if not isinstance(identifier,str) or len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):raise ValueError('Invalid annotation project ID')
        return self.store/identifier
    def _validate(self,value,hash_media=False):
        session=dict(value)
        if not isinstance(session.get('session_id'),str) or not session['session_id'].strip():raise ValueError('填写原始录制事件ID')
        if session.get('split') not in ('train','validation','test'):raise ValueError('Invalid split')
        if session.get('source') not in ('real','synthetic'):raise ValueError('明确选择真实素材或合成测试；不能把合成改成人审真素材')
        duration=float(session.get('duration_seconds',0));start=float(session.get('start_seconds',0))
        if not math.isfinite(duration) or not 2<=duration<=120 or not math.isfinite(start) or start<0:raise ValueError('片段长度2～120秒，开始时间非负')
        session['duration_seconds']=duration;session['start_seconds']=start
        cameras=session.get('cameras',[])
        if not 1<=len(cameras)<=8 or len({c.get('id') for c in cameras})!=len(cameras):raise ValueError('1～8个不重复机位ID')
        for camera in cameras:
            if not isinstance(camera.get('id'),str) or not camera['id']:raise ValueError('机位ID不能为空')
            for media in ('video','audio'):
                if media=='audio' and not camera.get(media):camera['audio']=None;camera['audio_sha256']=None;continue
                path=self._path(camera.get(media));camera[media]=path.relative_to(self.root).as_posix()
                if hash_media:camera[media+'_sha256']=digest_file(path)
            for key in ('video_offset','audio_offset','video_delay','audio_delay'):
                offset=float(camera.get(key,0))
                if not math.isfinite(offset) or (key.endswith('delay') and not 0<=offset<=1) or (key.endswith('offset') and abs(offset)>86400):raise ValueError('无效偏移；offset表示源时间=会话时间+offset，delay为0～1秒可用性延迟')
                camera[key]=offset
        ids={c['id'] for c in cameras}
        if session.get('initial_program') not in ids:raise ValueError('初始节目机位必须存在')
        labels=session.get('labels',[]);events=session.get('program_events',[])
        if len(labels)>10000 or len(events)>10000:raise ValueError('单片段标注数量过多，请拆分')
        label_times=set()
        for label in labels:
            label.setdefault('id',uuid.uuid4().hex)
            t=float(label.get('time_s',-1))
            if not math.isfinite(t) or not 1<=t<duration:raise ValueError('标注时刻在[1,片段长度)内；最初1秒用于输入预热')
            if round(t,3) in label_times:raise ValueError('同一时刻存在重复标注，请编辑原条')
            label_times.add(round(t,3));label['time_s']=t
            if label.get('kind')=='uncertain':continue
            if 'scores' in label:
                if not isinstance(label['scores'],dict):raise ValueError('Invalid scores')
                for action,score in label['scores'].items():
                    if action not in ids|{'keep'} or not isinstance(score,(int,float)) or not math.isfinite(score):raise ValueError('无效候选分值')
            else:
                if label.get('preferred') not in ids|{'keep'}:raise ValueError('请选择偏好候选')
                rejected=label.get('rejected',[])
                if not rejected or any(x not in ids|{'keep'} or x==label['preferred'] for x in rejected):raise ValueError('至少选择一个不同的较差候选')
        for event in events:
            t=float(event.get('time_s',-1))
            if not math.isfinite(t) or not 0<=t<duration or event.get('target') not in ids:raise ValueError('无效实际节目事件')
        for label in labels:
            if label.get('kind')=='uncertain':continue
            pgm=session['initial_program']
            for event in sorted(events,key=lambda e:e['time_s']):
                if event['time_s']<=label['time_s']:pgm=event['target']
            def action(value):return 'keep' if value==pgm else value
            if 'preferred' in label and any(action(x)==action(label['preferred']) for x in label['rejected']):raise ValueError('KEEP与当前节目机位是同一个动作，不能互相标成更好/更差')
            if 'scores' in label:
                values={}
                for key,value in label['scores'].items():
                    key=action(key)
                    if key in values and values[key]!=value:raise ValueError('KEEP与当前机位的分值冲突')
                    values[key]=value
                if len(set(values.values()))<2:raise ValueError('仅相等分值没有偏好比较，请记录为不确定/可接受集合')
        session['labels']=sorted(labels,key=lambda x:(x['time_s'],x.get('id','')));session['program_events']=sorted(events,key=lambda x:x['time_s'])
        for camera in cameras:
            if camera.get('preview_video'):
                from .media_proxy import proxy_signature
                if camera.get('proxy_signature')!=proxy_signature(session,camera):
                    for key in ('preview_video','preview_audio','proxy_signature'):camera.pop(key,None)
                else:
                    self._path(camera['preview_video'])
                    if camera.get('preview_audio'):self._path(camera['preview_audio'])
        tags=session.get('scenario_tags',[])
        if not isinstance(tags,list) or len(tags)>20 or any(not isinstance(t,str) or len(t)>80 for t in tags):raise ValueError('场景切片标签须为最多20个短文本；它们只用于评估，不是模型分类头')
        session['scenario_tags']=tags
        return session
    def create(self,payload):
        session=self._validate(json.loads(json.dumps(payload)),hash_media=True)
        identifier=uuid.uuid4().hex;directory=self.directory(identifier);directory.mkdir()
        project=dict(id=identifier,revision=1,created_at=now(),updated_at=now(),session=session,review=None)
        atomic_json(directory/'draft.json',project);atomic_json(directory/'rev-000001.json',project)
        return project
    def list(self):
        result=[]
        for path in sorted(self.store.glob('*/draft.json'),key=lambda p:p.stat().st_mtime,reverse=True):
            try:
                value=json.loads(path.read_text(encoding='utf-8'));s=value['session'];result.append(dict(id=value['id'],revision=value['revision'],session_id=s['session_id'],split=s['split'],label_count=len(s['labels']),reviewed=bool(value.get('review')),updated_at=value['updated_at']))
            except (OSError,ValueError,KeyError):continue
        return result
    def get(self,identifier):return json.loads((self.directory(identifier)/'draft.json').read_text(encoding='utf-8'))
    def save(self,identifier,revision,session):
        directory=self.directory(identifier)
        with file_lock(directory/'edit.lock'):
            old=self.get(identifier)
            if old['revision']!=revision:raise ValueError('标注已在另一页面修改，请重新载入，避免覆盖')
            session=self._validate(json.loads(json.dumps(session)))
            # Preserve original source identity; editing offsets/labels invalidates review.
            if session['source']!=old['session']['source']:raise ValueError('素材来源类型不能在编辑中改变，请另建项目并保留原来源')
            for camera in session['cameras']:
                for media in ('video','audio'):
                    if not camera.get(media):continue
                    previous=next((c for c in old['session']['cameras'] if c['id']==camera['id']),{})
                    if previous.get(media)==camera[media]:camera[media+'_sha256']=previous.get(media+'_sha256')
                    else:camera[media+'_sha256']=digest_file(self._path(camera[media]))
            updated=dict(old,session=session,revision=revision+1,updated_at=now(),review=None)
            atomic_json(directory/f'rev-{updated["revision"]:06}.json',updated);atomic_json(directory/'draft.json',updated)
            return updated
    def export_preview(self,identifier,revision,alignment_confirmed):
        if alignment_confirmed is not True:raise ValueError('回放仍需先检查时间对齐；不要求偏好标签审核')
        project=self.get(identifier)
        if project['revision']!=revision:raise ValueError('项目版本已变化，请重新载入')
        session=self._validate(project['session'],hash_media=True)
        destination=self.root/'data/preview_manifests'/identifier/(str(revision)+'-'+uuid.uuid4().hex[:8]);destination.mkdir(parents=True)
        manifest={k:v for k,v in json.loads(json.dumps(session)).items() if k not in ('labels','program_events')}
        for camera in manifest['cameras']:
            for media in ('video','audio'):
                if camera.get(media):camera[media]=os.path.relpath(self.root/camera[media],destination)
        manifest.update(split='test',preview_only=True,human_reviewed=False,alignment_verified=True,supervision='unlabelled_preview_only',program_history_source='declared_for_preview',annotations='preferences.jsonl',program_events='program-events.jsonl',provenance=dict(origin='unlabelled_offline_inspection_NOT_training_data'))
        (destination/'preferences.jsonl').write_text('',encoding='utf-8')
        (destination/'program-events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in session['program_events']),encoding='utf-8')
        atomic_json(destination/'session.json',manifest)
        return dict(manifest=(destination/'session.json').relative_to(self.root).as_posix(),preview_only=True)
    def review(self,identifier,revision,reviewer,alignment_confirmed,labels_confirmed,notes=''):
        if not isinstance(reviewer,str) or not reviewer.strip():raise ValueError('填写本地审核人名称（非企业认证身份）')
        if alignment_confirmed is not True or labels_confirmed is not True:raise ValueError('须分别确认已检查音画对齐和全部偏好标注')
        directory=self.directory(identifier)
        with file_lock(directory/'edit.lock'):
            project=self.get(identifier)
            if project['revision']!=revision:raise ValueError('版本已变化，请重新载入后审核')
            session=self._validate(project['session'],hash_media=True)
            if not any(l.get('kind')!='uncertain' for l in session['labels']):raise ValueError('没有可审核的偏好比较')
            digest=review_digest(session,session['labels'],session['program_events'])
            evidence=dict(reviewer=reviewer.strip(),reviewed_at=now(),content_sha256=digest,notes=str(notes)[:4000],identity_assurance='local_self_declared_not_SSO')
            project.update(session=session,review=evidence,revision=revision+1,updated_at=now())
            destination=self.root/'data/annotations'/identifier/f'v{project["revision"]:06}'
            destination.mkdir(parents=True)
            manifest={k:v for k,v in json.loads(json.dumps(session)).items() if k not in ('labels','program_events')}
            for camera in manifest['cameras']:
                for media in ('video','audio'):
                    if camera.get(media):camera[media]=os.path.relpath(self.root/camera[media],destination)
            manifest.update(alignment_verified=True,human_reviewed=True,supervision='human_director_preferences',program_history_source='actual_recorded' if session['program_events'] else 'declared_initial_only',program_events='program-events.jsonl',annotations='preferences.jsonl',annotation_version=identifier+':'+str(project['revision']),review=evidence,provenance=dict(origin='local_visual_annotation',review_content_sha256=digest))
            # Paths are relative to the exported manifest; review digest uses media hashes, not paths.
            (destination/'preferences.jsonl').write_text(''.join(json.dumps(l,ensure_ascii=False,allow_nan=False)+'\n' for l in session['labels']),encoding='utf-8')
            (destination/'program-events.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False,allow_nan=False)+'\n' for e in session['program_events']),encoding='utf-8')
            atomic_json(destination/'session.json',manifest)
            project['export_manifest']=(destination/'session.json').relative_to(self.root).as_posix()
            atomic_json(directory/f'rev-{project["revision"]:06}.json',project);atomic_json(directory/'draft.json',project)
            return project
