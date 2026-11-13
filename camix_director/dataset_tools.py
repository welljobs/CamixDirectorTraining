"""Dataset quality reports and immutable, content-addressed dataset versions."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .config import ModelConfig
from .data import read_pack
from .lineage import canonical_hash
from .runtime import atomic_json,file_lock
from .storage import dataset_paths,metadata_only,digest_file


def quality(root,cache_root=None,stop_file=None,selected_paths=None):
    paths=[Path(p) for p in selected_paths] if selected_paths is not None else dataset_paths(root);reports=[];errors=[];warnings=[];seen_ids={};seen_content={};seen_versions={};seen_sources={}
    if not paths:errors.append('没有数据包')
    for path in paths:
        if stop_file and Path(stop_file).exists():raise InterruptedError('Quality check cancelled')
        try:
            metadata=metadata_only(path);config=ModelConfig(**metadata['config'])
            data,meta=read_pack(path,config,allow_weak=True,cache_root=cache_root,mapped=True)
            sid=meta.get('session_id');split=meta['split']
            if not sid:raise ValueError('Missing original session_id')
            if sid in seen_ids and seen_ids[sid]!=split:errors.append(f'{sid}: 同一原始事件跨越了数据划分')
            seen_ids[sid]=split
            for item in (meta.get('source_media_sha256') or {}).values():
                source_video=item.get('video')
                if source_video:
                    if source_video in seen_sources and seen_sources[source_video]!=split:errors.append(f'{sid}: 跨划分复用了同一源视频，即使改了事件ID也不能当独立测试')
                    seen_sources[source_video]=split
            pref=data['preferences'];mask=data['camera_mask'];current=data['current_program']
            valid=np.concatenate((np.ones_like(mask[:,:1]),mask*(1-current)),1)>.5
            annotated=np.isfinite(pref)&valid;safe=np.nan_to_num(pref)
            pairs=(safe[:,:,None]>safe[:,None,:])&annotated[:,:,None]&annotated[:,None,:]
            counts=pairs.sum((1,2));labelled=int((counts>0).sum());n=len(pref)
            video_coverage=data['modality_mask'][:,:,0].mean(0).tolist();audio_coverage=data['modality_mask'][:,:,1].mean(0).tolist()
            for slot,coverage in enumerate(video_coverage):
                if coverage<.95:warnings.append(f'{sid}: 机位{slot+1}有效视频覆盖{coverage:.1%}，检查片段边界、源偏移或故障样本意图')
            if not labelled:warnings.append(f'{sid}: 无合法偏好比较，不能贡献训练')
            if labelled/max(1,n)<.05:warnings.append(f'{sid}: 有效标签覆盖低于5%')
            if meta['source']=='real' and not meta.get('human_reviewed'):warnings.append(f'{sid}: 弱监督，未人工审核导演偏好')
            if meta.get('human_reviewed') and not meta.get('review'):warnings.append(f'{sid}: 人审标志无版本化审核证据（旧格式／自声明）')
            # The cached RGB/PCM fingerprint detects renamed identical episodes, not just IDs.
            content=canonical_hash([digest_file(data.directory/(key+'.npy')) for key in ('frames','audio')])
            if content in seen_content and seen_content[content]!=split:errors.append(f'{sid}: 与其他划分存在相同音视频内容')
            version=meta.get('annotation_version')
            if content in seen_versions and seen_versions[content]!=version:errors.append(f'{sid}: 同一音视频有不同标注版本，请只选择一个审核版本')
            seen_versions[content]=version
            seen_content[content]=split
            rms=[]
            for camera in range(config.max_cameras):
                total=0.;samples=0
                for start in range(0,n,16):
                    chunk=data['audio'][start:start+16,camera];total+=float(np.square(chunk,dtype=np.float64).sum());samples+=chunk.size
                rms.append(float(np.sqrt(total/max(1,samples))))
            reports.append(dict(pack_file=path.name,sha256=meta['pack_sha256'],session_id=sid,split=split,steps=n,labelled_ticks=labelled,pairs=int(counts.sum()),keep_better_pairs=int(pairs[:,0].sum()),take_better_pairs=int(pairs[:,1:].sum()),label_coverage=labelled/max(1,n),source=meta['source'],human_reviewed=bool(meta.get('human_reviewed')),review_evidence=meta.get('review'),scenario_tags=meta.get('scenario_tags',[]),video_valid_ratio=data['modality_mask'][:,:,0].mean(0).tolist(),audio_valid_ratio=data['modality_mask'][:,:,1].mean(0).tolist(),audio_rms=rms,media_fingerprint=content,raw_bytes=sum(data[key].nbytes for key in data)))
        except Exception as error:errors.append(f'{path.name}: {error}')
    splits={name:sum(r['split']==name for r in reports) for name in ('train','validation','test')}
    return dict(format='camix_dataset_quality_v2',created_at=datetime.now(timezone.utc).isoformat(),passed=not errors,errors=errors,warnings=warnings,splits=splits,sessions=len({r['session_id'] for r in reports}),packs=reports,raw_bytes=sum(r['raw_bytes'] for r in reports),warning='格式检查、标签覆盖和波形统计不等于已经证明音画同步或导演审美；人工审核证据仍需真实。')


def freeze(root,workspace,name='',purpose='train',allow_weak=False,allow_synthetic=False,cache_root=None,stop_file=None,selected_paths=None):
    workspace=Path(workspace).resolve();root=Path(root).resolve()
    if purpose not in ('train','evaluation'):raise ValueError('Invalid version purpose')
    report=quality(root,cache_root,stop_file,selected_paths)
    if report['errors']:raise ValueError('; '.join(report['errors']))
    if purpose=='train' and not(report['splits']['train'] and report['splits']['validation']):raise ValueError('冻结训练版本需要 train 和 validation')
    if purpose=='evaluation' and not(report['splits']['test'] or report['splits']['validation']):raise ValueError('评估版本需要 test 或 validation')
    for pack in report['packs']:
        if pack['source']=='synthetic' and not allow_synthetic:raise ValueError('合成数据需要明确确认')
        if pack['source']=='real' and not pack['human_reviewed'] and not allow_weak:raise ValueError('弱监督需要明确确认，不会自动变成人审标签')
    paths=[Path(p) for p in selected_paths] if selected_paths is not None else dataset_paths(root);store=workspace/'datasets/objects';versions=workspace/'datasets/versions';store.mkdir(parents=True,exist_ok=True);versions.mkdir(parents=True,exist_ok=True)
    identity=canonical_hash(dict(purpose=purpose,files=sorted(r['sha256'] for r in report['packs'])))
    version=versions/('v-'+identity[:20]);manifest=version/'dataset-manifest.json'
    with file_lock(workspace/'datasets/freeze.lock'):
        if manifest.exists():return json.loads(manifest.read_text(encoding='utf-8'))
        missing=sum(path.stat().st_size for path in paths if not (store/(digest_file(path)+'.npz')).exists())
        if missing+256*1024*1024>shutil.disk_usage(store).free:raise ValueError('Insufficient disk for immutable dataset objects')
        entries=[]
        for path in paths:
            if stop_file and Path(stop_file).exists():raise InterruptedError('Dataset freeze cancelled')
            sha=digest_file(path);target=store/(sha+'.npz')
            if not target.exists():
                temporary=target.with_suffix('.copying');shutil.copyfile(path,temporary)
                if digest_file(temporary)!=sha:raise ValueError('Source changed while freezing; retry after editing stops')
                temporary.replace(target);target.chmod(0o444)
            elif digest_file(target)!=sha:raise ValueError('Object store integrity failure')
            m=metadata_only(path);entries.append(dict(object_path='../../objects/'+target.name,sha256=sha,source_name=path.name,session_id=m['session_id'],split=m['split']))
        version.mkdir(exist_ok=True)
        value=dict(format='camix_dataset_version_v1',id=version.name,name=name or version.name,purpose=purpose,created_at=datetime.now(timezone.utc).isoformat(),files=entries,quality=report,production_approved=False)
        atomic_json(manifest,value)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['quality','freeze']);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--workspace');p.add_argument('--name',default='');p.add_argument('--purpose',choices=['train','evaluation'],default='train');p.add_argument('--cache-root');p.add_argument('--selection-file');p.add_argument('--stop-file');p.add_argument('--allow-weak-supervision',action='store_true');p.add_argument('--allow-synthetic',action='store_true');a=p.parse_args()
    selected=json.loads(Path(a.selection_file).read_text(encoding='utf-8')) if a.selection_file else None
    result=quality(a.data,a.cache_root,a.stop_file,selected) if a.action=='quality' else freeze(a.data,a.workspace,a.name,a.purpose,a.allow_weak_supervision,a.allow_synthetic,a.cache_root,a.stop_file,selected)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);atomic_json(out/('quality-report.json' if a.action=='quality' else 'frozen-dataset.json'),result);print(json.dumps(result,ensure_ascii=False,indent=2))
    if a.action=='quality' and not result['passed']:raise SystemExit(2)
