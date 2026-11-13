"""AMI original-resolution corpus acquisition; bounded, resumable and auditable.

Original AMI PAL is 720x576, NOT 720p/1080p. Weak transcript labels are NOT
professional switching ground truth. Existing pilot/models are never replaced.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import time
import urllib.request
import zipfile

import numpy as np
from .ami import HOME, SIGNALS, BASE, ANNOTATIONS, ANNOTATION_SHA, camera_mapping, word_intervals, active
from .config import ModelConfig
from .download import download
from .runtime import atomic_json, file_lock
from .storage import digest_file

MEETINGS=(('ES2006a','train'),('ES2007a','train'),('ES2008a','validation'),('ES2009a','test'))
GIB=1024**3
CONFIG=dict(max_cameras=4,image_size=224,sample_rate=16000,audio_samples=4000,
    decision_hz=2.0,hidden_dim=64,heads=4,layers=2,memory_tokens=16,dropout=0.1,
    visual_encoder='tiny_cnn',pretrained_visual=False,audio_encoder='normalized_cnn')


def now():return datetime.now(timezone.utc).isoformat()


def validate_entry(entry):
    meeting=entry['meeting'];kind=entry['kind'];name=entry['name']
    if meeting not in dict(MEETINGS) or entry['split']!=dict(MEETINGS)[meeting]:raise ValueError('Unexpected meeting/split')
    pattern=re.escape(meeting)+(r'\.Closeup[1-4]_orig\.avi' if kind=='video' else r'\.Headset-[0-3]\.wav')
    if kind not in ('video','audio') or not re.fullmatch(pattern,name):raise ValueError('Unexpected source filename')
    if entry['url']!=f'{BASE}/{meeting}/{kind}/{name}':raise ValueError('Only official AMI source URLs are allowed')
    if not 0<int(entry['bytes'])<GIB:raise ValueError('Invalid source length')
    return Path('raw')/meeting/kind/name


def disk_guard(path,required,reserve):
    if shutil.disk_usage(path).free < required+reserve:
        raise OSError('Insufficient free disk; paused safely. Preserve sources, free space, then rerun to resume.')


def fetch_entry(entry,destination,rate_mib=6,reserve_bytes=5*GIB,stop_file=None,opener=None):
    """One sequential stream; .part resume uses validated identity and byte ranges."""
    opener=opener or urllib.request.urlopen
    validate_entry(entry)
    if rate_mib<=0:raise ValueError('A positive bandwidth limit is required')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    receipt=destination.with_suffix(destination.suffix+'.receipt.json')
    part=destination.with_suffix(destination.suffix+'.part')
    identity=destination.with_suffix(destination.suffix+'.part.json')
    expected=int(entry['bytes']);expected_identity={k:entry.get(k) for k in ('url','bytes','etag')}
    with file_lock(destination.with_suffix(destination.suffix+'.lock')):
        if destination.exists():
            if not receipt.exists():raise ValueError('Existing source without receipt; never silently overwrite')
            saved=json.loads(receipt.read_text())
            if saved.get('url')!=entry['url'] or destination.stat().st_size!=expected or digest_file(destination)!=saved.get('sha256'):
                raise ValueError('Existing source identity/checksum changed')
            return saved
        if part.exists():
            if not identity.exists() or json.loads(identity.read_text())!=expected_identity:
                raise ValueError('Partial source identity changed; preserve and inspect before retry')
        else:atomic_json(identity,expected_identity)
        if part.exists() and part.stat().st_size>expected:raise ValueError('Partial file exceeds declared size')
        for attempt in range(12):
            offset=part.stat().st_size if part.exists() else 0
            if offset==expected:break
            disk_guard(destination.parent,0,reserve_bytes)
            headers={'User-Agent':'CamixDirectorTraining/0.3 (rate-limited AMI research)'}
            if offset:
                headers['Range']=f'bytes={offset}-'
                if entry.get('etag'):headers['If-Range']=entry['etag']
            started=time.monotonic();received=0;last_print=started
            try:
                with opener(urllib.request.Request(entry['url'],headers=headers),timeout=120) as response:
                    etag=response.headers.get('ETag')
                    if entry.get('etag') and etag and etag!=entry['etag']:raise ValueError('Remote source ETag changed')
                    if offset:
                        if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {offset}-{expected-1}/{expected}':
                            raise ValueError('Server did not honor exact resume range; no bytes appended')
                    elif response.status!=200:raise ValueError('Unexpected HTTP status for original source')
                    if int(response.headers.get('Content-Length','-1'))!=expected-offset:raise ValueError('Remote source length changed')
                    with part.open('ab') as target:
                        while True:
                            if stop_file and Path(stop_file).exists():raise InterruptedError('Stopped; partial download retained')
                            disk_guard(destination.parent,0,reserve_bytes)
                            block=response.read(min(64*1024,expected-offset-received+1))
                            if not block:break
                            if offset+received+len(block)>expected:raise ValueError('Overlong source refused')
                            target.write(block);received+=len(block)
                            delay=received/(rate_mib*1024**2)-(time.monotonic()-started)
                            if delay>0:time.sleep(delay)
                            if time.monotonic()-last_print>=15:
                                print(f"{entry['name']}: {(offset+received)/expected:.1%}",flush=True);last_print=time.monotonic()
                        target.flush();os.fsync(target.fileno())
                if offset+received!=expected:raise IOError('Truncated source; will resume')
                break
            except (ValueError,InterruptedError):raise
            except Exception as error:
                saved=part.stat().st_size if part.exists() else 0
                print(f"RETRY {entry['name']} {attempt+1}/12 at {saved}/{expected} bytes: {type(error).__name__}: {error}",flush=True)
                if attempt==11:raise
                time.sleep(min(2**attempt,30))
        sha=digest_file(part)
        # The official server does not publish per-video SHA256: this is a local
        # reproducibility hash, not a fabricated publisher signature.
        result=dict(entry,sha256=sha,downloaded_at=now(),publisher_checksum_available=False)
        atomic_json(receipt,result);part.replace(destination);identity.unlink(missing_ok=True)
        return result


def acquire(root,plan,rate_mib=6,reserve_gib=5,stop_file=None,workers=4):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if workers not in (1,2,4):raise ValueError('Use at most4 bounded connections')
    entries=plan['entries'];paths=[validate_entry(e) for e in entries]
    if len(paths)!=32 or len(set(paths))!=32:raise ValueError('Expected exactly four meetings, four videos and four audio tracks each')
    if not 6e9<=sum(e['bytes'] for e in entries)<=7e9:raise ValueError('Download plan is outside the requested 6..7 GB budget')
    remaining=sum(max(0,e['bytes']-(root/p).stat().st_size) if (root/p).exists() else max(0,e['bytes']-(root/p.with_suffix(p.suffix+'.part')).stat().st_size) if (root/p.with_suffix(p.suffix+'.part')).exists() else e['bytes'] for e,p in zip(entries,paths))
    disk_guard(root,remaining+2*GIB,reserve_gib*GIB)
    atomic_json(root/'download-plan.json',plan)
    docs=root/'sources';docs.mkdir(exist_ok=True)
    for url,name in [(HOME,'ami-license.html'),(SIGNALS,'ami-signals.html'),('https://groups.inf.ed.ac.uk/ami/download/','ami-download.html')]:
        path=docs/name
        if not path.exists():
            with urllib.request.urlopen(url,timeout=45) as response:path.write_bytes(response.read(2*1024**2))
    if 'CC BY 4.0' not in (docs/'ami-license.html').read_text(errors='replace'):raise ValueError('Official license statement changed; review first')
    rows=[]
    def fetch(item):
        entry,path=item
        print(f"SOURCE {entry['name']} ({entry['bytes']/1024**2:.1f} MiB)",flush=True)
        return fetch_entry(entry,root/path,rate_mib/workers,reserve_gib*GIB,stop_file)
    with file_lock(root/'download-originals.lock'),ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fetch,item) for item in zip(entries,paths)]
        for future in as_completed(futures):
            rows.append(future.result())
            atomic_json(root/'download-progress.json',dict(updated_at=now(),completed_files=len(rows),total_files=len(entries),verified_bytes=sum(r['bytes'] for r in rows),total_bytes=sum(e['bytes'] for e in entries)))
    rows.sort(key=lambda e:(e['meeting'],e['agent'],e['kind']))
    result=dict(format='camix_ami_original_sources_v1',completed_at=now(),entries=rows,total_bytes=sum(e['bytes'] for e in rows),license='CC BY 4.0 for signals and word transcriptions',attribution='AMI Meeting Corpus, AMI Consortium; no endorsement',original_resolution=[720,576],not_720p_or_1080p=True,production_approved=False)
    atomic_json(root/'source-manifest.json',result)
    return result


def weak_preferences(intervals,times,program,n=4):
    activity=np.stack([active(x,times) for x in intervals],axis=1)
    preferences=np.full((len(times),n+1),np.nan,np.float32)
    single=activity.sum(axis=1)==1
    for tick in np.flatnonzero(single):
        speaker=int(activity[tick].argmax());preferences[tick,0]=float(speaker==program)
        preferences[tick,1:]=0;preferences[tick,speaker+1]=1
    return preferences


def prepare_originals(root,episodes=4,seconds=60,reserve_gib=5):
    from .media import video_samples,audio_samples
    from .media_studio import probe
    from .config import SCHEMA_VERSION
    root=Path(root).resolve();source=json.loads((root/'source-manifest.json').read_text())
    if not 1<=episodes<=16 or not 15<=seconds<=120:raise ValueError('episodes=1..16; seconds=15..120')
    config=ModelConfig(**CONFIG);out=root/f'prepared_224_{episodes}x{seconds}';out.mkdir(exist_ok=True)
    archive=root/'sources/ami_public_manual_1.6.2.zip'
    if not archive.exists():download(ANNOTATIONS,archive,max_bytes=100*1024**2,sha256=ANNOTATION_SHA,workers=1)
    elif digest_file(archive)!=ANNOTATION_SHA:raise ValueError('Word annotation archive SHA256 mismatch')
    reports=[]
    with file_lock(root/'prepare-originals.lock'),zipfile.ZipFile(archive) as words:
        for meeting,split in MEETINGS:
            entries=[e for e in source['entries'] if e['meeting']==meeting]
            by={(e['agent'],e['kind']):e for e in entries}
            for e in entries:
                path=root/validate_entry(e)
                if digest_file(path)!=e['sha256']:raise ValueError('Original source checksum changed')
            probes=[probe(root/validate_entry(by[(a,'video')])) for a in 'ABCD']
            duration=min(p['duration_seconds'] for p in probes)
            for p in probes:
                s=next(s for s in p['streams'] if s['type']=='video')
                if [s['width'],s['height']]!=[720,576]:raise ValueError('Unexpected original video resolution')
            intervals=[word_intervals(words,meeting,a) for a in 'ABCD']
            starts=np.linspace(30,max(30,duration-seconds-30),episodes)
            if len(starts)>1 and np.min(np.diff(starts))<seconds:raise ValueError('Requested episodes would overlap; lower count/duration')
            for index,start in enumerate(starts):
                path=out/f'{meeting}_{index:02}.npz'
                if path.exists():
                    from .storage import metadata_only
                    meta=metadata_only(path)
                    if meta.get('config')!=config.to_dict() or meta.get('source_start_seconds')!=float(start):raise ValueError('Prepared settings changed; choose another episode count/duration')
                    reports.append(dict(file=path.name,steps=meta['steps'],labelled_ticks=meta['labelled_ticks'],split=split));continue
                times=np.arange(1,seconds-.01,1/config.decision_hz,dtype=np.float64);t=len(times);n=4
                estimate=t*n*(config.image_size**2*3+config.audio_samples*4+32)
                if estimate>512*1024**2:raise ValueError('Episode exceeds512MiB; reduce duration')
                disk_guard(root,int(estimate*2.2),reserve_gib*GIB)
                frames=np.zeros((t,n,224,224,3),np.uint8);audio=np.zeros((t,n,config.audio_samples),np.float32)
                modality=np.zeros((t,n,2),np.float32);ages=np.full((t,n,2),10,np.float32)
                print(f'PREPARE {meeting} {index+1}/{episodes} source {start:.2f}s..{start+seconds:.2f}s -> 224px',flush=True)
                for slot,agent in enumerate('ABCD'):
                    frames[:,slot],modality[:,slot,0],ages[:,slot,0]=video_samples(root/validate_entry(by[(agent,'video')]),start,times,224)
                    audio[:,slot],modality[:,slot,1]=audio_samples(root/validate_entry(by[(agent,'audio')]),start,times,config)
                    ages[:,slot,1]=np.where(modality[:,slot,1]>0,0,10)
                if (modality.mean(axis=0)<.98).any():raise ValueError('Insufficient aligned AV coverage; source must be inspected')
                program=index%4;current=np.zeros((t,n),np.float32);current[:,program]=1
                prefs=weak_preferences(intervals,start+times,program)
                labelled=int(np.isfinite(prefs).any(axis=1).sum())
                if not labelled:raise ValueError('No single-speaker labels in this source interval')
                meta=dict(schema_version=SCHEMA_VERSION,session_id=meeting,split=split,source='real',human_reviewed=False,
                    alignment_verified=True,alignment_evidence='AMI source-declared synchronized clocks; decoded timestamp/coverage checks, not an independent human review',
                    supervision='weak_single_speaker_transcript_all_four',weak_label_policy='only exactly one speaker among A/B/C/D; silence/overlap unlabelled',
                    program_history_source='simulated_constant_per_episode_rotating_slot_NOT_recorded_director',
                    config=config.to_dict(),camera_ids=list('ABCD'),steps=t,labelled_ticks=labelled,source_start_seconds=float(start),duration_seconds=seconds,
                    scenario_tags=['meeting','AMI_original_576'],annotation_version='AMI_words_1.6.2_weak_four_agent_v1',
                    source_media_sha256={a:{k:by[(a,k)]['sha256'] for k in ('video','audio')} for a in 'ABCD'},
                    provenance=dict(dataset='AMI Meeting Corpus',license='CC BY 4.0',url=HOME,attribution=source['attribution'],modifications='letterboxed224px RGB, causal16kHz PCM windows, transcript-derived weak preferences; original source retained'),production_approved=False)
                partial=path.with_suffix('.npz.partial')
                with partial.open('wb') as f:np.savez_compressed(f,frames=frames,audio=audio,modality_mask=modality,camera_mask=modality[:,:,0].copy(),media_age=ages,current_program=current,held_seconds=times[:,None].astype(np.float32),preferences=prefs,times=times,metadata=np.array(json.dumps(meta)))
                partial.replace(path);reports.append(dict(file=path.name,steps=t,labelled_ticks=labelled,split=split))
                del frames,audio,modality,ages,prefs
    report=dict(format='camix_ami_original_preparation_v1',updated_at=now(),original_sources_bytes=source['total_bytes'],config=config.to_dict(),episodes=reports,prepared_directory=str(out),prepared_bytes=sum(p.stat().st_size for p in out.glob('*.npz')),weak_supervision=True,human_reviewed=False,production_approved=False)
    atomic_json(root/'preparation-report.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['download','prepare'])
    p.add_argument('--root',default='data/ami_originals_576');p.add_argument('--plan');p.add_argument('--rate-mib',type=float,default=6)
    p.add_argument('--reserve-gib',type=float,default=5);p.add_argument('--accept-ami-cc-by',action='store_true');p.add_argument('--stop-file')
    p.add_argument('--workers',type=int,choices=[1,2,4],default=4);p.add_argument('--prepare-after-download',action='store_true');p.add_argument('--episodes',type=int,default=4);p.add_argument('--seconds',type=int,default=60);a=p.parse_args()
    if a.reserve_gib<5:p.error('Keep at least5GiB free; this command will not fill the system disk')
    if not a.accept_ami_cc_by:p.error('Read source CC BY 4.0 attribution terms and pass --accept-ami-cc-by')
    status_path=Path(a.root)/'pipeline-status.json'
    phase='downloading' if a.action=='download' else 'preparing'
    def status(value,**extra):
        atomic_json(status_path,dict(updated_at=now(),status=value,phase=phase,training_started=False,**extra))
    try:
        status('running')
        if a.action=='download':
            plan=Path(a.plan or Path(a.root)/'download-plan.json')
            if not a.plan and not plan.exists():
                plan=Path(__file__).resolve().parent.parent/'examples/ami-originals-download-plan.json'
            result=acquire(a.root,json.loads(plan.read_text()),a.rate_mib,a.reserve_gib,a.stop_file,a.workers)
            if a.prepare_after_download:
                phase='preparing';status('running')
                result=prepare_originals(a.root,a.episodes,a.seconds,a.reserve_gib)
        else:result=prepare_originals(a.root,a.episodes,a.seconds,a.reserve_gib)
        status('completed',prepared_directory=result.get('prepared_directory'))
    except KeyboardInterrupt:
        status('interrupted',error='User interrupted; completed files and partial downloads retained')
        raise
    except Exception as error:
        status('failed',error=f'{type(error).__name__}: {error}')
        raise
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
