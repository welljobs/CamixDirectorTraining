"""Small, source-attributed AMI pilot. REAL media, WEAK speaker-derived supervision.

This is NOT professional director ground truth and NOT a general-director dataset.
Only signal and transcription data described as CC BY 4.0 by the source are used.
"""
import argparse,hashlib,html,json,re,urllib.request,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
from .config import ModelConfig,SCHEMA_VERSION
from .download import download

BASE="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
HOME="https://groups.inf.ed.ac.uk/ami/corpus/"
SIGNALS=HOME+"signals.shtml"
ANNOTATIONS="https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
ANNOTATION_SHA="b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d"


def source_text(url,path):
    path=Path(path)
    if not path.exists():
        with urllib.request.urlopen(url,timeout=30) as response:path.write_bytes(response.read())
    return path.read_text(errors="replace", encoding="utf-8")


def camera_mapping(page,meeting):
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>",page,re.S):
        cells=[html.unescape(re.sub(r"<[^>]+>"," ",x)).strip() for x in re.findall(r"<td[^>]*>(.*?)</td>",row,re.S)]
        if cells and cells[0]==meeting:
            return {agent:(int(camera),int(channel)) for camera,channel,agent in re.findall(r"Closeup(\d+)\s*,\s*(\d+)\s*,\s*([A-D])",cells[1])}
    raise ValueError(f"No official camera/channel mapping for {meeting}")


def word_intervals(archive,meeting,agent):
    path=f"words/{meeting}.{agent}.words.xml"
    root=ET.fromstring(archive.read(path))
    result=[]
    for item in root.iter():
        if item.tag.split('}')[-1]!='w' or item.get('punc')=='true':continue
        if item.get('starttime') is None or item.get('endtime') is None:continue
        start,end=float(item.get('starttime')),float(item.get('endtime'))
        if end>start:result.append((start,end))
    return result


def active(intervals,times):
    out=np.zeros(len(times),bool)
    for start,end in intervals:out|=(times>=start)&(times<end)
    return out


def transcript_preferences(archive,meeting,start,times,mask,modalities,n):
    """Only a single speaker among ALL four participants can supply a weak label."""
    activity=np.stack([active(word_intervals(archive,meeting,agent),start+times) for agent in "ABCD"],axis=1)
    prefs=np.full((len(times),n+1),np.nan,np.float32)
    for i in range(len(times)):
        speakers=np.flatnonzero(activity[i])
        if len(speakers)!=1 or speakers[0]>=2 or not mask[i,:2].all() or not modalities[i,:2,1].all():continue
        speaker=int(speakers[0])
        prefs[i,0]=1 if speaker==0 else 0
        prefs[i,1]=prefs[i,0];prefs[i,2]=1 if speaker==1 else 0
    return prefs


def choose_start(intervals,duration,hz):
    best=None
    for start in range(30,271,15):
        times=start+np.arange(1,duration-.01,1/hz)
        a,b=(active(x,times) for x in intervals)
        a_only=int((a&~b).sum());b_only=int((b&~a).sum())
        score=min(a_only,b_only)+.1*(a_only+b_only)
        if best is None or score>best[0]:best=(score,start,a_only,b_only)
    if min(best[2:])<3:raise ValueError("No useful two-speaker pilot interval")
    return best[1],{"agent_A_only_ticks":best[2],"agent_B_only_ticks":best[3]}


from .media import video_samples, audio_samples


def build(root,config,duration=45,meetings=None):
    if config.max_cameras<2:raise ValueError("AMI pilot needs at least two slots")
    if not 15<=duration<=120:raise ValueError("pilot duration must be 15..120 seconds")
    root=Path(root);sources=root/'sources';packs=root/'prepared';sources.mkdir(parents=True,exist_ok=True);packs.mkdir(exist_ok=True)
    license_page=source_text(HOME,sources/'ami-license-page.html')
    if 'CC BY 4.0' not in license_page:raise ValueError("Source licence statement changed; manual review required")
    page=source_text(SIGNALS,sources/'ami-signals.html')
    archive_path=sources/'ami_public_manual_1.6.2.zip'
    receipt=download(ANNOTATIONS,archive_path,max_bytes=100*1024*1024,sha256=ANNOTATION_SHA)
    reports=[]
    with zipfile.ZipFile(archive_path) as archive:
        for meeting,split in (meetings or [('ES2002a','train'),('ES2003a','validation'),('ES2004a','test')]):
            output=packs/f'{meeting}.npz'
            if output.exists():
                with np.load(output,allow_pickle=False) as f:existing=json.loads(str(f['metadata'].item()))
                if all(existing.get('config',{}).get(k)==getattr(config,k) for k in ('max_cameras','image_size','sample_rate','audio_samples','decision_hz')) and existing.get('duration_seconds')==duration:
                    with np.load(output,allow_pickle=False) as z:payload={k:z[k] for k in z.files}
                    payload['preferences']=transcript_preferences(archive,meeting,existing['source_start_seconds'],payload['times'],payload['camera_mask'],payload['modality_mask'],config.max_cameras)
                    existing['labelled_ticks']=int(np.isfinite(payload['preferences']).any(axis=1).sum())
                    existing['weak_label_policy']='single_speaker_among_all_four_v2'
                    existing.pop('prepared_sha256',None)
                    payload['metadata']=np.array(json.dumps(existing));np.savez_compressed(output,**payload)
                    existing['prepared_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
                    reports.append(existing);continue
                raise ValueError(f"Existing pack has different config; choose a fresh output directory: {output}")
            mapping=camera_mapping(page,meeting)
            intervals=[word_intervals(archive,meeting,agent) for agent in ['A','B']]
            start,balance=choose_start(intervals,duration,config.decision_hz)
            times=np.arange(1,duration-.01,1/config.decision_hz,dtype=np.float64)
            t,n,h,s=len(times),config.max_cameras,config.image_size,config.audio_samples
            frames=np.zeros((t,n,h,h,3),np.uint8);audio=np.zeros((t,n,s),np.float32)
            modalities=np.zeros((t,n,2),np.float32);age=np.full((t,n,2),10,np.float32)
            media=[]
            for slot,agent in enumerate(['A','B']):
                camera,channel=mapping[agent]
                video_url=f'{BASE}/{meeting}/video/{meeting}.Closeup{camera}.avi'
                audio_url=f'{BASE}/{meeting}/audio/{meeting}.Headset-{channel}.wav'
                print(f'AMI {meeting} {agent}: {start}s..{start+duration}s cam={camera} mic={channel}',flush=True)
                frames[:,slot],modalities[:,slot,0],age[:,slot,0]=video_samples(video_url,start,times,h)
                audio[:,slot],modalities[:,slot,1]=audio_samples(audio_url,start,times,config)
                age[:,slot,1]=np.where(modalities[:,slot,1]>0,0,10)
                media.append(dict(agent=agent,camera=camera,microphone_channel=channel,video_url=video_url,audio_url=audio_url))
                (sources/f'{meeting}.{agent}.words.xml').write_bytes(archive.read(f'words/{meeting}.{agent}.words.xml'))
            mask=modalities[:,:,0].copy()
            current=np.zeros((t,n),np.float32);current[:,0]=1
            prefs=transcript_preferences(archive,meeting,start,times,mask,modalities,n)
            labelled=int(np.isfinite(prefs).any(axis=1).sum())
            metadata=dict(schema_version=SCHEMA_VERSION,session_id=meeting,split=split,source='real',
                supervision='weak_ami_transcript_speaker',human_reviewed=False,alignment_verified=True,
                weak_label_policy='single_speaker_among_all_four_v2',
                program_history_source='simulated_constant_slot_0_NOT_director_ground_truth',
                alignment_method='AMI published synchronized signals; past-frame and timestamp-based PCM selection; not live-network validation',
                config=config.to_dict(),duration_seconds=duration,source_start_seconds=start,camera_ids=['AMI_A','AMI_B'],
                labelled_ticks=labelled,media=media,balance=balance,license='CC BY 4.0 (signals/transcription only)',
                provenance=dict(source=HOME,signals=SIGNALS,annotations_sha256=receipt['sha256'],
                    attribution='AMI Meeting Corpus, AMI Consortium; resized/trimmed audio-video excerpts; weak preferences derived from word timings; no endorsement'),
                warning='Real meeting media, weak speech-turn proxy only. Not professional shot-selection labels, not multi-scene acceptance.')
            if labelled<10:raise ValueError(f'Insufficient labels or failed media alignment: {meeting}')
            np.savez_compressed(output,frames=frames,audio=audio,modality_mask=modalities,camera_mask=mask,media_age=age,
                current_program=current,held_seconds=times.astype(np.float32)[:,None],preferences=prefs,times=times,
                metadata=np.array(json.dumps(metadata)))
            metadata['prepared_sha256']=hashlib.sha256(output.read_bytes()).hexdigest();reports.append(metadata)
    (root/'source-report.json').write_text(json.dumps(reports,indent=2), encoding="utf-8")
    return reports

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Download small AMI excerpts for an explicitly WEAK supervision experiment')
    p.add_argument('--root',required=True);p.add_argument('--config',required=True);p.add_argument('--seconds',type=int,default=45)
    p.add_argument('--holdout-meeting',help='Prepare only this separate AMI meeting as test; never train on it')
    p.add_argument('--accept-ami-cc-by',action='store_true',help='Read source CC BY 4.0 terms; preserve attribution and transformation notices')
    a=p.parse_args()
    if not a.accept_ami_cc_by:p.error('Review source licence and pass --accept-ami-cc-by explicitly')
    if a.holdout_meeting and not re.fullmatch(r'[A-Z]{2}[0-9]{4}[a-d]',a.holdout_meeting):p.error('Invalid AMI meeting ID')
    build(a.root,ModelConfig(**json.loads(Path(a.config).read_text(encoding="utf-8"))),a.seconds,[(a.holdout_meeting,'test')] if a.holdout_meeting else None)
