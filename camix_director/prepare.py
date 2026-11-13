"""Convert explicit synchronized session manifests into causal sequence packs.
Uses isolated PyAV media libraries, never downloads media implicitly.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
from .config import ModelConfig,SCHEMA_VERSION


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def past_video(path,times,size,fps_delay=0):
    from .media import video_samples
    images,valid,age=video_samples(str(path),0,times-fps_delay,size)
    age=age+fps_delay
    return images,(valid*(age<=1)).astype(np.float32),age


def past_audio(path,times,config,delay=0):
    from .media import audio_samples
    audio,valid=audio_samples(str(path),0,times-delay,config)
    return audio,valid,np.where(valid>0,delay,10).astype(np.float32)


def prepare(manifest_path,output,config):
    manifest_path=Path(manifest_path);m=json.loads(manifest_path.read_text(encoding="utf-8"));base=manifest_path.parent
    if not m.get("session_id") or m.get("split") not in ("train","validation","test"):raise ValueError("session_id/split required")
    if not 0<float(m["duration_seconds"])<=120:raise ValueError("Prepare short episodes, at most 120 seconds each")
    for key in ("alignment_verified","human_reviewed"):
        if key in m and not isinstance(m[key],bool):raise ValueError(key+" must be a JSON boolean")
    if not m.get("alignment_verified"):raise ValueError("Explicitly verified source synchronization is required")
    source_start=float(m.get('start_seconds',0))
    if not math.isfinite(source_start) or source_start<0:raise ValueError('Invalid source start')
    if m.get('source','real') not in ('real','synthetic'):raise ValueError('Invalid provenance')
    cams=m["cameras"];ids=[x["id"] for x in cams]
    if len(set(ids))!=len(ids) or not 1<=len(ids)<=config.max_cameras:raise ValueError("invalid camera IDs/count")
    times=np.arange(1.0,float(m["duration_seconds"])-.01,1/config.decision_hz,dtype=np.float64)
    t,n,h,s=len(times),config.max_cameras,config.image_size,config.audio_samples
    estimated=t*n*(h*h*3+s*4+32)
    if estimated>512*1024*1024:raise ValueError('One prepared episode expands beyond512MiB; reduce duration/resolution/decision rate or shard it. No oversized allocation was attempted.')
    frames=np.zeros((t,n,h,h,3),np.uint8);audio=np.zeros((t,n,s),np.float32)
    modality=np.zeros((t,n,2),np.float32);age=np.full((t,n,2),10,np.float32)
    for slot,camera in enumerate(cams):
        for key in ("video_delay","audio_delay"):
            delay=float(camera.get(key,0))
            if not math.isfinite(delay) or not 0<=delay<=1:
                raise ValueError("Media delays must be finite in [0,1] seconds; align source clocks first")
        from .media import video_samples,audio_samples
        video_offset=float(camera.get('video_offset',0));audio_offset=float(camera.get('audio_offset',0))
        if not math.isfinite(video_offset) or not math.isfinite(audio_offset):raise ValueError('Invalid source clock offset')
        frames[:,slot],modality[:,slot,0],age[:,slot,0]=video_samples(str(base/camera['video']),source_start+video_offset,times-float(camera.get('video_delay',0)),h)
        age[:,slot,0]+=float(camera.get('video_delay',0))
        modality[:,slot,0]*=(age[:,slot,0]<=1)
        if camera.get('audio'):
            audio[:,slot],modality[:,slot,1]=audio_samples(str(base/camera['audio']),source_start+audio_offset,times-float(camera.get('audio_delay',0)),config)
            age[:,slot,1]=np.where(modality[:,slot,1]>0,float(camera.get('audio_delay',0)),10)
    mask=modality[:,:,0].copy()
    current=np.zeros((t,n),np.float32);held=np.zeros((t,1),np.float32)
    events=sorted(rows(base/m["program_events"]),key=lambda x:x["time_s"]) if m.get("program_events") else []
    pgm=m.get("initial_program",ids[0]);last_change=0.;event_index=0
    for i,time in enumerate(times):
        while event_index<len(events) and events[event_index]["time_s"]<=time:
            event=events[event_index];pgm=event["target"];last_change=event["time_s"];event_index+=1
        if pgm in ids:current[i,ids.index(pgm)]=1
        held[i,0]=time-last_change
    preferences=np.full((t,n+1),np.nan,np.float32)
    labels=rows(base/m['annotations'])
    from .storage import digest_file
    for camera in cams:
        for media in ('video','audio'):
            if camera.get(media) and camera.get(media+'_sha256') and digest_file(base/camera[media])!=camera[media+'_sha256']:raise ValueError('Reviewed source media changed')
    if m.get('review'):
        from .annotations import review_digest
        if review_digest(m,labels,events)!=m['review'].get('content_sha256'):raise ValueError('Annotations or timing changed after review; re-review is required')
    def assign(index,action,value):
        if np.isfinite(preferences[index,action]) and preferences[index,action]!=value:raise ValueError('Conflicting preferences map to the same decision tick; resolve annotations before preparing')
        preferences[index,action]=value
    for label in labels:
        index=int(np.searchsorted(times,float(label["time_s"]),side="right")-1)
        if index<0 or index>=t:continue
        def action(key):
            if key=="keep":return 0
            if key not in ids:raise ValueError(f"Unknown annotation camera {key}")
            slot=ids.index(key)
            return 0 if current[index,slot]>.5 else slot+1
        if "scores" in label:
            for key,value in label["scores"].items():assign(index,action(key),float(value))
        if "preferred" in label:
            preferred=action(label["preferred"]);assign(index,preferred,1)
            for rejected in label.get("rejected",[]):
                target=action(rejected)
                if target!=preferred:assign(index,target,0)
    meta={"schema_version":SCHEMA_VERSION,"session_id":m["session_id"],"split":m["split"],"source":m.get("source","real"),
          "human_reviewed":bool(m.get("human_reviewed",False)),"alignment_verified":True,
          "supervision":m.get("supervision","human_director_preferences"),"provenance":m.get("provenance",{}),
          "camera_ids":ids,"config":config.to_dict(),"program_history_source":m.get("program_history_source","actual_recorded"),"annotation_version":m.get("annotation_version"),"review":m.get("review"),"scenario_tags":m.get("scenario_tags",[]),"source_start_seconds":source_start,"duration_seconds":float(m["duration_seconds"]),"preview_only":bool(m.get("preview_only",False)),"source_media_sha256":{c["id"]:{k:c.get(k+"_sha256") for k in ("video","audio")} for c in cams}}
    output=Path(output)
    if output.exists():raise ValueError("Choose a fresh pack path; prepared data is never overwritten")
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,frames=frames,audio=audio,modality_mask=modality,camera_mask=mask,
        media_age=age,current_program=current,held_seconds=held,preferences=preferences,times=times,metadata=np.array(json.dumps(meta)))
    return {"session_id":m["session_id"],"steps":t,"annotated_steps":int(np.isfinite(preferences).sum(-1).astype(bool).sum()),"output":str(output)}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("manifest");p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();print(json.dumps(prepare(a.manifest,a.output,ModelConfig(**json.loads(Path(a.config).read_text(encoding="utf-8")))),indent=2))
