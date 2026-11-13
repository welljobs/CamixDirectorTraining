"""Closed-loop OFFLINE simulation. No network, mixer, Camix control or production TAKE."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .config import INPUT_NAMES
from .data import read_pack
from .storage import dataset_paths
from .runtime import atomic_json
from .train import load_checkpoint


def replay(checkpoint,data_root,output,min_hold=2.,gain_margin=0.,allow_weak=False,stop_file=None):
    if min_hold<0 or gain_margin<0:raise ValueError('Nonnegative experimental guardrails required')
    model,meta=load_checkpoint(checkpoint);torch.set_num_threads(2);reports=[];trace=[];provenance=[]
    for path in dataset_paths(data_root):
        data,info=read_pack(path,model.config,allow_weak,mapped=True,inference_only=True)
        if info['split'] not in ('test','validation'):continue
        training_ids={p['session_id'] for p in meta['training']['provenance']}
        seen_training=info['session_id'] in training_ids
        provenance.append(dict(session_id=info['session_id'],pack_sha256=info['pack_sha256'],preview_only=info.get('preview_only',False),seen_training_event=seen_training))
        labels_eligible=not seen_training and (info.get('human_reviewed') or (allow_weak and info.get('supervision','').startswith('weak_')))
        memory,mask=model.initial_memory();current=None;last_cut=None;cuts=[];invalid=0;guarded=0;eligible=0;matched=0
        with torch.no_grad():
            for step,timestamp in enumerate(data['times']):
                if stop_file and Path(stop_file).exists():raise InterruptedError('Offline replay stopped')
                available=np.flatnonzero(data['camera_mask'][step]>.5)
                if current is None and len(available):current=int(available[0]);last_cut=float(timestamp)
                pgm=np.zeros((1,model.config.max_cameras),np.float32)
                if current is not None:pgm[0,current]=1
                held=0. if last_cut is None else float(timestamp)-last_cut
                values=[]
                for name in INPUT_NAMES[:-2]:
                    if name=='current_program':value=pgm
                    elif name=='held_seconds':value=np.array([[held]],np.float32)
                    else:
                        source={'video_frames':'frames','audio_pcm':'audio'}.get(name,name);value=data[source][step:step+1]
                        if name=='video_frames':value=value.transpose(0,1,4,2,3).astype(np.float32)/127.5-1
                    values.append(torch.from_numpy(np.array(value,dtype=np.float32,copy=True)))
                keep,take,gains,memory,mask=model(*values,memory,mask)
                reason='hold';target=current
                if not len(available):invalid+=1;reason='no_valid_video_manual_intervention'
                elif current not in available:target=int(available[torch.argmax(take[0,available]).item()]);reason='signal_fallback'
                else:
                    candidate=int(available[torch.argmax(gains[0,available]).item()])
                    if candidate!=current and float(gains[0,candidate])>gain_margin:
                        if held>=min_hold:target=candidate;reason='experimental_gain'
                        else:guarded+=1;reason='minimum_hold_guard'
                # KEEP is relative to the recorded PGM. Do not score it after contexts diverge.
                reference=np.flatnonzero(data['current_program'][step]>.5)
                if labels_eligible and current is not None and len(reference)==1 and reference[0]==current:
                    prefs=data['preferences'][step];annotated=np.flatnonzero(np.isfinite(prefs))
                    if len(annotated)>1:
                        eligible+=1;action=0 if target==current else int(target)+1
                        if action in annotated and prefs[action]>=max(prefs[annotated]):matched+=1
                if target!=current:
                    cuts.append(dict(time_s=float(timestamp),from_camera=current,to_camera=target,reason=reason,execute_allowed=False));current=target;last_cut=float(timestamp)
                trace.append(dict(session_id=info['session_id'],time_s=float(timestamp),simulated_program=current,switch_gain=gains[0].tolist(),reason=reason,execute_allowed=False))
        duration=float(data['times'][-1]-data['times'][0]) if len(data['times'])>1 else 0
        boundaries=[float(data['times'][0])]+[c['time_s'] for c in cuts]+[float(data['times'][-1])]
        lengths=np.diff(boundaries).tolist()
        reports.append(dict(session_id=info['session_id'],duration_seconds=duration,switches=len(cuts),switches_per_minute=len(cuts)/max(duration/60,.001),shot_lengths_seconds=lengths,unavailable_ticks=invalid,guarded_ticks=guarded,reference_context_eligible_ticks=eligible,reference_context_preference_match=matched/max(1,eligible) if eligible else None,cuts=cuts))
    if not reports:raise ValueError('No validation/test sessions for offline replay')
    from .storage import digest_file
    result=dict(data_provenance=provenance,checkpoint_sha256=digest_file(checkpoint),format='camix_offline_closed_loop_v1',policy='experimental_guarded_greedy_NOT_a_learned_long_horizon_policy',min_hold_seconds=min_hold,gain_margin=gain_margin,production_approved=False,execute_allowed=False,sessions=reports,warning='Offline simulation, not a live RTC test or professional directing acceptance. Numeric gains are not calibrated probabilities. Preference matching is reported only while reference PGM contexts match.')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);atomic_json(output/'replay-report.json',result)
    (output/'replay-trace.jsonl').write_text(''.join(json.dumps(t,allow_nan=False)+'\n' for t in trace),encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--min-hold',type=float,default=2);p.add_argument('--gain-margin',type=float,default=0);p.add_argument('--allow-weak-supervision',action='store_true');p.add_argument('--stop-file');a=p.parse_args();print(json.dumps(replay(a.checkpoint,a.data,a.output,a.min_hold,a.gain_margin,a.allow_weak_supervision,a.stop_file),indent=2))
