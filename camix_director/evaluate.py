"""Held-out preference evaluation + memory ablation. Not director-aesthetic acceptance."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .data import read_pack
from .config import INPUT_NAMES
from .train import load_checkpoint,preference_loss


def pair_diagnostics(values,targets,camera_mask,current_program):
    valid=torch.cat((torch.ones_like(camera_mask[:,:1]),camera_mask*(1-current_program)),1)>.5
    annotated=torch.isfinite(targets)&valid;safe=torch.nan_to_num(targets)
    pairs=(safe[:,:,None]>safe[:,None,:])&annotated[:,:,None]&annotated[:,None,:]
    ok=(values[:,:,None]>values[:,None,:])&pairs
    return np.array([int(pairs[:,0,:].sum()),int(ok[:,0,:].sum()),int(pairs[:,1:,:].sum()),int(ok[:,1:,:].sum())])


def evaluate(checkpoint_path,data_root,split='test',allow_weak=False):
    model,checkpoint=load_checkpoint(checkpoint_path);torch.set_num_threads(2)
    train_ids={x['session_id'] for x in checkpoint['training']['provenance'] if x['split']=='train' or (split=='test' and x['split']=='validation')}
    results=[];trace=[];data_provenance=[]
    from .storage import dataset_paths,metadata_only
    paths=dataset_paths(data_root)
    with torch.no_grad():
        for path in paths:
            if metadata_only(path).get('split')!=split:continue
            data,meta=read_pack(path,model.config,allow_weak,mapped=True)
            data_provenance.append({k:meta.get(k) for k in ('session_id','source','human_reviewed','review','pack_sha256','scenario_tags')})
            if meta['session_id'] in train_ids:raise ValueError('Training/validation-session leakage into held-out evaluation')
            for ablation,use_memory in [("full",True),("memory_reset_each_step",False),("audio_muted",True),("video_content_muted",True),("one_camera_disconnected",True),("all_cameras_disconnected",True)]:
                memory,mask=model.initial_memory();pairs=0;correct=0;losses=[];latencies=[];groups=np.zeros(4,np.int64);finite_ticks=0;mask_violations=0
                for step in range(len(data['times'])):
                    if not use_memory:memory,mask=model.initial_memory()
                    inputs=[]
                    for name in INPUT_NAMES[:-2]:
                        source={'video_frames':'frames','audio_pcm':'audio'}.get(name,name)
                        value=data[source][step:step+1]
                        if name=='video_frames':value=value.transpose(0,1,4,2,3).astype(np.float32)/127.5-1
                        inputs.append(torch.from_numpy(np.array(value,dtype=np.float32,copy=True)))
                    inputs += [memory,mask]
                    if ablation=="audio_muted":
                        inputs[1]=torch.zeros_like(inputs[1]);inputs[2]=inputs[2].clone();inputs[2][:,:,1]=0
                    if ablation=="video_content_muted":
                        # Preserve selectability; replace visual content, not the live camera mask.
                        inputs[0]=torch.zeros_like(inputs[0])
                    if ablation=='one_camera_disconnected':inputs[2][:,-1,:]=0;inputs[3][:,-1]=0
                    if ablation=='all_cameras_disconnected':inputs[2][:]=0;inputs[3][:]=0
                    started=time.perf_counter();keep,take,gain,memory,mask=model(*inputs)
                    finite_ticks+=int(all(torch.isfinite(x).all() for x in (keep,take,gain,memory,mask)))
                    mask_violations+=int(((inputs[3]<.5)&(take>-9990)).any())
                    latencies.append((time.perf_counter()-started)*1000)
                    loss,count,ok=preference_loss(torch.cat((keep,take),1),torch.from_numpy(data['preferences'][step:step+1].copy()),
                                                inputs[3],inputs[5])
                    groups+=pair_diagnostics(torch.cat((keep,take),1),torch.from_numpy(data["preferences"][step:step+1].copy()),inputs[3],inputs[5])
                    if int(count):losses.append(float(loss));pairs+=int(count);correct+=int(ok)
                    if ablation=="full":
                        trace.append(dict(session_id=meta['session_id'],pack_file=meta['pack_file'],time_s=float(data['times'][step]),
                                          camera_ids=meta.get('camera_ids',[]),keep_value=float(keep[0,0]),
                                          take_value=take[0].tolist(),switch_gain=gain[0].tolist(),execute_allowed=False))
                results.append(dict(session_id=meta['session_id'],pack_file=meta['pack_file'],finite_output_ratio=finite_ticks/len(data['times']),unavailable_selection_mask_violations=mask_violations,split=split,ablation=ablation,memory_enabled=use_memory,
                    pair_accuracy=correct/pairs if pairs else None,pairs=pairs,
                    keep_better_pair_accuracy=float(groups[1]/groups[0]) if groups[0] else None,
                    take_better_pair_accuracy=float(groups[3]/groups[2]) if groups[2] else None,
                    balanced_keep_take_accuracy=float((groups[1]/groups[0]+groups[3]/groups[2])/2) if groups[0] and groups[2] else None,
                    keep_better_pairs=int(groups[0]),take_better_pairs=int(groups[2]),ranking_loss=float(np.mean(losses)) if losses else None,
                    mac_cpu_step_ms_p50=float(np.percentile(latencies,50)),mac_cpu_step_ms_p95=float(np.percentile(latencies,95)),
                    state_mode='sequential with recorded/simulated PGM inputs; not closed-loop aesthetic validation'))
    if not results:raise ValueError('No held-out sessions')
    baseline=[]
    for path in paths:
        if metadata_only(path).get('split')!=split or model.config.max_cameras!=2:continue
        data,meta=read_pack(path,model.config,allow_weak,mapped=True)
        rms_values=np.zeros(data['audio'].shape[:2],np.float32)
        for begin in range(0,len(rms_values),16):rms_values[begin:begin+16]=np.sqrt(np.mean(data['audio'][begin:begin+16]**2,axis=-1))
        rms=torch.from_numpy(rms_values)
        current=torch.from_numpy(data['current_program'].copy());mask=torch.from_numpy(data['camera_mask'].copy());targets=torch.from_numpy(data['preferences'].copy())
        values=torch.cat(((rms*current).sum(1,keepdim=True),rms),1)
        _,count,correct=preference_loss(values,targets,mask,current)
        groups=pair_diagnostics(values,targets,mask,current)
        baseline.append(dict(session_id=meta['session_id'],method='two_camera_raw_pcm_rms_NOT_a_director',pair_accuracy=int(correct)/max(int(count),1),pairs=int(count),
            balanced_keep_take_accuracy=float((groups[1]/groups[0]+groups[3]/groups[2])/2) if groups[0] and groups[2] else None,
            always_keep_pair_accuracy=int(groups[0])/max(int(count),1),always_other_camera_pair_accuracy=int(groups[2])/max(int(count),1)))
    by_session={}
    for row in results:
        if row['ablation']=='full' and row['pairs']:
            entry=by_session.setdefault(row['session_id'],dict(session_id=row['session_id'],pairs=0,correct=0.))
            entry['pairs']+=row['pairs'];entry['correct']+=row['pair_accuracy']*row['pairs']
    full=[dict(session_id=s,pairs=r['pairs'],pair_accuracy=r['correct']/r['pairs']) for s,r in by_session.items()]
    if not full:raise ValueError('No eligible labelled comparisons; use replay for unlabelled inspection or add reviewed preferences')
    accuracies=np.array([r['pair_accuracy'] for r in full]);interval=None
    if len(accuracies)>=2:
        rng=np.random.default_rng(42);boot=np.array([rng.choice(accuracies,len(accuracies),replace=True).mean() for _ in range(1000)])
        interval=np.percentile(boot,[2.5,97.5]).tolist()
    slices={}
    for row in full:
        tags=set(tag for p in data_provenance if p['session_id']==row['session_id'] for tag in (p.get('scenario_tags') or ['untagged']))
        for tag in tags:slices.setdefault(tag,[]).append(row['pair_accuracy'])
    summary=dict(independent_sessions=len(full),episodes=sum(r['ablation']=='full' for r in results),per_session=full,macro_session_pair_accuracy=float(accuracies.mean()),micro_pair_accuracy=sum(r['pair_accuracy']*r['pairs'] for r in full)/sum(r['pairs'] for r in full),session_bootstrap_95_interval=interval,scenario_slices={tag:dict(sessions=len(values),macro_pair_accuracy=float(np.mean(values))) for tag,values in slices.items()},confidence_warning='Single original event: no defensible across-session confidence interval' if interval is None else 'Bootstrap across original event IDs, not clips/ticks; few sessions still limit confidence')
    from .storage import digest_file
    return dict(checkpoint_sha256=digest_file(checkpoint_path),data_provenance=data_provenance,qualification=checkpoint['qualification'],production_approved=False,summary=summary,baselines=baseline,
                warning='Weak/synthetic preference metrics are NOT professional directing quality. Mac CPU timing is NOT iPad latency.',
                evaluations=results),trace

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True)
    p.add_argument('--output',required=True);p.add_argument('--split',default='test');p.add_argument('--allow-weak-supervision',action='store_true')
    a=p.parse_args();report,trace=evaluate(a.checkpoint,a.data,a.split,a.allow_weak_supervision)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'evaluation.json').write_text(json.dumps(report,indent=2), encoding="utf-8");(out/'value-trace.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in trace), encoding="utf-8")
    print(json.dumps(report,indent=2))
