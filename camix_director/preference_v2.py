"""Reviewed reference-relative trajectory preferences, not online RL or reward hacking.

Each variant has its own observed PGM/held time and causal recurrent state.
Only its past is fed to the policy; full-clip preferences remain targets.
"""
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from .contracts_v2 import program_at
from .features_v2 import iter_features
from .model_v2 import action_log_probs
from .policy_runtime_v2 import CausalSignals, tensors_for_step
from .training_state import TrainingCancelled

MAX_POLICY_CLIP_STEPS = 128


class UnavailableVariant(ValueError):
    pass


def usable_comparisons(event):
    result=[]
    for row in event.manifest['comparisons']:
        if row['label']=='uncertain':continue
        steps=math.ceil((row['end_s']-row['start_s'])*event.index['config']['decision_hz'])
        if steps>MAX_POLICY_CLIP_STEPS:
            raise ValueError('Preference clip exceeds 128 decision steps; obtain a shorter reviewed comparison, do not silently split its label')
        result.append(row)
    return result


def trajectory_log_likelihood(root, model, event, row, variant, stop_file=None):
    device=next(model.parameters()).device
    state=model.initial_state(device=device);signals=CausalSignals()
    ids=[c['id'] for c in event.manifest['cameras']]
    current=event.manifest['initial_program'];last_cut=0.;event_position=0;values=[]
    end_tick=math.ceil(row['end_s']*event.index['config']['decision_hz'])
    for chunk,arrays in iter_features(root,event.cache_id,tick_end=min(end_tick,event.index['ticks'])):
        for local,timestamp in enumerate(arrays['times']):
            t=float(timestamp)
            if t>=row['end_s']:break
            if stop_file and Path(stop_file).exists():raise TrainingCancelled('Preference training stopped at epoch boundary recovery scope')
            before=t<row['start_s']
            cuts=event.manifest['program_events']
            prefix_limit=t if before else row['start_s']
            while event_position<len(cuts) and cuts[event_position]['time_s']<prefix_limit:
                cut=cuts[event_position]
                if current!=cut['camera_id']:current,last_cut=cut['camera_id'],cut['time_s']
                event_position+=1
            pgm=np.zeros((1,len(ids)),np.float32);pgm[0,ids.index(current)]=1
            observed={key:value[local:local+1] for key,value in arrays.items()}
            observed.update(current_program=pgm,held_seconds=np.array([[max(0.,t-last_cut)]],np.float32),
                            style_features=np.array([[event.manifest['style'].get(k,0.) for k in ('pace','reaction_bias','detail_bias')]],np.float32))
            inputs,prepared=tensors_for_step(observed,0,model.config,device,signals,event.manifest['clock'])
            if before:
                with torch.no_grad():outputs=model(*inputs,*state)
                state=tuple(x.detach() for x in outputs[4:])
            else:
                outputs=model(*inputs,*state);state=outputs[4:]
                target=program_at(variant['initial_program'],variant['cuts'],t)
                logp,legal=action_log_probs(outputs[0],outputs[1],prepared['video_mask'],prepared['current_program'])
                action=0 if target==current else ids.index(target)+1
                if not bool(legal[0,action]):raise UnavailableVariant('A/B trajectory selects unavailable video; re-review rather than invent an action')
                values.append(logp[0,action])
                if target!=current:current,last_cut=target,t
    if not values:raise UnavailableVariant('No sampled comparison ticks')
    return torch.stack(values).mean()


def preference_epoch(root, model, reference, events, beta, optimizer=None, stop_file=None, progress=None):
    model.eval();reference.eval()  # Disable dropout noise in trajectory comparisons, gradients remain enabled.
    groups={};correct=0;count=0;ties=0;skipped=0
    for event_no,event in enumerate(events):
        rows=usable_comparisons(event)
        for row_no,row in enumerate(rows):
            try:
                a=trajectory_log_likelihood(root,model,event,row,row['a'],stop_file)
                b=trajectory_log_likelihood(root,model,event,row,row['b'],stop_file)
                with torch.no_grad():
                    ra=trajectory_log_likelihood(root,reference,event,row,row['a'],stop_file)
                    rb=trajectory_log_likelihood(root,reference,event,row,row['b'],stop_file)
            except UnavailableVariant:
                skipped+=1;continue
            relative=beta*((a-b)-(ra-rb))
            if row['label']=='tie':loss=relative.square();ties+=1
            else:
                sign=1. if row['label']=='a' else -1.
                loss=F.softplus(-sign*relative);correct+=float(sign*relative.detach()>0);count+=1
            if not torch.isfinite(loss):raise ValueError('Non-finite trajectory preference objective')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            groups.setdefault(event.manifest['original_event_id'],[]).append(float(loss.detach()))
            if progress:progress(phase='preference',event=event_no+1,events=len(events),batch=row_no+1,batches=len(rows))
    if not groups:raise ValueError('No usable reviewed preference trajectories')
    return dict(macro_event_loss=float(np.mean([np.mean(v) for v in groups.values()])),
                action_accuracy=None,preference_relative_sign_accuracy=correct/count if count else None,
                comparisons=count,ties=ties,unavailable_comparisons=skipped,independent_events=len(groups),
                objective='reference_relative_mean_trajectory_loglikelihood_v1; ties constrain relative changes, not unique camera probabilities')
