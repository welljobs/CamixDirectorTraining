"""Offline clip-quality learning from reviewed A/B. Never a causal policy input."""
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contracts_v2 import fingerprint, integer, number, program_at
from .corpus_v2 import feature_space, feature_space_id, load_corpus, provenance
from .features_v2 import iter_features
from .runtime import atomic_json
from .storage import digest_file
from .training_state import save_torch

QUALITY_FORMAT='camix_clip_quality_checkpoint_v2'


class ClipQuality(nn.Module):
    def __init__(self,visual_dim,hidden_dim=32):
        super().__init__();self.visual_dim=visual_dim;self.hidden_dim=hidden_dim
        self.input=nn.Sequential(nn.Linear(visual_dim*2+9,hidden_dim),nn.GELU(),nn.LayerNorm(hidden_dim))
        self.temporal=nn.GRU(hidden_dim,hidden_dim,batch_first=True)
        self.output=nn.Linear(hidden_dim*2,1)
    def forward(self,sequence):
        tokens,_=self.temporal(self.input(sequence))
        return self.output(torch.cat((tokens[:,-1],tokens.mean(1)),-1)).squeeze(-1)


def clip_descriptor(root,event,row,variant,max_steps=512):
    ids=[c['id'] for c in event.manifest['cameras']];hz=event.index['config']['decision_hz']
    if math.ceil((row['end_s']-row['start_s'])*hz)>max_steps:raise ValueError('Quality clip exceeds bounded 512 steps; obtain shorter reviewed comparisons')
    values=[];current=variant['initial_program'];last_cut=row['start_s'];position=0
    for chunk,arrays in iter_features(root,event.cache_id,tick_start=max(0,math.floor(row['start_s']*hz)),tick_end=min(event.index['ticks'],math.ceil(row['end_s']*hz))):
        for local,timestamp in enumerate(arrays['times']):
            t=float(timestamp)
            if not row['start_s']<=t<row['end_s']:continue
            selected=program_at(variant['initial_program'],variant['cuts'],t);cut=selected!=current
            if cut:current,last_cut=selected,t
            slot=ids.index(selected);valid=arrays['video_mask'][local];visual=arrays['visual'][local]
            pooled=(visual*valid[:,None]).sum(0)/max(float(valid.sum()),1.)
            audio=arrays['audio_rms'][local];amask=arrays['audio_mask'][local]
            extra=np.array([np.log1p(float(audio[slot])*100),np.log1p(float((audio*amask).sum()/max(float(amask.sum()),1))*100),
                            valid[slot],valid.mean(),float(cut),np.log1p(t-last_cut)/10],np.float32)
            style=np.array([event.manifest['style'].get(k,0.) for k in ('pace','reaction_bias','detail_bias')],np.float32)
            values.append(np.concatenate((visual[slot]*valid[slot],pooled,extra,style)))
    if not values:raise ValueError('Empty quality comparison')
    return torch.from_numpy(np.stack(values)[None].astype(np.float32))


def comparison_loss(a,b,label):
    if label=='uncertain':return None
    if label=='tie':return (a-b).square().mean()
    sign=1 if label=='a' else -1
    return F.softplus(-sign*(a-b)).mean()


def load_quality(path):
    path=Path(path)
    if not path.is_file() or path.stat().st_size>128*1024**2:raise ValueError('Missing/excessive quality checkpoint')
    p=torch.load(path,map_location='cpu',weights_only=True)
    if p.get('format')!=QUALITY_FORMAT or p.get('production_approved') is not False:raise ValueError('Not a research clip-quality checkpoint')
    if fingerprint(p['feature_space'])!=p['feature_space_id']:raise ValueError('Quality feature-space identity changed')
    integer(p['hidden_dim'],'quality hidden dim',8,256)
    model=ClipQuality(p['feature_space']['feature_dim'],p['hidden_dim']);model.load_state_dict(p['model_state'],strict=True)
    return model.eval(),p


def quality_pass(root,model,events,optimizer=None,stop_file=None):
    losses={};correct=0;ranked=0;ties=0
    for event in events:
        for row in event.manifest['comparisons']:
            if row['label']=='uncertain':continue
            if stop_file and Path(stop_file).exists():raise InterruptedError('Offline quality learning stopped')
            a=model(clip_descriptor(root,event,row,row['a']));b=model(clip_descriptor(root,event,row,row['b']))
            loss=comparison_loss(a,b,row['label'])+.001*(a.square()+b.square()).mean()
            if not torch.isfinite(loss):raise ValueError('Non-finite clip-quality objective')
            if optimizer:
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if row['label']=='tie':ties+=1
            else:ranked+=1;correct+=float((a-b).detach()[0]*(1 if row['label']=='a' else -1)>0)
            losses.setdefault(event.manifest['original_event_id'],[]).append(float(loss.detach()))
    if not losses:raise ValueError('No reviewed non-uncertain clip comparisons')
    return dict(macro_event_loss=float(np.mean([np.mean(v) for v in losses.values()])),pair_accuracy=correct/ranked if ranked else None,
                ranked_comparisons=ranked,ties=ties,original_events=len(losses))


def train_quality(root,corpus,output,epochs=5,lr=.001,hidden_dim=32,allow_synthetic=False,stop_file=None):
    integer(epochs,'quality epochs',1,1000);integer(hidden_dim,'quality hidden dim',8,256);number(lr,'quality learning rate',1e-7,.1)
    events=load_corpus(root,corpus,allow_synthetic)
    train=[e for e in events if e.manifest['split']=='train'];val=[e for e in events if e.manifest['split']=='validation']
    if not train or not val:raise ValueError('Quality learning requires independent train and validation events')
    for group in (train,val):
        if not any(r['label'] in ('a','b') for e in group for r in e.manifest['comparisons']):raise ValueError('Need reviewed directional comparisons, not only ties/uncertainty')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Choose empty quality run output')
    torch.manual_seed(42);torch.set_num_threads(2)
    model=ClipQuality(events[0].header['feature_dim'],hidden_dim);optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    best=float('inf');history=[];qualification='synthetic_smoke_only' if any(e.manifest['source']=='synthetic' for e in train+val) else 'offline_quality_research_unqualified'
    try:
        for epoch in range(epochs):
            model.train();training=quality_pass(root,model,train,optimizer,stop_file)
            model.eval()
            with torch.no_grad():validation=quality_pass(root,model,val,stop_file=stop_file)
            row=dict(epoch=epoch+1,train=training,validation=validation);history.append(row)
            if validation['macro_event_loss']<best:
                best=validation['macro_event_loss'];save_torch(output/'quality-best.pt',dict(format=QUALITY_FORMAT,model_state=model.state_dict(),hidden_dim=hidden_dim,
                    feature_space=feature_space(events[0].header),feature_space_id=feature_space_id(events[0].header),qualification=qualification,
                    seen_data=provenance(train+val),production_approved=False,execute_allowed=False,offline_full_clip_only=True))
        report=dict(format='camix_clip_quality_training_v2',status='completed',qualification=qualification,history=history,
                    resume_supported=False,professional_evidence=False,production_approved=False,execute_allowed=False,
                    warning='Offline quality scores are not calibrated probabilities, not policy inputs and not independent director certification')
        atomic_json(output/'quality-training-report.json',report);return output/'quality-best.pt',report
    except Exception as error:
        atomic_json(output/'quality-training-report.json',dict(status='failed',error=str(error),history=history,production_approved=False));raise


def evaluate_quality(root,checkpoint,corpus,output,split='test',allow_synthetic=False):
    model,p=load_quality(checkpoint);events=[e for e in load_corpus(root,corpus,allow_synthetic) if e.manifest['split']==split]
    if not events:raise ValueError('No independent quality evaluation events')
    seen={r['original_event_id'] for r in p['seen_data']};media={h for r in p['seen_data'] for h in r['media_hashes']}
    for e in events:
        if feature_space_id(e.header)!=p['feature_space_id']:raise ValueError('Quality evaluation feature-space mismatch')
        row=provenance([e])[0]
        if row['original_event_id'] in seen or set(row['media_hashes'])&media:raise ValueError('Quality evaluation event was seen during training/model selection')
    with torch.no_grad():summary=quality_pass(root,model,events)
    result=dict(format='camix_clip_quality_evaluation_v2',checkpoint_sha256=digest_file(Path(checkpoint)),summary=summary,
                feature_space_id=p['feature_space_id'],qualification=p['qualification'],independent=True,
                quality_checkpoint=Path(checkpoint).resolve().relative_to(Path(root).resolve()).as_posix(),
                data_provenance=provenance(events),professional_evidence=False,production_approved=False)
    path=Path(output)
    if path.exists():raise ValueError('Quality evaluation report already exists')
    atomic_json(path,result);return result
