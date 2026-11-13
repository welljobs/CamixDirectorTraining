import argparse
from dataclasses import replace
import json,random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .config import ModelConfig,SCHEMA_VERSION,INPUT_NAMES
from .model import MemoryDirector
from .data import SessionDataset,validate_split_separation


def preference_loss(values, targets, camera_mask, current_program,positive_weights=None):
    # Current-camera TAKE duplicates KEEP and is not a separate action.
    valid = torch.cat((torch.ones_like(camera_mask[:,:1]),camera_mask*(1-current_program)),dim=1) > .5
    annotated = torch.isfinite(targets) & valid
    safe = torch.nan_to_num(targets)
    pairs = (safe[:,:,None] > safe[:,None,:]) & annotated[:,:,None] & annotated[:,None,:]
    differences = values[:,:,None] - values[:,None,:]
    count = pairs.sum()
    losses=F.softplus(-differences) * pairs
    if positive_weights is not None:losses=losses*positive_weights[None,:,None]
    loss = losses.sum() / count.clamp(min=1)
    correct = ((differences > 0) & pairs).sum()
    return loss,count,correct


def sequence_loss(model,batch,burn_in,augment=False,positive_weights=None):
    b,t,n = batch["camera_mask"].shape
    if augment:
        permutation = torch.randperm(n,device=batch["camera_mask"].device)
        for name in ("video_frames","audio_pcm","modality_mask","camera_mask","media_age","current_program"):
            batch[name] = batch[name].index_select(2,permutation)
        targets=batch["preferences"]
        batch["preferences"] = torch.cat((targets[:,:,:1],targets[:,:,1:].index_select(2,permutation)),dim=2)
    memory,mask=model.initial_memory(b,batch["video_frames"].device)
    losses=[];pairs=0;correct=0
    for step in range(t):
        args=[batch[name][:,step] for name in INPUT_NAMES[:-2]]+[memory,mask]
        if step < burn_in:
            with torch.no_grad(): result=model(*args)
            memory,mask=result[-2].detach(),result[-1].detach()
            continue
        keep,take,_,memory,mask=model(*args)
        loss,count,ok=preference_loss(torch.cat((keep,take),dim=1),batch["preferences"][:,step],batch["camera_mask"][:,step],batch["current_program"][:,step],positive_weights)
        if int(count):losses.append(loss);pairs+=int(count);correct+=int(ok)
    if not losses:
        return None,0,0
    return torch.stack(losses).mean(),pairs,correct


def action_balance_weights(dataset):
    counts=np.zeros(2,np.float64)
    for pack in dataset.packs:
        pref=pack['preferences'];valid=np.concatenate((np.ones_like(pack['camera_mask'][:,:1]),pack['camera_mask']*(1-pack['current_program'])),1)>.5
        annotated=np.isfinite(pref)&valid;safe=np.nan_to_num(pref)
        pairs=(safe[:,:,None]>safe[:,None,:])&annotated[:,:,None]&annotated[:,None,:]
        counts[0]+=pairs[:,0,:].sum();counts[1]+=pairs[:,1:,:].sum()
    if np.any(counts==0):raise ValueError('Action balancing needs both KEEP-better and TAKE-better training comparisons')
    weights=np.minimum(counts.sum()/(2*counts),10.)
    result=np.array([weights[0]]+[weights[1]]*dataset.config.max_cameras,np.float32)
    return torch.from_numpy(result),dict(keep_positive_pairs=int(counts[0]),take_positive_pairs=int(counts[1]),positive_weights=result.tolist())


from .training_state import TrainingCancelled


def train(*args,**kwargs):
    from .engine import run_training
    return run_training(*args,**kwargs)


def load_checkpoint(path):
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    if checkpoint.get("schema_version")!=SCHEMA_VERSION:raise ValueError("unsupported checkpoint")
    config=ModelConfig(**checkpoint["config"])
    # Loading an existing checkpoint must not download any new pretrained weights.
    model=MemoryDirector(replace(config,pretrained_visual=False))
    model.load_state_dict(checkpoint["model_state"],strict=True);model.eval()
    return model,checkpoint


def main():
    p=argparse.ArgumentParser(description="Train causal audiovisual preference values; never scene classes")
    p.add_argument("--data",required=True);p.add_argument("--output",required=True);p.add_argument("--config",required=True)
    p.add_argument("--epochs",type=int,default=5);p.add_argument("--sequence-steps",type=int,default=16)
    p.add_argument("--burn-in",type=int,default=4);p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--init-checkpoint");p.add_argument("--resume-checkpoint");p.add_argument("--cache-root")
    p.add_argument("--scheduler",choices=["none","cosine","plateau"],default="none")
    p.add_argument("--patience",type=int,default=0);p.add_argument("--min-delta",type=float,default=0.)
    p.add_argument("--accumulation-steps",type=int,default=1);p.add_argument("--checkpoint-steps",type=int,default=10)
    p.add_argument("--precision",choices=["fp32","amp_fp16"],default="fp32");p.add_argument("--num-workers",type=int,default=0)
    p.add_argument("--lr",type=float,default=.001);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--stop-file");p.add_argument("--progress-file")
    p.add_argument("--allow-weak-supervision",action="store_true")
    p.add_argument("--balance-actions",action="store_true",help="Importance-weight KEEP/TAKE ranking pairs using training counts only")
    p.add_argument("--device",choices=["cpu","mps","cuda"],default="cpu");p.add_argument("--allow-synthetic",action="store_true")
    a=p.parse_args();config=ModelConfig(**json.loads(Path(a.config).read_text(encoding="utf-8")))
    try:
        train(a.data,a.output,config,a.epochs,a.sequence_steps,a.burn_in,a.batch_size,lr=a.lr,seed=a.seed,device=a.device,allow_synthetic=a.allow_synthetic,allow_weak=a.allow_weak_supervision,balance_actions=a.balance_actions,stop_file=a.stop_file,progress_file=a.progress_file,init_checkpoint=a.init_checkpoint,resume_checkpoint=a.resume_checkpoint,cache_root=a.cache_root,scheduler=a.scheduler,patience=a.patience,min_delta=a.min_delta,accumulation_steps=a.accumulation_steps,checkpoint_steps=a.checkpoint_steps,precision=a.precision,num_workers=a.num_workers)
    except TrainingCancelled as error:
        print(str(error),flush=True);raise SystemExit(130)

if __name__=="__main__":main()
