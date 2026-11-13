"""Versioned training engine: weight initialization, exact safe-boundary resume, mmap data."""
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from .config import SCHEMA_VERSION
from .data import SessionDataset,validate_split_separation
from .model import MemoryDirector
from .runtime import atomic_json
from .storage import digest_file
from .lineage import dataset_identity,start_run,model_card
from .training_state import TrainingCancelled,save_torch,capture_rng,restore_rng,Telemetry

FORMAT='camix_training_state_v2'


def run_training(data_root,output,config,epochs=5,sequence_steps=16,burn_in=4,batch_size=2,lr=.001,device='cpu',allow_synthetic=False,seed=42,allow_weak=False,balance_actions=False,stop_file=None,progress_file=None,init_checkpoint=None,resume_checkpoint=None,cache_root=None,scheduler='none',patience=0,min_delta=0.,accumulation_steps=1,precision='fp32',checkpoint_steps=10,num_workers=0,max_updates=None):
    from .train import sequence_loss,action_balance_weights
    if epochs<1 or batch_size<1 or not 0<lr<=1 or accumulation_steps<1 or checkpoint_steps<1 or not 0<=num_workers<=4:raise ValueError('Invalid training settings')
    if scheduler not in ('none','cosine','plateau') or precision not in ('fp32','amp_fp16'):raise ValueError('Unsupported scheduler/precision')
    if patience<0 or min_delta<0 or not math.isfinite(min_delta):raise ValueError('Invalid early stopping settings')
    if init_checkpoint and resume_checkpoint:raise ValueError('Choose weight initialization OR full resume, not both')
    if device not in ('cpu','mps','cuda'):raise ValueError('Unsupported training device')
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable; select CPU')
    if device=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS unavailable; select CPU')
    if precision!='fp32' and device!='cuda':raise ValueError('AMP FP16 is enabled only for available CUDA; CPU/MPS use FP32')
    output=Path(output)
    if any((output/name).exists() for name in ('best.pt','last.pt','training-report.json')):raise ValueError('Choose a fresh output directory; existing results are never overwritten')
    output.mkdir(parents=True,exist_ok=True)
    settings=dict(epochs=epochs,sequence_steps=sequence_steps,burn_in=burn_in,batch_size=batch_size,lr=lr,device=device,seed=seed,balance_actions=balance_actions,allow_synthetic=allow_synthetic,allow_weak=allow_weak,scheduler=scheduler,patience=patience,min_delta=min_delta,accumulation_steps=accumulation_steps,precision=precision,checkpoint_steps=checkpoint_steps,num_workers=num_workers)
    resumed=torch.load(resume_checkpoint,map_location='cpu',weights_only=True) if resume_checkpoint else None
    if resumed:
        if resumed.get('checkpoint_format')!=FORMAT:raise ValueError('This file has weights only, not a full training state. Use fine-tuning instead.')
        if resumed['config']!=config.to_dict():raise ValueError('Resume configuration differs from checkpoint')
        for key in settings:
            if key in ('checkpoint_steps','num_workers'):continue
            if settings[key]!=resumed['settings'][key]:raise ValueError(f'Resume setting mismatch: {key}. Restore checkpoint settings; use fine-tuning to change them.')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    started=time.monotonic();history=[];qualification='trained_unqualified';provenance=[];balance_report=None;parameter_count=0
    parent=dict(mode='resume' if resumed else 'fine_tune',sha256=digest_file(resume_checkpoint or init_checkpoint)) if resumed or init_checkpoint else None
    state=dict(epoch=0,next_batch=0,phase='train',order=None,global_step=0,train_sum=0.,train_batches=0,train_pairs=0,train_correct=0,best_loss=float('inf'),bad_epochs=0,early_best=float('inf'))
    telemetry=Telemetry(device);last_saved=None;manifest=None;model=None;best_payload=None;planned_steps=None;run_first_step=0;compute_started=started
    def stopping():return bool(stop_file and Path(stop_file).exists())
    def progress(**extra):
        completed_updates=state['global_step']-run_first_step
        eta=(time.monotonic()-compute_started)/completed_updates*max(0,planned_steps-state['global_step']) if planned_steps and completed_updates>=5 else None
        value=dict(event='progress',eta_seconds=eta,epochs=epochs,epoch=min(epochs,state['epoch']+1),optimizer_steps=state['global_step'],elapsed_seconds=time.monotonic()-started,history=history,device=device,resources=telemetry.sample(),checkpoint=last_saved,**extra)
        if progress_file:atomic_json(progress_file,value)
    def report(status,error=None):
        return dict(status=status,qualification=qualification,production_approved=False,error=error,config=config.to_dict(),history=history,training=dict(settings,provenance=provenance,parameter_count=parameter_count,action_balance=balance_report,parent=parent),checkpoint_available=(output/'best.pt').exists(),resume_checkpoint_available=(output/'last.pt').exists(),resume_boundary=last_saved)
    try:
        if stopping():raise TrainingCancelled('Stopped before dataset loading')
        progress(phase='loading',batch=0,batches=0)
        train_set=SessionDataset(data_root,'train',config,sequence_steps,burn_in,allow_synthetic,allow_weak,cache_root)
        if stopping():raise TrainingCancelled('Stopped after dataset loading')
        val_set=SessionDataset(data_root,'validation',config,sequence_steps,burn_in,allow_synthetic,allow_weak,cache_root)
        validate_split_separation(train_set,val_set)
        provenance=train_set.provenance()+val_set.provenance();fingerprint=dataset_identity(provenance)
        if resumed and resumed.get('training_code'):
            from .lineage import code_snapshot
            current_code=code_snapshot()
            if any(current_code.get(k)!=v for k,v in resumed['training_code'].items()):raise ValueError('Training code changed; restore the recorded source_snapshot or use fine-tuning, not exact resume')
        if resumed and resumed['dataset_fingerprint']!=fingerprint:raise ValueError('Dataset or annotation content changed; exact resume is refused')
        synthetic=any(p['source']=='synthetic' for p in provenance)
        qualification='synthetic_smoke_only' if synthetic else ('weak_supervision_experiment' if allow_weak else 'trained_unqualified')
        positive_weights,balance_report=action_balance_weights(train_set) if balance_actions else (None,None)
        if positive_weights is not None:positive_weights=positive_weights.to(device)
        model=MemoryDirector(replace(config,pretrained_visual=False) if parent else config).to(device)
        if init_checkpoint:
            initial=torch.load(init_checkpoint,map_location='cpu',weights_only=True)
            if initial.get('schema_version')!=SCHEMA_VERSION:raise ValueError('Unsupported initial checkpoint schema')
            try:model.load_state_dict(initial['model_state'],strict=True)
            except RuntimeError as error:raise ValueError('Initial weights do not match this architecture; choose compatible model dimensions') from error
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
        schedule=(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs) if scheduler=='cosine' else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=max(1,patience//2 or 3)) if scheduler=='plateau' else None)
        scaler=torch.amp.GradScaler('cuda',enabled=precision=='amp_fp16')
        generator=torch.Generator().manual_seed(seed)
        parameter_count=sum(p.numel() for p in model.parameters())
        if resumed:
            model.load_state_dict(resumed['model_state'],strict=True);optimizer.load_state_dict(resumed['optimizer_state'])
            if schedule:schedule.load_state_dict(resumed['scheduler_state'])
            scaler.load_state_dict(resumed['scaler_state']);state=resumed['position'];history=resumed['history'];generator.set_state(resumed['sampler_rng']);restore_rng(resumed['rng'],device)
            best_payload=resumed.get('best_state')
            if best_payload is not None:save_torch(output/'best.pt',best_payload)

        manifest=start_run(output,config,settings,provenance,parent)
        manifest['dataset_reference']=str(Path(data_root).resolve());atomic_json(output/'run-manifest.json',manifest)
        def base_payload():
            return dict(schema_version=SCHEMA_VERSION,config=config.to_dict(),model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},qualification=qualification,production_approved=False,training=dict(settings,synthetic=synthetic,provenance=provenance,parameter_count=parameter_count,action_balance=balance_report,parent=parent,metrics=history[-1] if history else None))
        def save_last():
            nonlocal last_saved
            payload=base_payload();payload.update(training_code={k:manifest['code'][k] for k in ('model.py','engine.py','train.py','data.py','training_state.py','storage.py','config.py')},checkpoint_format=FORMAT,settings=settings,position=state,history=history,optimizer_state=optimizer.state_dict(),scheduler_state=schedule.state_dict() if schedule else None,scaler_state=scaler.state_dict(),rng=capture_rng(device),sampler_rng=generator.get_state(),dataset_fingerprint=fingerprint,best_state=best_payload,best_checkpoint_sha256=digest_file(output/'best.pt') if (output/'best.pt').exists() else None)
            save_torch(output/'last.pt',payload)
            last_saved=dict(epoch=state['epoch'],next_batch=state['next_batch'],phase=state['phase'],optimizer_steps=state['global_step'])
            atomic_json(output/'checkpoint-info.json',dict(format=FORMAT,config=config.to_dict(),settings=settings,position=last_saved,qualification=qualification,production_approved=False,dataset_fingerprint=fingerprint))
        run_first_step=state['global_step'];compute_started=time.monotonic();planned_steps=epochs*math.ceil(math.ceil(len(train_set)/batch_size)/accumulation_steps)
        if not resumed:save_last()
        if patience and state['bad_epochs']>=patience:
            final=report('early_stopped');atomic_json(output/'training-report.json',final);progress(phase='early_stopped',batch=0,batches=0)
            model_card(output,output/'best.pt',final['training'],qualification)
            return output/'best.pt',final
        while state['epoch']<epochs:
            epoch=state['epoch'];model.train();optimizer.zero_grad(set_to_none=True)
            if state['order'] is None:state['order']=torch.randperm(len(train_set),generator=generator)
            total_batches=math.ceil(len(train_set)/batch_size)
            if state['phase']=='train':
                subset=Subset(train_set,state['order'][state['next_batch']*batch_size:].tolist())
                loader_generator=torch.Generator().manual_seed(seed+epoch*997)
                loader=DataLoader(subset,batch_size=batch_size,shuffle=False,generator=loader_generator,num_workers=num_workers,pin_memory=device=='cuda',**({'prefetch_factor':2} if num_workers else {}))
                begin=state['next_batch'];pending=False;last_batch_end=time.monotonic()
                for bno,batch in enumerate(loader,start=begin):
                    load_ms=(time.monotonic()-last_batch_end)*1000
                    if stopping() and not pending:save_last();raise TrainingCancelled('Stopped at a safe optimizer boundary')
                    batch={k:v.to(device) for k,v in batch.items()}
                    group_start=bno-bno%accumulation_steps;denominator=min(accumulation_steps,total_batches-group_start)
                    with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=precision=='amp_fp16'):
                        loss,pairs,correct=sequence_loss(model,batch,burn_in,augment=True,positive_weights=positive_weights)
                    if loss is not None:
                        if not torch.isfinite(loss):raise ValueError('Non-finite loss; last complete checkpoint preserved')
                        scaler.scale(loss/denominator).backward();pending=True
                        state['train_sum']+=float(loss.detach());state['train_batches']+=1;state['train_pairs']+=pairs;state['train_correct']+=correct
                    state['next_batch']=bno+1
                    boundary=(bno+1)%accumulation_steps==0 or bno+1==total_batches
                    if boundary:
                        if pending:
                            scaler.unscale_(optimizer)
                            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('Non-finite gradients; last complete checkpoint preserved')
                            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);state['global_step']+=1
                        pending=False
                        if state['global_step']%checkpoint_steps==0 or stopping():save_last()
                        if max_updates and state['global_step']>=max_updates:save_last();raise TrainingCancelled('Update budget reached (test/controlled stop)')
                        if stopping():save_last();raise TrainingCancelled('User requested stop; full training state saved')
                    last_batch_end=time.monotonic()
                    progress(phase='training',batch=bno+1,batches=total_batches,train_loss=state['train_sum']/max(1,state['train_batches']),learning_rate=optimizer.param_groups[0]['lr'],data_load_ms=load_ms,windows_per_second=(bno+1-begin)*batch_size/max(.001,time.monotonic()-started))
                if not state['train_batches']:raise ValueError('No valid preference supervision')
                state['phase']='validation';save_last()
            model.eval();val_sum=0.;val_pairs=0;val_correct=0
            val_loader=DataLoader(val_set,batch_size=batch_size,shuffle=False,generator=torch.Generator().manual_seed(seed+epoch),num_workers=0)
            with torch.no_grad():
                for index,batch in enumerate(val_loader):
                    if stopping():save_last();raise TrainingCancelled('Stopped during validation; validation will restart on resume')
                    batch={k:v.to(device) for k,v in batch.items()}
                    loss,pairs,correct=sequence_loss(model,batch,burn_in)
                    if loss is not None:
                        if not torch.isfinite(loss):raise ValueError('Non-finite validation loss')
                        val_sum+=float(loss)*pairs;val_pairs+=pairs;val_correct+=correct
                    progress(phase='validation',batch=index+1,batches=len(val_loader))
            if not val_pairs:raise ValueError('No valid validation comparisons')
            validation_loss=val_sum/val_pairs
            metrics=dict(epoch=epoch+1,train_loss=state['train_sum']/state['train_batches'],validation_loss=validation_loss,validation_pair_accuracy=val_correct/val_pairs,validation_pairs=val_pairs,learning_rate=optimizer.param_groups[0]['lr'],optimizer_steps=state['global_step'])
            history.append(metrics);print(json.dumps(metrics),flush=True)
            if validation_loss<state['best_loss']:
                state['best_loss']=validation_loss;best_payload=base_payload();save_torch(output/'best.pt',best_payload)
            if validation_loss<state['early_best']-min_delta:state['early_best']=validation_loss;state['bad_epochs']=0
            else:state['bad_epochs']+=1
            if schedule:
                if scheduler=='plateau':schedule.step(validation_loss)
                else:schedule.step()
            state.update(epoch=epoch+1,next_batch=0,phase='train',order=None,train_sum=0.,train_batches=0,train_pairs=0,train_correct=0)
            save_last();atomic_json(output/'training-report.json',report(qualification))
            if patience and state['bad_epochs']>=patience:
                final=report('early_stopped');atomic_json(output/'training-report.json',final);progress(phase='early_stopped',batch=0,batches=0);break
        else:
            final=report(qualification);atomic_json(output/'training-report.json',final);progress(phase='completed',batch=0,batches=0)
        if not (output/'best.pt').exists():raise ValueError('No best checkpoint available; full checkpoint did not contain a recoverable best state')
        model_card(output,output/'best.pt',report(qualification)['training'],qualification)
        return output/'best.pt',final
    except TrainingCancelled:
        atomic_json(output/'training-report.json',report('cancelled'));progress(phase='cancelled',batch=0,batches=0);raise
    except Exception as error:
        text=str(error)
        if 'out of memory' in text.lower():text+='; last.pt is retained. Reduce batch/sequence for a new fine-tuning run; exact resume does not silently change settings.'
        atomic_json(output/'training-report.json',report('failed',text));raise
