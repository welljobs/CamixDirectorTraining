"""Supervised, causal distillation and reviewed trajectory-preference training.

Resume is explicitly at a completed EPOCH boundary; partial epochs are replayed
from saved weights/optimizer/RNG, never continued with mismatched hidden state.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from .contracts_v2 import fingerprint, integer, number
from .corpus_v2 import corpus_identity, feature_space, load_corpus, provenance
from .data_v2 import EventFeatureSequences
from .losses_v2 import DEFAULT_WEIGHTS, distillation_loss, policy_losses
from .memory_v2 import detach_state
from .model_v2 import CHECKPOINT_FORMAT, MODEL_FAMILY, FeatureDirector, PolicyConfig
from .policy_io_v2 import load_policy
from .policy_runtime_v2 import CausalSignals, tensor_targets, tensors_for_step
from .runtime import atomic_json
from .storage import digest_file
from .training_state import Telemetry, TrainingCancelled, capture_rng, restore_rng, save_torch


def training_code():
    names = ('train_v2.py', 'model_v2.py', 'memory_v2.py', 'losses_v2.py', 'data_v2.py', 'corpus_v2.py',
             'policy_io_v2.py', 'policy_runtime_v2.py', 'features_v2.py', 'preference_v2.py',
             'contracts_v2.py','annotations_v2.py','prepare_v2.py','model.py')
    return {name: digest_file(Path(__file__).with_name(name)) for name in names if Path(__file__).with_name(name).exists()}


def merge_metrics(total, metrics):
    for name, value in metrics.items():
        if isinstance(value, dict):
            entry = total.setdefault(name, dict(sum=0., count=0))
            entry['sum'] += value['sum'];entry['count'] += value['count']
        else:
            total[name] = total.get(name, 0) + value


def metric_loss(metrics, weights=DEFAULT_WEIGHTS):
    parts = [weights[k]*metrics[k]['sum']/metrics[k]['count'] for k in weights if metrics.get(k, {}).get('count')]
    return sum(parts) if parts else None


def event_pass(root, model, event, device='cpu', optimizer=None, teacher=None, distill_weight=.5,
               stop_file=None, callback=None, ablation='full', max_updates=None):
    state = model.initial_state(device=device)
    signals = CausalSignals()
    teacher_state = teacher.initial_state(device=device) if teacher is not None else None
    teacher_signals = CausalSignals()
    metrics = {};updates = 0
    dataset = EventFeatureSequences(root, event.index, event.cache_id)
    for chunk_index, sample in enumerate(dataset):
        losses = []
        for step in range(len(sample['observations']['times'])):
            if stop_file and Path(stop_file).exists():
                raise TrainingCancelled('V2 stopped; resume repeats the uncommitted epoch')
            inputs, observed = tensors_for_step(sample['observations'], step, model.config, device, signals,
                                                event.manifest['clock'], ablation)
            if ablation == 'memory_reset_each_step':
                state = model.initial_state(device=device)
            outputs = model(*inputs, *state)
            state = outputs[4:]
            if not all(torch.isfinite(v).all() for v in outputs):
                raise ValueError('Non-finite V2 policy output')
            targets = tensor_targets(sample['targets'], step, model.config, device)
            loss, details = policy_losses(outputs, observed, targets)
            merge_metrics(metrics, details)
            if teacher is not None:
                ti, _ = tensors_for_step(sample['observations'], step, teacher.config, device, teacher_signals, event.manifest['clock'])
                with torch.no_grad():
                    expected = teacher(*ti, *teacher_state)
                teacher_state = expected[4:]
                imitation = distillation_loss(outputs, expected, observed)
                loss = (outputs[0].sum()*0 if loss is None else loss) + distill_weight*imitation
                merge_metrics(metrics, {'distillation': dict(sum=float(imitation.detach()), count=1)})
            if loss is not None:
                losses.append(loss)
        if optimizer is not None and losses:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite V2 training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step();updates += 1
        state = detach_state(state)
        if teacher_state is not None:
            teacher_state = detach_state(teacher_state)
        if callback:
            callback(chunk_index+1, len(dataset), updates)
        if max_updates is not None and updates >= max_updates:
            raise TrainingCancelled('V2 test/update budget reached; epoch is not committed')
    return metrics, updates


def aggregate_events(rows):
    groups = {}
    total = {}
    for event, metrics in rows:
        merge_metrics(total, metrics)
        loss = metric_loss(metrics)
        entry = groups.setdefault(event.manifest['original_event_id'], [])
        if loss is not None:
            entry.append(loss)
    values = [sum(v)/len(v) for v in groups.values() if v]
    return dict(macro_event_loss=sum(values)/len(values) if values else None,
                action_accuracy=total.get('correct', 0)/total['action_count'] if total.get('action_count') else None,
                action_count=total.get('action_count', 0), independent_events=len(groups), totals=total)


def _check_parent_validation(parent, validation):
    trained = [p for p in parent.get('seen_data', []) if p['split'] == 'train']
    ids = {p['original_event_id'] for p in trained}
    media = {h for p in trained for h in p.get('media_hashes', [])}
    for row in provenance(validation):
        if row['original_event_id'] in ids or set(row['media_hashes']) & media:
            raise ValueError('Validation event/source was used to train parent/teacher policy')


def train(root, corpus, output, config=None, epochs=10, lr=.0003, seed=42, device='cpu', patience=0,
          allow_synthetic=False, mode='supervised', init_checkpoint=None, resume_checkpoint=None,
          teacher_checkpoint=None, beta=.2, distill_weight=.5, stop_file=None, progress_file=None,
          max_updates=None, max_epochs_this_call=None):
    integer(epochs, 'epochs', 1, 10000);integer(seed, 'seed', 0, 2**31-1);integer(patience, 'patience', 0, 10000)
    number(lr, 'learning rate', 1e-7, .1);number(beta, 'preference beta', .01, 10);number(distill_weight, 'distill weight', 0, 10)
    if device not in ('cpu', 'mps', 'cuda') or (device == 'mps' and not torch.backends.mps.is_available()) or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested V2 training device is unavailable')
    if mode not in ('supervised', 'distill', 'preference'):
        raise ValueError('Unknown V2 training objective')
    if init_checkpoint and resume_checkpoint:
        raise ValueError('Choose fine-tuning or epoch-boundary resume, not both')
    if mode in ('distill', 'preference') and not teacher_checkpoint:
        raise ValueError('A causal V2 teacher/reference checkpoint is required')
    events = load_corpus(root, corpus, allow_synthetic)
    training = [e for e in events if e.manifest['split'] == 'train']
    validation = [e for e in events if e.manifest['split'] == 'validation']
    if not training or not validation:
        raise ValueError('V2 policy learning requires independent train AND validation events')
    space = feature_space(events[0].header)
    space_id = fingerprint(space)
    parent_payload = None;teacher_payload = None;teacher = None
    if init_checkpoint:
        _, parent_payload = load_policy(init_checkpoint)
        _check_parent_validation(parent_payload, validation)
    if teacher_checkpoint:
        teacher, teacher_payload = load_policy(teacher_checkpoint, device)
        _check_parent_validation(teacher_payload, validation)
        if teacher_payload['feature_space_id'] != space_id:
            raise ValueError('Teacher/reference feature-space mismatch')
        if mode == 'preference' and parent_payload is None and not resume_checkpoint:
            init_checkpoint = teacher_checkpoint;parent_payload = teacher_payload
    if config is None:
        if resume_checkpoint:
            _,resume_preview=load_policy(resume_checkpoint)
            config=PolicyConfig(**resume_preview['config'])
        else:
            config = PolicyConfig(**parent_payload['config']) if parent_payload else PolicyConfig(visual_dim=space['feature_dim'])
    elif isinstance(config, dict):
        config = PolicyConfig(**config)
    if config.visual_dim != space['feature_dim'] or any(len(e.manifest['cameras']) > config.max_cameras for e in training+validation):
        raise ValueError('Policy config does not fit corpus feature dimensions/camera count')
    if parent_payload and (parent_payload['config'] != config.to_dict() or parent_payload['feature_space_id'] != space_id):
        raise ValueError('Fine-tuning architecture/feature-space mismatch')
    if teacher and (teacher.config.max_cameras != config.max_cameras or (mode == 'preference' and teacher.config != config)):
        raise ValueError('Teacher/reference camera slots or preference architecture mismatch')
    if mode == 'preference':
        from .preference_v2 import usable_comparisons
        if not all(any(usable_comparisons(e) for e in part) for part in (training, validation)):
            raise ValueError('Preference learning needs reviewed non-uncertain comparisons in train and validation')
    else:
        for part in (training, validation):
            if not any(any(s['targets']['action_annotated'].any() for s in EventFeatureSequences(root, e.index, e.cache_id)) for e in part):
                raise ValueError('No legal action supervision; labels/availability/cut-window context need review')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty V2 training output; never overwrite a run')
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    model = FeatureDirector(config).to(device)
    if parent_payload:
        model.load_state_dict(parent_payload['model_state'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0001)
    settings = dict(epochs=epochs, lr=lr, seed=seed, device=device, patience=patience, mode=mode,
                    allow_synthetic=allow_synthetic, beta=beta, distill_weight=distill_weight,
                    teacher_sha256=digest_file(Path(teacher_checkpoint)) if teacher_checkpoint else None)
    code = training_code();identity = corpus_identity(training+validation)
    snapshot=output/'source_snapshot/camix_director';snapshot.mkdir(parents=True,exist_ok=True)
    for name,expected in code.items():
        source=Path(__file__).with_name(name)
        shutil.copy2(source,snapshot/name)
        if digest_file(snapshot/name)!=expected:raise ValueError('Training source changed while snapshotting')
    seen = provenance(training+validation)
    for payload in (parent_payload, teacher_payload):
        if payload:
            seen += payload.get('seen_data', [])
    seen = list({fingerprint(p): p for p in seen}.values())
    qualification = 'synthetic_smoke_only' if any(e.manifest['source'] == 'synthetic' for e in training+validation) or any(p and p.get('qualification') == 'synthetic_smoke_only' for p in (parent_payload, teacher_payload)) else 'trained_research_unqualified'
    history = [];completed = 0;best = float('inf');bad_epochs = 0;best_state = None
    if resume_checkpoint:
        loaded_model, resumed = load_policy(resume_checkpoint, device)
        if (resumed.get('resume_scope') != 'completed_epoch_boundary' or resumed.get('settings') != settings
                or resumed.get('training_code') != code or resumed.get('corpus_sha256') != identity
                or resumed.get('config') != config.to_dict() or resumed.get('feature_space_id') != space_id):
            raise ValueError('Epoch resume identity/settings/code mismatch; fine-tune instead')
        model.load_state_dict(resumed['model_state']);optimizer.load_state_dict(resumed['optimizer_state'])
        history = resumed['history'];completed = resumed['completed_epochs'];best = resumed['best_metric'];bad_epochs = resumed['bad_epochs'];best_state = resumed['best_state']
        restore_rng(resumed['rng'], device)
        seen = resumed['seen_data'];qualification = resumed['qualification']
        if completed >= epochs or (patience and bad_epochs >= patience):
            raise ValueError('Run already completed/early-stopped; use a new fine-tuning plan')
    telemetry = Telemetry(device);started = time.monotonic();total_updates = 0
    clock_profiles = list({fingerprint(e.manifest['clock']): e.manifest['clock'] for e in training+validation}.values())
    def payload(full=False):
        value = dict(format=CHECKPOINT_FORMAT, model_family=MODEL_FAMILY, config=config.to_dict(), model_state=deepcopy(model.state_dict()),
                     feature_space=space, feature_space_id=space_id, clock_profiles=clock_profiles,
                     qualification=qualification, seen_data=seen, production_approved=False, execute_allowed=False,
                     training_code=code, corpus_sha256=identity, settings=settings,
                     head_supervision=history[-1].get('train', {}).get('totals', {}) if history else {}, completed_epochs=completed)
        if full:
            value.update(resume_scope='completed_epoch_boundary', optimizer_state=optimizer.state_dict(), rng=capture_rng(device),
                         history=history, best_metric=best, bad_epochs=bad_epochs, best_state=best_state)
        return value
    def report(status, error=None):
        result = dict(format='camix_policy_training_report_v2', status=status, error=error, qualification=qualification,
                      mode=mode, config=config.to_dict(), settings=settings, feature_space_id=space_id,
                      corpus_sha256=identity, completed_epochs=completed, history=history,
                      resume_scope='completed_epoch_boundary; partial epoch replays from saved state',
                      elapsed_s=time.monotonic()-started, production_approved=False, execute_allowed=False,
                      warning='Engineering/label objectives do not establish aesthetics, narrative quality or production fitness')
        atomic_json(output/'training-report.json', result)
        return result
    def progress(**fields):
        if progress_file:
            atomic_json(progress_file, dict(phase=fields.pop('phase', 'training'), epochs=epochs, epoch=completed+1,
                                           resources=telemetry.sample(), elapsed_seconds=time.monotonic()-started,
                                           history=history, **fields))
    save_torch(output/'last.pt', payload(True))
    if best_state:
        save_torch(output/'best.pt', best_state)
    atomic_json(output/'run-manifest.json', dict(settings=settings, config=config.to_dict(), corpus_sha256=identity,
                                                seen_data=seen, training_code=code, feature_space_id=space_id,
                                                parent_sha256=digest_file(Path(init_checkpoint)) if init_checkpoint else None,
                                                production_approved=False))
    try:
        for epoch in range(completed, epochs):
            model.train();train_rows=[]
            order = torch.randperm(len(training)).tolist()
            if mode == 'preference':
                from .preference_v2 import preference_epoch
                train_summary = preference_epoch(root, model, teacher, [training[i] for i in order], beta, optimizer, stop_file, progress)
            else:
                for event_no, i in enumerate(order):
                    event = training[i]
                    def callback(chunk, chunks, updates):
                        progress(batch=chunk, batches=chunks, event=event_no+1, events=len(training))
                    with torch.enable_grad():
                        metrics, updates = event_pass(root, model, event, device, optimizer,
                                                      teacher if mode == 'distill' else None, distill_weight,
                                                      stop_file, callback, max_updates=max_updates-total_updates if max_updates else None)
                    total_updates += updates;train_rows.append((event, metrics))
                train_summary = aggregate_events(train_rows)
            model.eval()
            with torch.no_grad():
                if mode == 'preference':
                    from .preference_v2 import preference_epoch
                    val_summary = preference_epoch(root, model, teacher, validation, beta, None, stop_file, None)
                else:
                    rows = [(e, event_pass(root, model, e, device, stop_file=stop_file)[0]) for e in validation]
                    val_summary = aggregate_events(rows)
            score = val_summary['macro_event_loss']
            if score is None or not np.isfinite(score):
                raise ValueError('No finite independent validation objective')
            completed = epoch+1
            row = dict(epoch=completed, train=train_summary, validation=val_summary, validation_loss=score,
                       validation_pair_accuracy=val_summary.get('action_accuracy'))
            history.append(row);print(__import__('json').dumps(row, allow_nan=False), flush=True)
            if score < best:
                best = score;bad_epochs = 0;best_state = payload(False);save_torch(output/'best.pt', best_state)
            else:
                bad_epochs += 1
            save_torch(output/'last.pt', payload(True));report('running')
            if patience and bad_epochs >= patience:
                return output/'best.pt', report('early_stopped')
            if max_epochs_this_call is not None and completed >= max_epochs_this_call and completed < epochs:
                raise TrainingCancelled('Stopped at a committed epoch boundary')
        return output/'best.pt', report('completed')
    except TrainingCancelled as error:
        report('cancelled', str(error));raise
    except Exception as error:
        report('failed', str(error));raise
