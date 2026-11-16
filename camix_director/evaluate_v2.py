"""Independent-event teacher-context ablations plus matched closed-loop policies."""
from pathlib import Path
import numpy as np
import torch

from .contracts_v2 import fingerprint
from .corpus_v2 import feature_space_id, load_corpus, provenance
from .policy_io_v2 import checkpoint_hash, load_policy
from .policy_replay_v2 import replay_policy
from .runtime import atomic_json
from .storage import digest_file
from .train_v2 import aggregate_events, event_pass, metric_loss


def evaluate(root, checkpoint, corpus, output, split='test', allow_synthetic=False, allow_seen=False, min_hold_s=2., stop_file=None):
    model,payload=load_policy(checkpoint);torch.set_num_threads(2)
    events=[e for e in load_corpus(root,corpus,allow_synthetic) if e.manifest['split']==split]
    if not events:raise ValueError('No events in requested evaluation split')
    seen_ids={p['original_event_id'] for p in payload['seen_data']}
    seen_media={h for p in payload['seen_data'] for h in p.get('media_hashes',[])}
    rows=[];overlap=[]
    for e in events:
        if feature_space_id(e.header)!=payload['feature_space_id']:raise ValueError('Evaluation feature-space mismatch')
        p=provenance([e])[0]
        if p['original_event_id'] in seen_ids or set(p['media_hashes']) & seen_media:overlap.append(p['original_event_id'])
    if overlap and not allow_seen:raise ValueError('Evaluation event/source was used in training/model selection; use a new holdout or explicit diagnostic mode')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Choose empty evaluation output')
    with torch.no_grad():
        for e in events:
            for ablation in ('full','video_content_muted','audio_muted','memory_reset_each_step'):
                metrics,_=event_pass(root,model,e,stop_file=stop_file,ablation=ablation)
                rows.append(dict(original_event_id=e.manifest['original_event_id'],ablation=ablation,
                                 action_accuracy=metrics.get('correct',0)/metrics['action_count'] if metrics.get('action_count') else None,
                                 action_count=metrics.get('action_count',0),loss=metric_loss(metrics),mode='teacher_pre_decision_pgm'))
    replays=[]
    for e in events:
        identifier=fingerprint(dict(index=e.index['index_sha256']))[:16]
        for policy in ('learned','hold_valid','audio_rms'):
            result=replay_policy(root,checkpoint,e.index,e.cache_id,output/identifier/policy,min_hold_s,policy=policy,stop_file=stop_file)
            report_path=output/identifier/policy/'replay-report.json'
            replays.append(dict(result,report_path=report_path.resolve().relative_to(Path(root).resolve()).as_posix(),report_sha256=digest_file(report_path)))
    full=[r['action_accuracy'] for r in rows if r['ablation']=='full' and r['action_accuracy'] is not None]
    interval=None
    if len(full)>=3:
        rng=np.random.default_rng(42);means=np.asarray(full)[rng.integers(0,len(full),(2000,len(full)))].mean(1)
        interval=np.percentile(means,[2.5,97.5]).tolist()
    contributions={}
    for ablation in ('video_content_muted','audio_muted','memory_reset_each_step'):
        differences=[]
        for e in events:
            pair={r['ablation']:r['action_accuracy'] for r in rows if r['original_event_id']==e.manifest['original_event_id']}
            if pair.get('full') is not None and pair.get(ablation) is not None:differences.append(pair['full']-pair[ablation])
        contributions[ablation]=float(np.mean(differences)) if differences else None
    report=dict(format='camix_policy_evaluation_v2',checkpoint_sha256=checkpoint_hash(checkpoint),
                feature_space_id=payload['feature_space_id'],qualification=payload['qualification'],
                split=split,diagnostic_seen_events=sorted(set(overlap)),independent=not overlap,
                original_events=len({e.manifest['original_event_id'] for e in events}),data_provenance=provenance(events),
                teacher_context=rows,macro_event_action_accuracy=float(np.mean(full)) if full else None,
                event_bootstrap_95_interval=interval,contributions=contributions,closed_loop=replays,
                professional_evidence=False,production_approved=False,execute_allowed=False,
                warning='Ablation/label metrics do not establish aesthetics or narrative. Bootstrap is across original events, never adjacent ticks; encoder pretraining provenance remains to be audited.')
    atomic_json(output/'evaluation.json',report)
    return report
