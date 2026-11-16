"""Research-stage evidence checks. Never manufactures professional/production approval."""
from pathlib import Path

from .contracts_v2 import load_json
from .policy_io_v2 import checkpoint_hash, load_policy
from .runtime import atomic_json
from .storage import digest_file
from .v2 import artifact_path


def assess(root,checkpoint,evaluation,output=None,export_report=None,shadow_report=None,quality_evaluation=None):
    model,p=load_policy(checkpoint);sha=checkpoint_hash(checkpoint)
    ev=load_json(artifact_path(root,str(evaluation)));checks=[]
    def add(name,passed,detail):checks.append(dict(name=name,passed=bool(passed),detail=detail))
    add('evaluation_identity',ev.get('format')=='camix_policy_evaluation_v2' and ev.get('checkpoint_sha256')==sha,'Evaluation must identify this policy checkpoint')
    provenance=ev.get('data_provenance',[]);ids={r.get('original_event_id') for r in provenance}
    seen_ids={r['original_event_id'] for r in p['seen_data']};seen_media={h for r in p['seen_data'] for h in r['media_hashes']}
    independent=bool(provenance) and not(ids&seen_ids) and not any(set(r.get('media_hashes',[]))&seen_media for r in provenance)
    add('independent_events',independent and len(ids)>=3,f'{len(ids)} original events; >=3 is a project research floor, not statistical/professional certification')
    reviewed=bool(provenance) and all(r.get('source')=='real' and r.get('review') for r in provenance)
    add('real_reviewed_data',reviewed and p['qualification']!='synthetic_smoke_only','Real content-bound local reviews; reviewer identity remains self-declared')
    for key in ('video_content_muted','memory_reset_each_step'):
        difference=ev.get('contributions',{}).get(key)
        add(key+'_contribution',difference is not None and difference>0,'Mean across-event contribution '+str(difference)+'; positive is not statistical significance')
    loops=[r for r in ev.get('closed_loop',[]) if r.get('policy')=='learned'];verified=bool(loops) and {r.get('original_event_id') for r in loops}==ids
    for row in loops:
        try:
            path=artifact_path(root,row['report_path']);actual=load_json(path)
            verified &= (digest_file(path)==row['report_sha256'] and actual.get('checkpoint_sha256')==sha
                         and actual.get('execute_allowed') is False and actual.get('original_event_id')==row.get('original_event_id') and digest_file(path.parent/'value-trace.jsonl')==actual['trace_sha256'])
        except (ValueError,OSError,KeyError):verified=False
    add('closed_loop_trace_identity',verified,'Requires same-checkpoint replay reports and their actual traces')
    if export_report:
        exp=load_json(artifact_path(root,str(export_report)))
        add('policy_export_parity',exp.get('checkpoint_sha256')==sha and exp.get('parity_passed') is True,'Policy-step parity only, not encoder/app/device acceptance')
    else:add('policy_export_parity',False,'Missing same-checkpoint export report')
    if shadow_report:
        sh=load_json(artifact_path(root,str(shadow_report)))
        add('recorded_shadow',sh.get('checkpoint_sha256')==sha and sh.get('packets',0)>0 and sh.get('execute_allowed') is False,'Recorded feature-packet boundary, not a live RTC trial')
    else:add('recorded_shadow',False,'Missing recorded shadow evidence')
    if quality_evaluation:
        q=load_json(artifact_path(root,str(quality_evaluation)))
        valid_quality=False
        try:
            from .quality_v2 import load_quality
            quality_path=artifact_path(root,q['quality_checkpoint']);_,quality_payload=load_quality(quality_path)
            prior_ids={r['original_event_id'] for r in quality_payload['seen_data']}
            prior_media={h for r in quality_payload['seen_data'] for h in r['media_hashes']}
            qdata=q.get('data_provenance',[])
            valid_quality=(q.get('format')=='camix_clip_quality_evaluation_v2' and q.get('summary',{}).get('ranked_comparisons',0)>0
                and q.get('feature_space_id')==p['feature_space_id'] and digest_file(quality_path)==q['checkpoint_sha256']
                and q.get('qualification')!='synthetic_smoke_only' and bool(qdata)
                and all(r['source']=='real' and r.get('review') and r['original_event_id'] not in prior_ids and not(set(r['media_hashes'])&prior_media) for r in qdata))
        except (ValueError,OSError,KeyError):pass
        add('offline_quality_model_evaluation',valid_quality,'Same-feature-space quality checkpoint, independently reviewed real-event labels; not expert certification')
    else:add('offline_quality_model_evaluation',False,'Missing independent quality-model evaluation')
    result=dict(format='camix_research_gates_v2',checkpoint_sha256=sha,status='offline_research_candidate' if all(c['passed'] for c in checks) else 'blocked',checks=checks,
                production_approved=False,execute_allowed=False,
                missing_production_evidence=['Independent qualified-director blind assessment','Audited encoder/training provenance and multi-scene generalization',
                                             'Probability calibration on independent real events','Complete encoder+policy+adapter target-device latency/thermal/fault acceptance',
                                             'Real RTC shadow acceptance, operator takeover and rollback review'],
                warning='Even all research checks passing never authorizes live programme control.')
    if output:
        path=Path(output)
        if path.exists():raise ValueError('Gate report exists')
        atomic_json(path,result)
    return result
