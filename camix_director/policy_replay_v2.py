"""Causal full-event learned/baseline replay, with raw/protected decisions apart."""
from pathlib import Path
import numpy as np

from .annotations_v2 import bind_media
from .contracts_v2 import canonical, number
from .corpus_v2 import feature_space_id
from .data_v2 import policy_targets
from .features_v2 import cache_header, iter_features
from .policy_io_v2 import checkpoint_hash
from .policy_runtime_v2 import StreamingDirector
from .prepare_v2 import IntervalCursor, validate_index
from .runtime import atomic_json
from .storage import digest_file


def feature_packet(index, header, arrays, local, current, held, generation):
    result = dict(format='camix_feature_packet_v2', event_time_s=float(arrays['times'][local]),
                  clock=index['document']['manifest']['clock'], generation=generation,
                  feature_space_id=feature_space_id(header), camera_ids=header['camera_ids'],
                  current_program=current, held_seconds=float(held),
                  style_features=[index['document']['manifest']['style'].get(k,0.) for k in ('pace','reaction_bias','detail_bias')])
    for key in ('visual', 'audio_rms', 'video_mask', 'audio_mask', 'video_end_s', 'video_available_s', 'audio_end_s', 'audio_available_s'):
        result[key] = arrays[key][local].tolist()
    return result


def replay_policy(root, checkpoint, index, cache_id, output, min_hold_s=2., min_probability=0.,
                  policy='learned', stop_file=None):
    validate_index(index)
    if policy not in ('learned', 'hold_valid', 'audio_rms'):
        raise ValueError('Unknown matched-information replay policy')
    min_hold_s = number(min_hold_s, 'minimum hold', 0, 60)
    actor = StreamingDirector(checkpoint, min_hold_s, min_probability)
    header = cache_header(root, cache_id)
    if header['identity']['index_sha256'] != index['index_sha256'] or feature_space_id(header) != actor.checkpoint['feature_space_id']:
        raise ValueError('Policy replay encoder/index identity mismatch')
    m = index['document']['manifest'];bind_media(root, m, verify=True)
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Policy replay output exists; choose an empty directory')
    current, last_cut = m['initial_program'], 0.
    ids = [c['id'] for c in m['cameras']]
    cuts=[];cut_count=0;guarded=0;no_video=0;scored=0;correct=0;covered=set();fallback=0
    shots = IntervalCursor(m['shot_preferences']);events = IntervalCursor(m['events'])
    trace_path = output/'value-trace.jsonl';partial = output/'value-trace.partial'
    with partial.open('w', encoding='utf-8') as trace:
        for chunk, arrays in iter_features(root, cache_id):
            for local, time_s in enumerate(arrays['times']):
                if stop_file and Path(stop_file).exists():
                    raise InterruptedError('Policy replay cancelled; no final acceptance report')
                t=float(time_s);held=t-last_cut;input_program=current
                packet=feature_packet(index, header, arrays, local, current, held, index['media_identity'])
                result=actor.step(packet)
                available=[ids[i] for i in np.flatnonzero(arrays['video_mask'][local]>.5)]
                if policy != 'learned':
                    selected=raw=current if current in available else available[0] if available else None
                    reason='hold' if current in available else 'signal_fallback' if available else 'manual_intervention_no_video'
                    if policy=='audio_rms' and current in available:
                        audible=[i for i,c in enumerate(ids) if c in available and arrays['audio_mask'][local,i]>.5]
                        if audible:
                            best=max(audible,key=lambda i:(float(arrays['audio_rms'][local,i]),ids[i]==current));raw=ids[best]
                            if raw!=current:
                                if held>=min_hold_s:selected=raw;reason='non_learned_audio_rms'
                                else:reason='experimental_guard'
                    result.update(raw_candidate_camera=raw,selected_camera=selected,reason=reason,raw_action_probability=None)
                chosen=result['selected_camera']
                if chosen is not None and chosen not in available:
                    raise ValueError('Execution guard returned an invalid camera')
                no_video+=not bool(available);guarded+=result['reason']=='experimental_guard';fallback+=result['reason']=='signal_fallback'
                if index['document']['review'] and chosen is not None:
                    active=shots.at(t)
                    target=policy_targets(m,t,input_program,arrays['video_mask'][local],active,events.at(t))
                    if target['action_annotated']:
                        scored+=1;correct+=bool(target['policy_acceptable_cameras'][ids.index(chosen)])
                    for event in events.at(t):
                        if chosen in event['visible_cameras']:covered.add(event['id'])
                if chosen is not None and chosen!=current:
                    cut_count+=1
                    if len(cuts)<50000:cuts.append(dict(time_s=t,camera_id=chosen))
                    current,last_cut=chosen,t
                trace.write(canonical(result)+'\n')
    bind_media(root,m,verify=True);partial.replace(trace_path)
    report=dict(format='camix_policy_replay_v2',checkpoint_sha256=checkpoint_hash(checkpoint),
                feature_space_id=actor.checkpoint['feature_space_id'],index_sha256=index['index_sha256'],
                source_content_sha256=index['document']['content_sha256'],media_identity=index['media_identity'],cache_id=cache_id,
                original_event_id=m['original_event_id'],source=m['source'],split=m['split'],clock=m['clock'],
                qualification=actor.checkpoint['qualification'],policy=policy,learned_policy=policy=='learned',
                ticks=index['ticks'],cut_count=cut_count,cuts_per_minute=cut_count*60/m['duration_s'],
                nominal_mean_shot_s=m['duration_s']/(cut_count+1),minimum_hold_s=min_hold_s,minimum_probability=min_probability,
                probability_calibrated=False,guard_count=guarded,fallback_count=fallback,no_valid_video_ticks=no_video,
                annotated_ticks=scored,acceptable_camera_rate=correct/scored if scored else None,
                covered_event_ids=sorted(covered),event_coverage_rate=len(covered)/len(m['events']) if index['document']['review'] and m['events'] else None,
                variant=dict(initial_program=m['initial_program'],cuts=cuts) if cut_count<=50000 else None,
                trace_sha256=digest_file(trace_path),production_approved=False,execute_allowed=False,
                warning='Offline simulated PGM; correlated ticks and coarse coverage are not professional blind assessment or live device latency.')
    atomic_json(output/'replay-report.json',report)
    return report
