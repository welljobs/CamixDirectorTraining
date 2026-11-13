"""Complete-event OFFLINE baselines. No V2 learned policy or live TAKE exists here."""
from pathlib import Path
import uuid

import numpy as np

from .annotations_v2 import bind_media
from .contracts_v2 import canonical, number
from .features_v2 import cache_header, iter_features
from .prepare_v2 import IntervalCursor, validate_index
from .runtime import atomic_json


def replay(root, index, cache_id, output, policy='hold_valid', min_hold_s=2., stop_file=None):
    validate_index(index)
    min_hold_s = number(min_hold_s, 'minimum hold', 0, 60)
    if policy not in ('hold_valid', 'audio_rms'):
        raise ValueError('Only explicit non-learned hold_valid/audio_rms baselines are implemented')
    header = cache_header(root, cache_id)
    if header['identity']['index_sha256'] != index['index_sha256']:
        raise ValueError('Replay feature/index identity mismatch')
    m = index['document']['manifest']
    bind_media(root, m, verify=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Replay output is not empty; refusing overwrite')
    ids = [c['id'] for c in m['cameras']]
    current, last_cut = m['initial_program'], 0.
    initial_program = current
    cuts, cut_count, fallback_count, no_video, guarded, scored, correct = [], 0, 0, 0, 0, 0, 0
    covered = set()
    score_enabled = index['document']['review'] is not None
    shot_cursor = IntervalCursor(m['shot_preferences'])
    event_cursor = IntervalCursor(m['events'])
    # A metadata-only index does not certify that real-time capture/decode meets deadlines.
    trace_path = output / 'value-trace.jsonl'
    temporary = output / ('trace.' + uuid.uuid4().hex + '.partial')
    with temporary.open('w', encoding='utf-8') as trace:
        for chunk, arrays in iter_features(root, cache_id):
            for local, timestamp in enumerate(arrays['times']):
                if stop_file and Path(stop_file).exists():
                    raise InterruptedError('V2 offline replay stopped; no final acceptance report')
                t = float(timestamp)
                available = np.flatnonzero(arrays['video_mask'][local] > .5).tolist()
                input_program = current
                candidate = current
                raw_candidate = current
                reason = 'hold'
                if not available:
                    no_video += 1
                    candidate = None
                    raw_candidate = None
                    reason = 'manual_intervention_no_video'
                elif current not in [ids[s] for s in available]:
                    candidate = ids[available[0]]
                    raw_candidate = candidate
                    reason = 'signal_fallback'
                    fallback_count += 1
                elif policy == 'audio_rms':
                    audible = [s for s in available if arrays['audio_mask'][local, s] > .5]
                    if audible:
                        scores = arrays['audio_rms'][local]
                        # Prefer current program on exact ties, avoiding arbitrary slot oscillation.
                        best = max(audible, key=lambda s: (float(scores[s]), ids[s] == current))
                        raw_candidate = ids[best]
                        if ids[best] != current:
                            if t - last_cut >= min_hold_s:
                                candidate, reason = ids[best], 'non_learned_audio_rms'
                            else:
                                guarded += 1
                                reason = 'minimum_hold_guard'
                if candidate is not None and candidate != current:
                    cut_count += 1
                    if len(cuts) < 10000:
                        cuts.append(dict(time_s=t, camera_id=candidate))
                    current, last_cut = candidate, t
                chosen = candidate
                if score_enabled and chosen is not None:
                    rows = [r for r in shot_cursor.at(t) if r['context_program'] in (None, input_program)]
                    if rows:
                        accepted = {c for row in rows for c in row['acceptable_cameras']}
                        scored += 1
                        correct += chosen in accepted
                    for event in event_cursor.at(t):
                        if chosen in event['visible_cameras']:
                            covered.add(event['id'])
                trace.write(canonical(dict(event_time_s=t, decision_time_s=t + m['clock']['delay_s'],
                                           source_current_program=input_program, raw_candidate_camera=raw_candidate, selected_camera=chosen, reason=reason,
                                           execute_allowed=False)) + '\n')
    bind_media(root, m, verify=True)
    temporary.replace(trace_path)
    report = dict(format='camix_offline_baseline_v2', original_event_id=m['original_event_id'],
                  index_sha256=index['index_sha256'], cache_id=cache_id, policy=policy,
                  learned_policy=False, visual_features_used_for_decision=False,
                  clock=m['clock'], source=m['source'], split=m['split'],
                  qualification='synthetic_smoke_only' if m['source'] == 'synthetic' else 'offline_baseline_not_director_acceptance',
                  ticks=index['ticks'], cut_count=cut_count, cuts_per_minute=cut_count * 60 / m['duration_s'],
                  minimum_hold_s=min_hold_s, guard_count=guarded, fallback_count=fallback_count,
                  no_valid_video_ticks=no_video, annotated_ticks=scored,
                  acceptable_camera_rate=correct / scored if scored else None,
                  covered_event_ids=sorted(covered) if score_enabled else None,
                  event_coverage_rate=len(covered) / len(m['events']) if score_enabled and m['events'] else None,
                  variant=dict(initial_program=initial_program, cuts=cuts) if cut_count <= 10000 else None,
                  production_approved=False, execute_allowed=False,
                  warning='Non-learned offline baseline only. Correlated ticks and coarse event coverage are not professional blind evaluation or device timing.')
    atomic_json(output / 'replay-report.json', report)
    return report
