"""Bounded feature sequences with strict separation of observations and targets.

This is a dataset interface, not a V2 trainer or professional-quality model.
"""
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from .annotations_v2 import bind_media
from .contracts_v2 import INTENTS
from .features_v2 import cache_header, iter_features
from .prepare_v2 import IntervalCursor, validate_index


class EventFeatureSequences(IterableDataset):
    def __init__(self, root, index, cache_id):
        validate_index(index)
        if index['config']['purpose'] != 'training':
            raise ValueError('Preview indexes cannot become training samples')
        bind_media(root, index['document']['manifest'], verify=True)
        self.root, self.index, self.cache_id = root, index, cache_id
        header = cache_header(root, cache_id)
        if header['identity']['index_sha256'] != index['index_sha256']:
            raise ValueError('Features belong to another event/config/review revision')

    def __len__(self):
        return len(self.index['chunks'])

    def __iter__(self):
        if get_worker_info() is not None:
            raise ValueError('Single-event stateful sequences require num_workers=0; parallelize independent events instead')
        m = self.index['document']['manifest']
        ids = [c['id'] for c in m['cameras']]
        last_cut = 0.
        current = m['initial_program']
        event_index = 0
        cursor = IntervalCursor(m['shot_preferences'])
        event_cursor = IntervalCursor(m['events'])
        for ordinal, (chunk, arrays) in enumerate(iter_features(self.root, self.cache_id)):
            count = len(arrays['times'])
            pgm = np.zeros((count, len(ids)), np.float32)
            held = np.zeros((count, 1), np.float32)
            acceptable = np.zeros((count, len(ids)), np.float32)
            annotated = np.zeros(count, bool)
            extra = {key: [] for key in ('policy_acceptable_cameras', 'action_annotated', 'switch_target', 'switch_mask', 'intent_distribution', 'intent_mask', 'visibility_target', 'visibility_mask')}
            context_skipped = 0
            for tick, t in enumerate(arrays['times']):
                while event_index < len(m['program_events']) and m['program_events'][event_index]['time_s'] < t:
                    event = m['program_events'][event_index]
                    if current != event['camera_id']:
                        current, last_cut = event['camera_id'], event['time_s']
                    event_index += 1
                pgm[tick, ids.index(current)] = 1
                held[tick, 0] = t - last_cut
                # Relative-context annotations never silently follow a different PGM.
                active = cursor.at(float(t))
                rows = [r for r in active if r['context_program'] in (None, current)]
                context_skipped += len(active) - len(rows)
                accepted = {camera for row in rows for camera in row['acceptable_cameras']}
                for slot, camera in enumerate(ids):
                    acceptable[tick, slot] = camera in accepted and arrays['video_mask'][tick, slot] > .5
                annotated[tick] = bool(acceptable[tick].any())
                values = policy_targets(m, float(t), current, arrays['video_mask'][tick], rows, event_cursor.at(float(t)))
                for key in extra:
                    extra[key].append(values[key])
            observations = {key: arrays[key] for key in arrays}
            observations.update(current_program=pgm, held_seconds=held)
            yield dict(observations=observations,
                       targets=dict(acceptable_cameras=acceptable, annotated=annotated, **{key: np.asarray(value) for key, value in extra.items()}),
                       state=dict(original_event_id=m['original_event_id'], chunk_index=ordinal,
                                  tick_start=chunk['tick_start'], reset_event=ordinal == 0,
                                  history_policy='pre_decision_teacher_pgm; rebuild_before_random_access'),
                       context_skipped=context_skipped)


def policy_targets(manifest, time_s, current_program, available, rows, active_events):
    """Old absolute labels remain intact; policy targets honor explicit cut windows.

    A permitted interval is not an obligation to switch at every tick. Outside
    that interval, an off-current camera is not converted into an invented KEEP.
    """
    ids = [c['id'] for c in manifest['cameras']]
    rows = [r for r in rows if r['context_program'] in (None, current_program)]
    applicable = [r for r in rows if r['cut_window_s'] is None or current_program in r['acceptable_cameras']
                  or r['cut_window_s'][0] <= time_s <= r['cut_window_s'][1]]
    accepted = {c for r in applicable for c in r['acceptable_cameras']}
    good = np.array([c in accepted and available[i] > .5 for i, c in enumerate(ids)], np.float32)
    selected = {ids[i] for i in np.flatnonzero(good)}
    switch_mask = bool(selected) and (current_program not in selected or selected == {current_program})
    intent = np.zeros(len(INTENTS), np.float32)
    for row in applicable:
        if row['intent'] != 'unknown':
            intent[INTENTS.index(row['intent'])] += 1
    if intent.sum():
        intent /= intent.sum()
    visibility = np.zeros(len(ids), np.float32)
    importance = max((e['importance'] for e in active_events), default=0.)
    for event in active_events:
        if event['importance'] > 0:
            for i, c in enumerate(ids):
                visibility[i] = max(visibility[i], float(c in event['visible_cameras']))
    return dict(policy_acceptable_cameras=good, action_annotated=bool(good.any()),
                switch_target=float(current_program not in selected), switch_mask=switch_mask,
                intent_distribution=intent, intent_mask=bool(intent.sum()),
                visibility_target=visibility, visibility_mask=np.asarray(available, np.float32)*importance)
