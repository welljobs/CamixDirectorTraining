"""Long-event metadata indexes and leakage checks, without schema-1 conversion.

Indexing never loads a whole event into RAM and never starts training.
"""
import math
from pathlib import Path

from .annotations_v2 import bind_media, validate_document
from .contracts_v2 import (INDEX_FORMAT, fingerprint, integer, media_identity,
                           number, program_at)
from .runtime import atomic_json


def index_event(document, decision_hz=2., chunk_steps=64, history_s=8., purpose='preview', allow_synthetic=False):
    doc = validate_document(document)
    m = doc['manifest']
    hz = number(decision_hz, 'decision Hz', .1, 30)
    integer(chunk_steps, 'chunk steps', 1, 256)
    history = number(history_s, 'history seconds', 0, 600)
    if purpose not in ('preview', 'training') or type(allow_synthetic) is not bool:
        raise ValueError('Invalid index purpose/synthetic acknowledgment')
    if purpose == 'training':
        if not doc.get('review'):
            raise ValueError('Training index requires content-bound review')
        if not m['license']['usage_confirmed']:
            raise ValueError('Authorized source usage required')
        if m['source'] == 'synthetic' and not allow_synthetic:
            raise ValueError('Synthetic training index requires explicit acknowledgment')
        if not any(m[k] for k in ('shot_preferences', 'events', 'comparisons')):
            raise ValueError('No supervision')
    ticks = math.ceil(m['duration_s'] * hz - 1e-9)
    if math.ceil(ticks / chunk_steps) > 10000:
        raise ValueError('Event index exceeds 10000 chunks; increase chunk_steps or reduce decision_hz')
    history_steps = math.ceil(history * hz)
    chunks = [dict(tick_start=start, tick_end=min(start + chunk_steps, ticks),
                   context_start=max(0, start - history_steps), reset_event=start == 0)
              for start in range(0, ticks, chunk_steps)]
    result = dict(format=INDEX_FORMAT, document=doc, media_identity=media_identity(m),
                  config=dict(decision_hz=hz, chunk_steps=chunk_steps, history_s=history,
                              purpose=purpose, allow_synthetic=allow_synthetic),
                  ticks=ticks, chunks=chunks, label_policy='separate_targets_never_observation_features',
                  state_policy='carry_contiguous_or_rebuild_history_never_join_shuffled_chunks',
                  v2_model_available=False, production_approved=False, execute_allowed=False)
    result['index_sha256'] = fingerprint(result)
    return result


def validate_index(value):
    if not isinstance(value, dict) or value.get('format') != INDEX_FORMAT:
        raise ValueError('Not a V2 event index')
    expected = index_event(value['document'], **value['config'])
    if value != expected:
        raise ValueError('Index identity/config/chunk contents changed')
    return value


def prepare_index(root, document, output=None, **config):
    doc = validate_document(document)
    bind_media(root, doc['manifest'], verify=True)
    result = index_event(doc, **config)
    if output:
        path = Path(output)
        if path.exists():
            raise ValueError('Index output exists; refusing overwrite')
        atomic_json(path, result)
    return result


def iter_ticks(index, chunk):
    validate_index(index)
    if chunk not in index['chunks']:
        raise ValueError('Chunk does not belong to this event index')
    hz = index['config']['decision_hz']
    m = index['document']['manifest']
    for tick in range(chunk['tick_start'], chunk['tick_end']):
        t = tick / hz
        yield dict(tick=tick, event_time_s=t, decision_time_s=t + m['clock']['delay_s'],
                   reference_program=program_at(m['initial_program'], m['program_events'], t, include_at=False))


def supervision_at(document, event_time_s, current_program):
    """Absolute camera labels only; relative context mismatches become unscored."""
    m = validate_document(document)['manifest']
    t = number(event_time_s, 'supervision time', 0, m['duration_s'])
    if current_program not in {c['id'] for c in m['cameras']}:
        raise ValueError('Unknown current program')
    labels = []
    for row in m['shot_preferences']:
        if row['start_s'] <= t < row['end_s']:
            if row['context_program'] is None or row['context_program'] == current_program:
                labels.append(row)
    return labels


def collection_quality(root, documents):
    if not isinstance(documents, list) or not 1 <= len(documents) <= 1000:
        raise ValueError('Select 1..1000 V2 event revisions')
    errors, events, sources, splits = [], {}, {}, {}
    reviewed = 0
    synthetic = 0
    for value in documents:
        doc = validate_document(value)
        m = doc['manifest']
        bind_media(root, m, verify=True)
        identifier, split = m['original_event_id'], m['split']
        if identifier in events:
            errors.append('Select one full revision per original event; duplicate: ' + identifier)
        events[identifier] = split
        splits[split] = splits.get(split, 0) + 1
        reviewed += doc['review'] is not None
        synthetic += m['source'] == 'synthetic'
        for c in m['cameras']:
            for key in ('video_sha256', 'audio_sha256'):
                sha = c[key]
                if sha is None:
                    continue
                if sha in sources and sources[sha] != split:
                    errors.append('Source content crosses splits (including renamed media): ' + sha)
                sources[sha] = split
    return dict(format='camix_v2_collection_quality_v1', passed=not errors, errors=sorted(set(errors)),
                events=len(events), splits=splits, reviewed_events=reviewed, synthetic_events=synthetic,
                professional_evidence=False, production_approved=False,
                warning='Protocol/leakage checks do not certify director quality, legal rights or semantic deduplication')


class IntervalCursor:
    """Monotone label-side interval lookup; never part of policy observations."""
    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda r: (r['start_s'], r['id']))
        self.position = 0
        self.last_time = -1.
        self.active = {}

    def at(self, time_s):
        if time_s < self.last_time:
            raise ValueError('Reset interval cursor on event seek')
        self.last_time = time_s
        self.active = {key: row for key, row in self.active.items() if row['end_s'] > time_s}
        while self.position < len(self.rows) and self.rows[self.position]['start_s'] <= time_s:
            row = self.rows[self.position]
            if row['end_s'] > time_s:
                self.active[row['id']] = row
            self.position += 1
        return list(self.active.values())
