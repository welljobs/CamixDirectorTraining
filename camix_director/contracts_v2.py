"""V2 event/annotation contracts. Separate from schema-1 tensors and checkpoints.

All times are event-relative seconds. Labels are never observation features.
The strict JSON schema intentionally rejects unknown fields and truthy flags.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

EVENT_FORMAT = 'camix_event_v2'
REVIEW_FORMAT = 'camix_reviewed_event_v2'
INDEX_FORMAT = 'camix_event_index_v2'
INTENTS = ('hold', 'establish', 'subject', 'action', 'reaction', 'detail', 'unknown')
REASONS = ('technical', 'aesthetics', 'rhythm', 'continuity', 'coverage', 'narrative', 'other')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def load_json(path, limit=8 * 1024 * 1024):
    path = Path(path)
    if path.stat().st_size > limit:
        raise ValueError('V2 JSON exceeds bounded input size')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Non-finite JSON: ' + value)
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=invalid)


def object_fields(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError('Invalid object or unknown fields: ' + str(set(value) - set(allowed) if isinstance(value, dict) else type(value).__name__))
    if set(required) - set(value):
        raise ValueError('Missing fields: ' + str(set(required) - set(value)))


def text(value, name, limit=512):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError('Invalid ' + name)
    return value


def number(value, name, minimum=0, maximum=86400):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError('Invalid finite ' + name)
    return float(value)


def integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError('Invalid integer ' + name)
    return value


def boolean(value, name):
    if type(value) is not bool:
        raise ValueError(name + ' must be a JSON boolean')
    return value


def sequence(value, name, maximum=10000):
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError('Invalid bounded list: ' + name)
    return value


def strings(value, name, allowed=None, nonempty=False):
    values = sequence(value, name, 64)
    if (nonempty and not values) or any(not isinstance(v, str) or not v or len(v) > 120 for v in values) or len(set(values)) != len(values):
        raise ValueError('Invalid unique ' + name)
    if allowed is not None and set(values) - set(allowed):
        raise ValueError('Unknown ' + name)
    return values


def sha256(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('Expected SHA256 identity')
    return value


def data_path(root, value, exists=True):
    root = Path(root).resolve()
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError('Missing data path')
    path = (root / value).resolve()
    if not path.is_relative_to(root / 'data'):
        raise ValueError('V2 source must remain inside workspace data (no symlink escape)')
    if exists and not path.is_file():
        raise ValueError('Missing data file: ' + value)
    return path


def media_identity(manifest):
    return fingerprint({k: manifest[k] for k in ('original_event_id', 'duration_s', 'cameras', 'clock')})


def program_at(initial, cuts, time_s, include_at=True):
    current = initial
    for cut in cuts:
        if cut['time_s'] > time_s or (not include_at and cut['time_s'] == time_s):
            break
        current = cut['camera_id']
    return current


def validate_timeline(cuts, cameras, duration):
    last = -1.
    for cut in sequence(cuts, 'program timeline', 50000):
        object_fields(cut, ('time_s', 'camera_id'), ('time_s', 'camera_id'))
        t = number(cut['time_s'], 'cut time', 0, duration)
        if t >= duration or t <= last or cut['camera_id'] not in cameras:
            raise ValueError('Timeline must be strictly ordered with known cameras inside event')
        cut['time_s'] = t
        last = t


def interval(row, duration):
    start = number(row['start_s'], 'interval start', 0, duration)
    end = number(row['end_s'], 'interval end', 0, duration)
    if start >= end:
        raise ValueError('Interval must have start < end')
    row.update(start_s=start, end_s=end)


def validate_manifest(value):
    m = deepcopy(value)
    fields = ('format', 'original_event_id', 'split', 'source', 'duration_s', 'initial_program', 'cameras', 'clock', 'license', 'scenario_tags', 'style', 'program_events', 'shot_preferences', 'events', 'comparisons')
    object_fields(m, fields, fields)
    if m['format'] != EVENT_FORMAT:
        raise ValueError('Not a V2 event; schema-1 is not silently converted')
    text(m['original_event_id'], 'original event ID', 160)
    if m['split'] not in ('train', 'validation', 'test') or m['source'] not in ('real', 'synthetic'):
        raise ValueError('Explicit split/source required')
    duration = number(m['duration_s'], 'event duration', .1, 86400)
    m['duration_s'] = duration
    object_fields(m['clock'], ('mode', 'delay_s'), ('mode', 'delay_s'))
    delay = number(m['clock']['delay_s'], 'fixed delay', 0, 30)
    if m['clock']['mode'] not in ('strict_live', 'fixed_delay') or (m['clock']['mode'] == 'strict_live' and delay != 0):
        raise ValueError('Strict live cannot use future lookahead or a delay')
    m['clock']['delay_s'] = delay
    object_fields(m['license'], ('name', 'usage_confirmed'), ('name', 'usage_confirmed'))
    text(m['license']['name'], 'license')
    boolean(m['license']['usage_confirmed'], 'usage_confirmed')
    strings(m['scenario_tags'], 'scenario tags')
    object_fields(m['style'], ('name', 'notes'), ('name', 'notes'))
    text(m['style']['name'], 'style name', 120)
    if not isinstance(m['style']['notes'], str) or len(m['style']['notes']) > 4000:
        raise ValueError('Invalid style notes')
    cameras = sequence(m['cameras'], 'cameras', 16)
    if not cameras:
        raise ValueError('At least one camera required')
    ids = []
    for c in cameras:
        required = ('id', 'video', 'audio', 'video_offset_s', 'audio_offset_s', 'video_delay_s', 'audio_delay_s', 'video_sha256', 'audio_sha256')
        object_fields(c, required, required)
        identifier = text(c['id'], 'camera ID', 100)
        if identifier == 'keep' or identifier in ids:
            raise ValueError('Camera IDs must be unique and not the relative action keep')
        ids.append(identifier)
        text(c['video'], 'video path', 2048)
        sha256(c['video_sha256'])
        if c['audio'] is not None:
            text(c['audio'], 'audio path', 2048)
            sha256(c['audio_sha256'])
        elif c['audio_sha256'] is not None:
            raise ValueError('Missing audio cannot have a hash')
        for name in ('video_offset_s', 'audio_offset_s'):
            c[name] = number(c[name], name, -86400, 86400)
        for name in ('video_delay_s', 'audio_delay_s'):
            c[name] = number(c[name], name, 0, 30)
    if m['initial_program'] not in ids:
        raise ValueError('Initial program must be a known camera')
    validate_timeline(m['program_events'], ids, duration)
    for field in ('shot_preferences', 'events', 'comparisons'):
        seen = set()
        for row in sequence(m[field], field):
            text(row.get('id'), field + ' ID', 120)
            if row['id'] in seen:
                raise ValueError('Duplicate annotation ID')
            seen.add(row['id'])
            if field == 'shot_preferences':
                allowed = ('id', 'start_s', 'end_s', 'acceptable_cameras', 'cut_window_s', 'context_program', 'intent', 'reason_tags')
                object_fields(row, allowed, allowed)
                interval(row, duration)
                strings(row['acceptable_cameras'], 'absolute acceptable cameras', ids, nonempty=True)
                if row['context_program'] is not None and row['context_program'] not in ids:
                    raise ValueError('Unknown reference program')
                if row['intent'] not in INTENTS:
                    raise ValueError('Unknown intent')
                strings(row['reason_tags'], 'reason tags', REASONS)
                if row['cut_window_s'] is not None:
                    window = row['cut_window_s']
                    if not isinstance(window, list) or len(window) != 2:
                        raise ValueError('Cut window must be [start, end] or null')
                    lo, hi = [number(t, 'cut window', row['start_s'], row['end_s']) for t in window]
                    if lo > hi:
                        raise ValueError('Reversed cut window')
                    row['cut_window_s'] = [lo, hi]
            elif field == 'events':
                allowed = ('id', 'start_s', 'end_s', 'label', 'visible_cameras', 'importance')
                object_fields(row, allowed, allowed)
                interval(row, duration)
                text(row['label'], 'event label', 512)
                strings(row['visible_cameras'], 'visible cameras', ids)
                row['importance'] = number(row['importance'], 'importance', 0, 1)
            else:
                allowed = ('id', 'start_s', 'end_s', 'source_identity', 'a', 'b', 'label', 'reviewer', 'reason_tags', 'notes')
                object_fields(row, allowed, allowed)
                interval(row, duration)
                if row['source_identity'] != media_identity(m):
                    raise ValueError('A/B source identity changed; rebind and review actual variants')
                for key in ('a', 'b'):
                    v = row[key]
                    object_fields(v, ('initial_program', 'cuts'), ('initial_program', 'cuts'))
                    if v['initial_program'] not in ids:
                        raise ValueError('Unknown A/B initial program')
                    validate_timeline(v['cuts'], ids, duration)
                    if any(not row['start_s'] <= cut['time_s'] < row['end_s'] for cut in v['cuts']):
                        raise ValueError('A/B cuts must be inside compared interval')
                if row['a'] == row['b']:
                    raise ValueError('A/B variants must differ')
                if row['label'] not in ('a', 'b', 'tie', 'uncertain'):
                    raise ValueError('A/B label must allow tie/uncertain')
                text(row['reviewer'], 'self-declared comparison reviewer', 160)
                strings(row['reason_tags'], 'comparison reasons', REASONS)
                if not isinstance(row['notes'], str) or len(row['notes']) > 4000:
                    raise ValueError('Invalid comparison notes')
    return m


def select_available(records, event_time_s, clock, max_age_s):
    """Select only published past observations; labels cannot enter this API.

    Fixed-delay mode may see beyond target event time, but never beyond the
    actual decision time. That mode must not be reported as strict-live.
    """
    event_time_s = number(event_time_s, 'decision event time')
    max_age_s = number(max_age_s, 'max age', 0, 60)
    object_fields(clock, ('mode', 'delay_s'), ('mode', 'delay_s'))
    delay = number(clock['delay_s'], 'delay', 0, 30)
    if clock['mode'] not in ('strict_live', 'fixed_delay') or (clock['mode'] == 'strict_live' and delay):
        raise ValueError('Invalid causal clock')
    cutoff = event_time_s + delay
    eligible = []
    for row in records:
        object_fields(row, ('window_start_s', 'window_end_s', 'available_at_s', 'value'), ('window_start_s', 'window_end_s', 'available_at_s', 'value'))
        start = number(row['window_start_s'], 'window start', -86400)
        end = number(row['window_end_s'], 'window end', -86400)
        available = number(row['available_at_s'], 'availability', -86400)
        if start > end or available < end:
            raise ValueError('Impossible observation timing')
        if available <= cutoff and end <= cutoff and cutoff - end <= max_age_s:
            eligible.append(row)
    return max(eligible, key=lambda r: (r['window_end_s'], r['available_at_s'])) if eligible else None
