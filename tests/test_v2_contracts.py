from copy import deepcopy
import json
import pytest

from camix_director.annotations_v2 import EventStore, bind_media, import_v1_media
from camix_director.contracts_v2 import (data_path, fingerprint, load_json, media_identity,
                                       select_available, validate_manifest)
from camix_director.prepare_v2 import (IntervalCursor, collection_quality, index_event,
                                     prepare_index, supervision_at, validate_index)
from v2_helpers import add_comparison, make_event, reviewed


@pytest.fixture
def event(tmp_path):
    return bind_media(tmp_path, make_event(tmp_path))


def test_contract_retains_absolute_labels_and_uncertainty(tmp_path, event):
    p = EventStore(tmp_path).create(event)
    m = add_comparison(p)
    for label in ('a', 'b', 'tie', 'uncertain'):
        m['comparisons'][0]['label'] = label
        assert validate_manifest(m)['comparisons'][0]['label'] == label
    assert m['shot_preferences'][0]['acceptable_cameras'] == ['b']


@pytest.mark.parametrize('field,value', [('source', 'unknown'), ('split', 'dev'), ('duration_s', float('nan')),
                                         ('duration_s', True), ('duration_s', 0), ('extra', 1)])
def test_contract_rejects_ambiguous_fields(event, field, value):
    event[field] = value
    with pytest.raises(ValueError):
        validate_manifest(event)


@pytest.mark.parametrize('mutation', ['truthy_license', 'future_live', 'bad_mode', 'duplicate_camera', 'keep_camera',
                                     'negative_delay', 'duplicate_annotation', 'unknown_accepted', 'relative_keep',
                                     'bad_window', 'reverse_interval', 'unordered_pgm', 'unknown_pgm'])
def test_contract_rejects_invalid_timing_and_labels(event, mutation):
    if mutation == 'truthy_license': event['license']['usage_confirmed'] = 'true'
    if mutation == 'future_live': event['clock']['delay_s'] = 1
    if mutation == 'bad_mode': event['clock']['mode'] = 'offline_future'
    if mutation == 'duplicate_camera': event['cameras'][1]['id'] = 'a'
    if mutation == 'keep_camera': event['cameras'][0]['id'] = 'keep'
    if mutation == 'negative_delay': event['cameras'][0]['video_delay_s'] = -1
    if mutation == 'duplicate_annotation': event['shot_preferences'] *= 2
    if mutation == 'unknown_accepted': event['shot_preferences'][0]['acceptable_cameras'] = ['x']
    if mutation == 'relative_keep': event['shot_preferences'][0]['acceptable_cameras'] = ['keep']
    if mutation == 'bad_window': event['shot_preferences'][0]['cut_window_s'] = [4, 3]
    if mutation == 'reverse_interval': event['events'][0]['start_s'] = 5
    if mutation == 'unordered_pgm': event['program_events'].reverse()
    if mutation == 'unknown_pgm': event['program_events'][0]['camera_id'] = 'x'
    with pytest.raises(ValueError): validate_manifest(event)


def test_strict_json_rejects_duplicates_and_nonfinite(tmp_path):
    path = tmp_path / 'a.json'
    for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
        path.write_text(text)
        with pytest.raises(ValueError): load_json(path)


def test_source_paths_reject_remote_traversal_and_symlinks(tmp_path, event):
    outside = tmp_path / 'private.mp4'
    outside.write_bytes(b'not media')
    link = tmp_path / 'data/escape.mp4'
    link.symlink_to(outside)
    for path in ('../private.mp4', 'https://example.invalid/video.mp4', str(link)):
        with pytest.raises(ValueError): data_path(tmp_path, path)


def test_monotone_causal_observation_selection_ignores_future_records():
    clock = dict(mode='strict_live', delay_s=0)
    past = dict(window_start_s=0, window_end_s=1, available_at_s=1.2, value='past')
    future = dict(window_start_s=2, window_end_s=3, available_at_s=3.2, value='future')
    delayed = dict(window_start_s=.5, window_end_s=1.5, available_at_s=8., value='not_published')
    assert select_available([past, future, delayed], 2, clock, 2) == past
    future['value'] = 'arbitrarily changed future'
    assert select_available([future, delayed, past], 2, clock, 2) == past
    assert select_available([past], 4, clock, 1) is None
    assert select_available([past], 1.1, clock, 2) is None
    assert select_available([past, future], 2, dict(mode='fixed_delay', delay_s=1.5), 2) == future


def test_impossible_availability_is_rejected():
    with pytest.raises(ValueError, match='Impossible'):
        select_available([dict(window_start_s=0, window_end_s=2, available_at_s=1, value=0)], 2, dict(mode='strict_live', delay_s=0), 2)


def test_review_is_revision_bound_and_never_promotes_synthetic(tmp_path, event):
    store = EventStore(tmp_path)
    p = store.create(event)
    with pytest.raises(ValueError, match='boolean'): store.review(p['id'], 1, 'name', 'true', True)
    p = store.review(p['id'], 1, 'self declared', True, True)
    assert p['review']['content_sha256'] == p['content_sha256']
    assert p['manifest']['source'] == 'synthetic' and not p['production_approved']
    m = deepcopy(p['manifest']);m['style']['notes'] = 'changed'
    saved = store.save(p['id'], p['revision'], m)
    assert saved['review'] is None and saved['revision'] == p['revision'] + 1
    with pytest.raises(ValueError, match='conflict'): store.save(p['id'], p['revision'], m)
    assert store.get(p['id'], p['revision'])['review'] is not None
    m['source'] = 'real'
    with pytest.raises(ValueError, match='provenance'): store.save(saved['id'], saved['revision'], m)


def test_review_requires_authorization_and_actual_labels(tmp_path, event):
    store = EventStore(tmp_path)
    event['license']['usage_confirmed'] = False
    p = store.create(event)
    with pytest.raises(ValueError, match='usage'): store.review(p['id'], 1, 'reviewer', True, True)
    event['license']['usage_confirmed'] = True
    event['shot_preferences'] = [];event['events'] = []
    p = store.create(event)
    with pytest.raises(ValueError, match='No annotations'): store.review(p['id'], 1, 'reviewer', True, True)


def test_source_mutation_and_saved_content_tampering_invalidate_review(tmp_path, event):
    store = EventStore(tmp_path);p = store.create(event)
    source = tmp_path / event['cameras'][0]['video']
    with source.open('ab') as f: f.write(b'changed')
    with pytest.raises(ValueError, match='changed'): store.review(p['id'], 1, 'reviewer', True, True)
    path = store.path(p['id'], 1);value = json.loads(path.read_text());value['manifest']['style']['name'] = 'tampered';path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='identity'): store.get(p['id'], 1)


def test_ab_comparison_is_bound_to_exact_media_clock_and_timeline(tmp_path, event):
    store = EventStore(tmp_path);p = store.create(event)
    m = add_comparison(p)
    assert validate_manifest(m)['comparisons']
    m['cameras'][0]['video_offset_s'] += .1
    with pytest.raises(ValueError, match='A/B source identity'): validate_manifest(m)
    m = add_comparison(p);m['comparisons'][0]['b'] = deepcopy(m['comparisons'][0]['a'])
    with pytest.raises(ValueError, match='must differ'): validate_manifest(m)
    m = add_comparison(p);m['comparisons'][0]['b']['cuts'][0]['time_s'] = 5.5
    with pytest.raises(ValueError, match='inside'): validate_manifest(m)


def test_long_event_is_indexed_without_decoding_and_chunks_preserve_context(tmp_path, event):
    event['duration_s'] = 3600
    p = EventStore(tmp_path).create(event)
    idx = prepare_index(tmp_path, p, decision_hz=2, chunk_steps=64, history_s=8)
    assert idx['ticks'] == 7200 and len(idx['chunks']) == 113
    assert idx['chunks'][0]['reset_event'] and not idx['chunks'][1]['reset_event']
    assert idx['chunks'][1]['context_start'] == 48
    assert idx['chunks'][-1]['tick_end'] == 7200
    assert not idx['v2_model_available'] and not idx['execute_allowed']
    idx['chunks'][0]['tick_end'] += 1
    with pytest.raises(ValueError, match='changed'): validate_index(idx)


def test_index_resource_guard_before_allocating_millions_of_chunks(tmp_path, event):
    event['duration_s'] = 86400
    p = EventStore(tmp_path).create(event)
    with pytest.raises(ValueError, match='10000 chunks'): index_event(p, decision_hz=30, chunk_steps=1)


def test_preview_not_training_and_synthetic_opt_in_required(tmp_path, event):
    store = EventStore(tmp_path);p = store.create(event)
    with pytest.raises(ValueError, match='review'): index_event(p, purpose='training', allow_synthetic=True)
    p = store.review(p['id'], 1, 'tester', True, True)
    with pytest.raises(ValueError, match='Synthetic'): index_event(p, purpose='training')
    assert index_event(p, purpose='training', allow_synthetic=True)['config']['purpose'] == 'training'


def test_relative_context_cannot_be_reused_after_simulated_pgm_changes(tmp_path, event):
    event['shot_preferences'][0]['context_program'] = 'a'
    p = EventStore(tmp_path).create(event)
    assert supervision_at(p, 2, 'a')
    assert not supervision_at(p, 2, 'b')


def test_collection_detects_renamed_sources_and_duplicate_revisions(tmp_path, event):
    store = EventStore(tmp_path);a = store.create(event)
    other = deepcopy(event);other.update(original_event_id='renamed-event', split='test')
    b = store.create(other)
    report = collection_quality(tmp_path, [a, b])
    assert not report['passed'] and any('crosses splits' in e for e in report['errors'])
    assert not collection_quality(tmp_path, [a, a])['passed']
    assert collection_quality(tmp_path, [a])['passed']


def test_label_interval_cursor_crosses_chunks_without_rescanning_future(event):
    cursor = IntervalCursor(event['shot_preferences'])
    assert not cursor.at(0)
    assert cursor.at(2)
    assert not cursor.at(5)
    with pytest.raises(ValueError, match='seek'): cursor.at(1)


def test_explicit_v1_import_does_not_copy_legacy_labels_or_human_review(tmp_path, event):
    base = tmp_path / 'data/legacy';base.mkdir()
    legacy = dict(session_id='old-original', source='synthetic', split='train', duration_seconds=6, start_seconds=2,
                  initial_program='a', human_reviewed=True, alignment_verified=True, annotations='ignored.jsonl',
                  cameras=[dict(id=c['id'], video='../raw/'+c['id']+'.mp4', audio='../raw/'+c['id']+'.wav') for c in event['cameras']])
    path = base / 'session.json';path.write_text(json.dumps(legacy))
    imported = import_v1_media(tmp_path, str(path))
    assert imported['shot_preferences'] == [] and imported['events'] == [] and not imported['license']['usage_confirmed']
    assert imported['cameras'][0]['video_offset_s'] == 2
    assert EventStore(tmp_path).create(imported)['review'] is None


@pytest.mark.parametrize('mutation', ['media_identity', 'production_approved', 'execute_allowed', 'review_identity'])
def test_document_wrapper_cannot_silently_change_identity_or_approval(tmp_path, event, mutation):
    from camix_director.annotations_v2 import validate_document
    p = reviewed(tmp_path, event)
    if mutation == 'media_identity': p['media_identity'] = '0' * 64
    elif mutation == 'review_identity': p['review']['identity_type'] = 'certified_professional_director'
    else: p[mutation] = True
    with pytest.raises(ValueError): validate_document(p)
