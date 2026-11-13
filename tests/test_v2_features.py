from copy import deepcopy
import json
import numpy as np
import pytest
import torch

from camix_director.annotations_v2 import EventStore
from camix_director.contracts_v2 import fingerprint
from camix_director.data_v2 import EventFeatureSequences
from camix_director.features_v2 import (FrozenEncoder, cache_header, extract, iter_features)
from camix_director.prepare_v2 import index_event
from camix_director.replay_v2 import replay
from camix_director.runtime import atomic_json
from camix_director.storage import digest_file
from v2_helpers import encoder_weights, make_event, reviewed


@pytest.fixture
def prepared(tmp_path):
    p = reviewed(tmp_path)
    index = index_event(p, decision_hz=2, chunk_steps=3, purpose='training', allow_synthetic=True)
    weights, sha = encoder_weights(tmp_path)
    cache = extract(tmp_path, index, 'v1_visual', str(weights), sha)
    return index, cache, weights, sha


def rewrite_chunk(root, header, transform, ordinal=0):
    directory = root / '.cache/v2/features' / header['cache_id']
    path = directory / header['chunks'][ordinal]['path']
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    transform(arrays)
    np.savez_compressed(path, **arrays)
    header = deepcopy(header)
    header['chunks'][ordinal]['sha256'] = digest_file(path)
    header.pop('content_sha256')
    header['content_sha256'] = fingerprint(header)
    atomic_json(directory / 'features.json', header)
    return header


def test_local_frozen_extraction_is_bounded_causal_and_reuses_verified_cache(tmp_path, prepared, monkeypatch):
    index, cache, weights, sha = prepared
    rows = list(iter_features(tmp_path, cache['cache_id']))
    assert len(rows) == 4 and rows[0][1]['visual'].shape == (3, 2, 16)
    assert rows[0][1]['audio_mask'][0].sum() == 0
    for row, arrays in rows:
        assert len(arrays['times']) <= 3
        assert np.isfinite(arrays['visual']).all()
        valid = arrays['video_mask'] > .5
        assert np.all(arrays['video_available_s'][valid] <= np.broadcast_to(arrays['times'][:, None], valid.shape)[valid] + 1e-6)
    import camix_director.features_v2 as module
    def forbidden(*a, **k): raise AssertionError('Cache reuse must not reconstruct/download an encoder')
    monkeypatch.setattr(module, 'FrozenEncoder', forbidden)
    assert extract(tmp_path, index, 'v1_visual', str(weights), sha)['cache_id'] == cache['cache_id']


def test_encoder_weights_are_frozen_and_missing_sha_never_downloads(tmp_path, prepared, monkeypatch):
    _, _, weights, sha = prepared
    def forbidden(*a, **k): raise AssertionError('No network/download permitted')
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', forbidden)
    encoder = FrozenEncoder(tmp_path, 'v1_visual', str(weights), sha)
    assert not encoder.module.training and all(not p.requires_grad for p in encoder.module.parameters())
    with pytest.raises(ValueError, match='identity'): FrozenEncoder(tmp_path, 'v1_visual', str(weights), '0' * 64)
    with pytest.raises(ValueError): FrozenEncoder(tmp_path, 'unknown', str(weights), sha)


def test_changed_encoder_weights_invalidate_identity(tmp_path, prepared):
    index, old_cache, weights, sha = prepared
    checkpoint = torch.load(weights, weights_only=True)
    name = next(k for k in checkpoint['model_state'] if k.startswith('video.') and k.endswith('weight'))
    checkpoint['model_state'][name] += .01
    torch.save(checkpoint, weights)
    with pytest.raises(ValueError, match='identity'): extract(tmp_path, index, 'v1_visual', str(weights), sha)
    new_cache = extract(tmp_path, index, 'v1_visual', str(weights), digest_file(weights))
    assert new_cache['cache_id'] != old_cache['cache_id']


def test_changed_labels_do_not_change_observations_or_become_input_features(tmp_path, prepared):
    index, cache, weights, sha = prepared
    store = EventStore(tmp_path)
    p = index['document'];m = deepcopy(p['manifest'])
    m['events'][0]['label'] = 'arbitrary future narrative label'
    m['shot_preferences'][0]['acceptable_cameras'] = ['a']
    p = store.save(p['id'], p['revision'], m)
    p = store.review(p['id'], p['revision'], 'synthetic tester', True, True)
    changed = index_event(p, decision_hz=2, chunk_steps=3, purpose='training', allow_synthetic=True)
    new_cache = extract(tmp_path, changed, 'v1_visual', str(weights), sha)
    assert new_cache['cache_id'] != cache['cache_id']
    for (_, before), (_, after) in zip(iter_features(tmp_path, cache['cache_id']), iter_features(tmp_path, new_cache['cache_id'])):
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        assert not {'events', 'comparisons', 'shot_preferences'} & set(before)


def test_training_sequences_carry_pgm_across_chunks_and_mask_context(tmp_path):
    m = make_event(tmp_path);m['shot_preferences'][0]['context_program'] = 'a'
    p = reviewed(tmp_path, m)
    index = index_event(p, chunk_steps=3, decision_hz=2, purpose='training', allow_synthetic=True)
    weights, sha = encoder_weights(tmp_path)
    cache = extract(tmp_path, index, 'v1_visual', str(weights), sha)
    samples = list(EventFeatureSequences(tmp_path, index, cache['cache_id']))
    assert len(samples) == 4 and samples[0]['state']['reset_event']
    assert not samples[1]['state']['reset_event']
    # A teacher cut exactly at t=2 is the target action, not pre-decision input.
    assert samples[1]['observations']['current_program'].tolist() == [[1., 0.], [1., 0.], [0., 1.]]
    assert samples[1]['observations']['held_seconds'][:, 0].tolist() == [1.5, 2., .5]
    assert samples[1]['targets']['annotated'].tolist() == [True, True, False]
    assert samples[1]['context_skipped'] == 1
    assert 'acceptable_cameras' not in samples[1]['observations']


def test_preview_features_cannot_be_promoted_to_training_samples(tmp_path, prepared):
    index, cache, _, _ = prepared
    preview = index_event(index['document'], chunk_steps=3)
    with pytest.raises(ValueError, match='Preview'): EventFeatureSequences(tmp_path, preview, cache['cache_id'])
    other = deepcopy(index);other['config']['chunk_steps'] = 4
    other = index_event(index['document'], chunk_steps=4, purpose='training', allow_synthetic=True)
    with pytest.raises(ValueError, match='another'): EventFeatureSequences(tmp_path, other, cache['cache_id'])


def test_source_changed_after_review_cannot_extract_or_build_samples(tmp_path, prepared):
    index, cache, weights, sha = prepared
    path = tmp_path / index['document']['manifest']['cameras'][0]['video']
    with path.open('ab') as handle: handle.write(b'changed source')
    with pytest.raises(ValueError, match='changed'): extract(tmp_path, index, 'v1_visual', str(weights), sha)
    with pytest.raises(ValueError, match='changed'): EventFeatureSequences(tmp_path, index, cache['cache_id'])


def test_tampered_feature_bytes_or_manifest_are_rejected(tmp_path, prepared):
    _, cache, _, _ = prepared
    directory = tmp_path / '.cache/v2/features' / cache['cache_id']
    path = directory / cache['chunks'][0]['path']
    with path.open('ab') as handle: handle.write(b'changed')
    with pytest.raises(ValueError, match='changed'): list(iter_features(tmp_path, cache['cache_id']))
    header = json.loads((directory / 'features.json').read_text());header['feature_dim'] = 2
    atomic_json(directory / 'features.json', header)
    with pytest.raises(ValueError, match='manifest'): cache_header(tmp_path, cache['cache_id'])


def test_future_feature_timestamps_rejected_even_when_chunk_hash_is_updated(tmp_path, prepared):
    _, cache, _, _ = prepared
    def corrupt(arrays): arrays['video_available_s'][0, 0] = 100
    rewrite_chunk(tmp_path, cache, corrupt)
    with pytest.raises(ValueError, match='Future'): list(iter_features(tmp_path, cache['cache_id']))


def test_delayed_camera_does_not_get_future_frame_at_initial_tick(tmp_path):
    m = make_event(tmp_path)
    for c in m['cameras']: c['video_delay_s'] = 1.
    p = reviewed(tmp_path, m)
    index = index_event(p, chunk_steps=3)
    weights, sha = encoder_weights(tmp_path)
    cache = extract(tmp_path, index, 'v1_visual', str(weights), sha)
    first = next(iter_features(tmp_path, cache['cache_id']))[1]
    assert not first['video_mask'][:2].any()
    assert not first['visual'][:2].any()
    assert first['video_mask'][2].all()


def test_replay_keeps_state_across_chunks_and_logs_raw_context_before_action(tmp_path, prepared):
    index, cache, _, _ = prepared
    def outage(arrays):
        arrays['video_mask'][2, 0] = 0
        arrays['visual'][2, 0] = 0
    rewrite_chunk(tmp_path, cache, outage)
    result = replay(tmp_path, index, cache['cache_id'], tmp_path / 'runs/replay', 'hold_valid', 2.)
    assert result['fallback_count'] == 1 and result['cut_count'] == 1
    assert not result['learned_policy'] and not result['visual_features_used_for_decision']
    assert result['qualification'] == 'synthetic_smoke_only' and not result['execute_allowed']
    trace = [json.loads(line) for line in (tmp_path / 'runs/replay/value-trace.jsonl').read_text().splitlines()]
    assert len(trace) == index['ticks']
    assert trace[2]['source_current_program'] == 'a' and trace[2]['selected_camera'] == 'b'
    assert trace[3]['source_current_program'] == 'b' and trace[3]['selected_camera'] == 'b'
    with pytest.raises(ValueError, match='refusing'): replay(tmp_path, index, cache['cache_id'], tmp_path / 'runs/replay')


def test_unreviewed_replay_cannot_report_director_correctness(tmp_path, prepared):
    index, _, weights, sha = prepared
    store = EventStore(tmp_path);p = index['document'];p = store.save(p['id'], p['revision'], p['manifest'])
    idx = index_event(p, chunk_steps=3)
    cache = extract(tmp_path, idx, 'v1_visual', str(weights), sha)
    report = replay(tmp_path, idx, cache['cache_id'], tmp_path / 'runs/preview', 'audio_rms')
    assert report['annotated_ticks'] == 0 and report['acceptable_camera_rate'] is None and report['event_coverage_rate'] is None


def test_cancelled_extraction_does_not_commit_cache(tmp_path):
    p = reviewed(tmp_path);index = index_event(p)
    weights, sha = encoder_weights(tmp_path)
    stop = tmp_path / 'stop';stop.write_text('stop')
    with pytest.raises(InterruptedError): extract(tmp_path, index, 'v1_visual', str(weights), sha, stop)
    assert not list((tmp_path / '.cache/v2/features').glob('*/features.json'))


def test_frozen_encoder_future_frame_changes_cannot_affect_past_features(tmp_path, prepared):
    _, _, weights, sha = prepared
    encoder = FrozenEncoder(tmp_path, 'v1_visual', str(weights), sha)
    frames = np.zeros((8, 16, 16, 3), np.uint8)
    before = encoder.encode(frames)
    frames[4:] = 255
    after = encoder.encode(frames)
    np.testing.assert_array_equal(before[:4], after[:4])


def test_missing_modality_cannot_hide_nonzero_features(tmp_path, prepared):
    _, cache, _, _ = prepared
    def corrupt(arrays): arrays['video_mask'][0, 0] = 0
    rewrite_chunk(tmp_path, cache, corrupt)
    with pytest.raises(ValueError, match='zero features'): list(iter_features(tmp_path, cache['cache_id']))


def test_sample_interface_is_a_real_pytorch_iterable_and_rejects_worker_duplication(tmp_path, prepared, monkeypatch):
    from torch.utils.data import DataLoader
    import camix_director.data_v2 as module
    index, cache, _, _ = prepared
    dataset = EventFeatureSequences(tmp_path, index, cache['cache_id'])
    batches = list(DataLoader(dataset, batch_size=None, num_workers=0))
    assert len(batches) == 4 and torch.is_tensor(batches[0]['observations']['visual'])
    monkeypatch.setattr(module, 'get_worker_info', lambda: object())
    with pytest.raises(ValueError, match='num_workers=0'): list(dataset)


def test_optional_mobilenet_backend_strictly_loads_only_local_weights(tmp_path, monkeypatch):
    torchvision = pytest.importorskip('torchvision')
    model = torchvision.models.mobilenet_v3_small(weights=None)
    path = tmp_path / 'data/encoders/synthetic-mobilenet.pth';path.parent.mkdir(parents=True)
    torch.save(model.state_dict(), path)
    def forbidden(*args, **kwargs): raise AssertionError('No automatic pretrained download')
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', forbidden)
    encoder = FrozenEncoder(tmp_path, 'mobilenet_v3_small', str(path), digest_file(path))
    features = encoder.encode(np.zeros((1, 224, 224, 3), np.uint8))
    assert features.shape == (1, 576) and np.isfinite(features).all()
    assert all(not p.requires_grad for p in encoder.module.parameters())
