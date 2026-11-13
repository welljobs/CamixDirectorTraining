"""Offline, bounded, causal visual extraction with local frozen weights only.

No download, remote code, TorchScript execution, or random fallback is used.
Pretraining provenance cannot be inferred from arbitrary local weight bytes. V1 visual weights are a legacy baseline, NOT a foundation upgrade.
"""
import importlib.metadata
import math
from pathlib import Path
import uuid
import zipfile

import numpy as np
import torch

from .annotations_v2 import bind_media
from .contracts_v2 import fingerprint, load_json, sha256
from .media import audio_samples, video_samples
from .prepare_v2 import validate_index
from .runtime import atomic_json, file_lock
from .storage import digest_file

CACHE_FORMAT = 'camix_frozen_features_v2'
ARRAY_FIELDS = ('times', 'visual', 'video_mask', 'video_end_s', 'video_available_s',
                'audio_rms', 'audio_mask', 'audio_end_s', 'audio_available_s')


def local_weights(root, value, expected_sha):
    root = Path(root).resolve()
    path = (root / value).resolve()
    sha256(expected_sha)
    if (not any(path.is_relative_to(root / folder) for folder in ('runs', 'data'))
            or not path.is_file() or path.suffix not in ('.pt', '.pth')):
        raise ValueError('Encoder weights must be a local .pt/.pth under runs or data')
    if path.stat().st_size > 2 * 1024**3 or digest_file(path) != expected_sha:
        raise ValueError('Local encoder weight identity mismatch or excessive size')
    return path


class FrozenEncoder:
    def __init__(self, root, kind, weights, weights_sha256):
        path = local_weights(root, weights, weights_sha256)
        self.kind = kind
        if kind == 'v1_visual':
            from .train import load_checkpoint
            model, checkpoint = load_checkpoint(path)
            self.module = model.video.eval()
            self.image_size = model.config.image_size
            self.feature_dim = model.config.hidden_dim
            self.description = 'legacy weak-supervision V1 visual baseline; no proven aesthetic semantics'
        elif kind == 'mobilenet_v3_small':
            from torchvision.models import mobilenet_v3_small
            model = mobilenet_v3_small(weights=None)
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            self.module = torch.nn.Sequential(model.features, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()).eval()
            self.image_size = 224
            self.feature_dim = 576
            self.description = 'local MobileNet features; source/pretraining and suitability require independent review'
        else:
            raise ValueError('Choose v1_visual or local mobilenet_v3_small; automatic downloads disabled')
        if not 16 <= self.image_size <= 512 or not 1 <= self.feature_dim <= 4096:
            raise ValueError('Encoder shape exceeds bounded extraction limits')
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, frames):
        result = []
        for begin in range(0, len(frames), 16):
            x = torch.from_numpy(np.ascontiguousarray(frames[begin:begin+16].transpose(0, 3, 1, 2))).float() / 255
            if self.kind == 'v1_visual':
                x = x * 2 - 1
            else:
                mean = torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1)
                std = torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1)
                x = (x - mean) / std
            y = self.module(x).numpy()
            if y.shape != (len(x), self.feature_dim) or not np.isfinite(y).all():
                raise ValueError('Invalid frozen encoder output')
            result.append(y)
        return np.concatenate(result).astype(np.float32)


def _cache_directory(root, cache_id):
    sha256(cache_id)
    root = Path(root).resolve()
    base = root / '.cache/v2/features'
    directory = base / cache_id
    if not directory.resolve().is_relative_to(root):
        raise ValueError('Feature cache path escapes workspace')
    return directory


def cache_header(root, cache_id):
    directory = _cache_directory(root, cache_id)
    header = load_json(directory / 'features.json', limit=32 * 1024 * 1024)
    if (header.get('format') != CACHE_FORMAT or header.get('cache_id') != cache_id
            or fingerprint(header.get('identity')) != cache_id):
        raise ValueError('Feature cache identity changed')
    payload = {k: v for k, v in header.items() if k != 'content_sha256'}
    if fingerprint(payload) != header.get('content_sha256'):
        raise ValueError('Feature cache manifest changed')
    expected = math.ceil(header['ticks'] / header['chunk_steps'])
    if len(header['chunks']) != expected:
        raise ValueError('Missing feature chunk')
    for index, row in enumerate(header['chunks']):
        start = index * header['chunk_steps']
        if (row['path'] != f'{index:06}.npz' or row['tick_start'] != start
                or row['tick_end'] != min(start + header['chunk_steps'], header['ticks'])):
            raise ValueError('Feature chunk identity/order changed')
    return header


def iter_features(root, cache_id, tick_start=0, tick_end=None):
    header = cache_header(root, cache_id)
    directory = _cache_directory(root, cache_id)
    n, d = len(header['camera_ids']), header['feature_dim']
    if not isinstance(tick_start, int) or isinstance(tick_start, bool) or tick_start < 0:
        raise ValueError('Invalid feature range')
    if tick_end is None:
        tick_end = header['ticks']
    if not isinstance(tick_end, int) or isinstance(tick_end, bool) or not tick_start <= tick_end <= header['ticks']:
        raise ValueError('Invalid feature range')
    for row in header['chunks']:
        if row['tick_end'] <= tick_start or row['tick_start'] >= tick_end:
            continue
        path = directory / row['path']
        if not path.resolve().is_relative_to(directory) or digest_file(path) != row['sha256']:
            raise ValueError('Feature chunk changed or escaped cache')
        with zipfile.ZipFile(path) as archive:
            if sum(i.file_size for i in archive.infolist()) > 256 * 1024**2:
                raise ValueError('Excessive expanded feature chunk')
        with np.load(path, allow_pickle=False) as source:
            if set(source.files) != set(ARRAY_FIELDS):
                raise ValueError('Feature fields changed; labels must not be in observations')
            arrays = {key: source[key] for key in ARRAY_FIELDS}
        count = row['tick_end'] - row['tick_start']
        shapes = {key: (count, n) for key in ARRAY_FIELDS}
        shapes.update(times=(count,), visual=(count, n, d))
        if any(a.dtype.hasobject or a.shape != shapes[key] or not np.isfinite(a).all() for key, a in arrays.items()):
            raise ValueError('Invalid finite feature arrays')
        expected_times = np.arange(row['tick_start'], row['tick_end']) / header['decision_hz']
        if not np.allclose(arrays['times'], expected_times, atol=1e-9, rtol=0):
            raise ValueError('Feature times do not match event index')
        cutoff = arrays['times'][:, None] + header['clock']['delay_s']
        for kind in ('video', 'audio'):
            mask = arrays[kind + '_mask']
            if not np.isin(mask, [0, 1]).all():
                raise ValueError('Feature masks must be binary')
            valid = mask > .5
            end, available = arrays[kind + '_end_s'], arrays[kind + '_available_s']
            if np.any(valid & ((end > cutoff + 1e-6) | (available > cutoff + 1e-6) | (end > available + 1e-6))):
                raise ValueError('Future/unpublished observation in feature cache')
        if np.any(arrays['visual'][arrays['video_mask'] < .5] != 0) or np.any(arrays['audio_rms'][arrays['audio_mask'] < .5] != 0):
            raise ValueError('Missing modalities must have zero features')
        if np.any(arrays['audio_rms'] < 0) or np.any(arrays['audio_rms'] > 1.0001):
            raise ValueError('Invalid raw PCM RMS range')
        yield row, arrays


def extract(root, index, encoder_kind, weights, weights_sha256, stop_file=None):
    root = Path(root).resolve()
    validate_index(index)
    m = index['document']['manifest']
    bind_media(root, m, verify=True)
    weight_path = local_weights(root, weights, weights_sha256)
    identity = dict(index_sha256=index['index_sha256'], media_identity=index['media_identity'],
                    encoder_kind=encoder_kind, weights_sha256=weights_sha256,
                    preprocessing='letterbox_rgb_uint8_past_frame_v1; audio_rms_16k_past_250ms',
                    code={p: digest_file(Path(__file__).with_name(p)) for p in ('features_v2.py', 'contracts_v2.py', 'prepare_v2.py', 'media.py', 'model.py', 'config.py', 'train.py')},
                    environment={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'av')})
    if encoder_kind == 'mobilenet_v3_small':
        identity['environment']['torchvision'] = importlib.metadata.version('torchvision')
    cache_id = fingerprint(identity)
    directory = _cache_directory(root, cache_id)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(directory.parent / (cache_id + '.lock')):
        if (directory / 'features.json').exists():
            for _ in iter_features(root, cache_id):
                pass
            return cache_header(root, cache_id)
        encoder = FrozenEncoder(root, encoder_kind, weights, weights_sha256)
        # Account for letterboxed raw frames and one float32 inference batch.
        n, h = len(m['cameras']), encoder.image_size
        estimate = index['config']['chunk_steps'] * n * h * h * 3 + 16 * h * h * 3 * 4
        if estimate > 128 * 1024**2:
            raise ValueError('Extraction frame working set >128MiB; reduce chunk_steps')
        directory.mkdir(parents=True, exist_ok=True)
        chunks = []
        coverage = np.zeros((2, n), np.int64)
        from types import SimpleNamespace
        audio_config = SimpleNamespace(sample_rate=16000, audio_samples=4000)
        for ordinal, chunk in enumerate(index['chunks']):
            if stop_file and Path(stop_file).exists():
                raise InterruptedError('V2 extraction stopped; incomplete cache is not committed')
            times = np.arange(chunk['tick_start'], chunk['tick_end'], dtype=np.float64) / index['config']['decision_hz']
            cutoff = times + m['clock']['delay_s']
            count = len(times)
            arrays = {key: np.zeros((count, n), np.float64 if key.endswith('_s') else np.float32) for key in ARRAY_FIELDS if key not in ('times', 'visual')}
            arrays['times'] = times
            arrays['visual'] = np.zeros((count, n, encoder.feature_dim), np.float32)
            for slot, camera in enumerate(m['cameras']):
                effective = cutoff - camera['video_delay_s']
                frames, valid, ages = video_samples(root / camera['video'], camera['video_offset_s'], effective, h)
                end = effective - ages.astype(np.float64)
                valid = valid * (end >= 0) * (end < m['duration_s'])
                arrays['visual'][:, slot] = encoder.encode(frames) * valid[:, None]
                arrays['video_mask'][:, slot] = valid
                arrays['video_end_s'][:, slot] = np.where(valid, end, -1)
                arrays['video_available_s'][:, slot] = np.where(valid, end + camera['video_delay_s'], -1)
                if camera['audio']:
                    effective = cutoff - camera['audio_delay_s']
                    pcm, valid = audio_samples(root / camera['audio'], camera['audio_offset_s'], effective, audio_config)
                    valid = valid * (effective >= .25) * (effective < m['duration_s'])
                    arrays['audio_rms'][:, slot] = np.sqrt(np.mean(pcm * pcm, axis=1)) * valid
                    arrays['audio_mask'][:, slot] = valid
                    arrays['audio_end_s'][:, slot] = np.where(valid, effective, -1)
                    arrays['audio_available_s'][:, slot] = np.where(valid, cutoff, -1)
            coverage[0] += arrays['video_mask'].sum(axis=0).astype(np.int64)
            coverage[1] += arrays['audio_mask'].sum(axis=0).astype(np.int64)
            name = f'{ordinal:06}.npz'
            path = directory / name
            temporary = directory / (name + '.' + uuid.uuid4().hex + '.tmp')
            try:
                with temporary.open('wb') as handle:
                    np.savez_compressed(handle, **arrays)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            chunks.append(dict(path=name, tick_start=chunk['tick_start'], tick_end=chunk['tick_end'], sha256=digest_file(path)))
            print(f'V2 features {ordinal+1}/{len(index["chunks"])}', flush=True)
        # Never commit a cache under identities that changed during extraction.
        bind_media(root, m, verify=True)
        if digest_file(weight_path) != weights_sha256:
            raise ValueError('Encoder weights changed during extraction')
        header = dict(format=CACHE_FORMAT, cache_id=cache_id, identity=identity, camera_ids=[c['id'] for c in m['cameras']],
                      ticks=index['ticks'], chunk_steps=index['config']['chunk_steps'], decision_hz=index['config']['decision_hz'],
                      clock=m['clock'], feature_dim=encoder.feature_dim, image_size=h, description=encoder.description,
                      availability_source='declared_offsets_and_delays_not_measured_live_capture',
                      video_valid_fraction=(coverage[0] / index['ticks']).tolist(),
                      audio_valid_fraction=(coverage[1] / index['ticks']).tolist(),
                      chunks=chunks, production_approved=False, execute_allowed=False)
        header['content_sha256'] = fingerprint(header)
        atomic_json(directory / 'features.json', header)
        try:
            for _ in iter_features(root, cache_id):
                pass
        except Exception:
            (directory / 'features.json').unlink(missing_ok=True)
            raise
        return header
