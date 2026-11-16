"""Frozen feature-space corpora, event-level leakage checks, no silent label upgrades."""
from dataclasses import dataclass
from pathlib import Path

from .annotations_v2 import bind_media
from .contracts_v2 import fingerprint, load_json, object_fields
from .features_v2 import cache_header, iter_features
from .prepare_v2 import validate_index
from .runtime import atomic_json
from .v2 import artifact_path

CORPUS_FORMAT = 'camix_feature_corpus_v2'


def feature_space(header):
    identity = header['identity']
    return dict(encoder_kind=identity['encoder_kind'], weights_sha256=identity['weights_sha256'],
                preprocessing=identity['preprocessing'], code=identity['code'], environment=identity['environment'],
                feature_dim=header['feature_dim'], image_size=header['image_size'])


def feature_space_id(header):
    return fingerprint(feature_space(header))


@dataclass
class FeatureEvent:
    index: dict
    header: dict
    index_path: str

    @property
    def manifest(self):
        return self.index['document']['manifest']

    @property
    def cache_id(self):
        return self.header['cache_id']


def load_corpus(root, value, allow_synthetic=False, require_review=True, verify_chunks=True):
    if type(allow_synthetic) is not bool or type(require_review) is not bool or type(verify_chunks) is not bool:
        raise ValueError('Corpus flags must be explicit booleans')
    if isinstance(value, (str, Path)):
        value = load_json(artifact_path(root, str(value)))
    object_fields(value, ('format', 'name', 'entries'), ('format', 'entries'))
    if value['format'] != CORPUS_FORMAT or not isinstance(value['entries'], list) or not 1 <= len(value['entries']) <= 1000:
        raise ValueError('Invalid bounded V2 feature corpus')
    events, seen, ids, source_splits, source_owners, space = [], set(), {}, {}, {}, None
    for item in value['entries']:
        object_fields(item, ('index', 'cache_id'), ('index', 'cache_id'))
        path = artifact_path(root, item['index'])
        index = validate_index(load_json(path))
        doc, m = index['document'], index['document']['manifest']
        if require_review and (index['config']['purpose'] != 'training' or not doc['review']):
            raise ValueError('Learning/evaluation targets require reviewed research-sample indexes; preview cannot be promoted')
        if m['source'] == 'synthetic' and not allow_synthetic:
            raise ValueError('Explicit synthetic acknowledgment required')
        if require_review and m.get('trajectory_origin') and not (m['shot_preferences'] or any(r['label']!='uncertain' for r in m['comparisons'])):
            raise ValueError('Correction trajectory needs new reviewed action/clip annotations, not copied event facts only')
        bind_media(root, m, verify=True)
        header = cache_header(root, item['cache_id'])
        if header['identity']['index_sha256'] != index['index_sha256']:
            raise ValueError('Corpus feature/index/review identity mismatch')
        identity = (index['index_sha256'], item['cache_id'])
        if identity in seen:
            raise ValueError('Duplicate corpus member')
        seen.add(identity)
        new_space = feature_space(header)
        if space is not None and space != new_space:
            raise ValueError('Mixed encoder weights/preprocessing/feature space; re-extract consistently')
        space = new_space
        event_id, split = m['original_event_id'], m['split']
        if event_id in ids:
            old_split, old_source, trajectories = ids[event_id]
            trajectory = m.get('trajectory_origin', {}).get('replay_sha256')
            if split != old_split or split != 'train' or old_source != index['media_identity'] or not trajectory or trajectory in trajectories:
                raise ValueError('Duplicate original event/revisions or cross-split event leakage')
            trajectories.add(trajectory)
        else:
            ids[event_id] = (split, index['media_identity'], {m.get('trajectory_origin', {}).get('replay_sha256')})
        for c in m['cameras']:
            for key in ('video_sha256', 'audio_sha256'):
                sha = c[key]
                if sha is not None:
                    if sha in source_splits and source_splits[sha] != split:
                        raise ValueError('Source-content leakage across corpus splits (including renamed media)')
                    if sha in source_owners and source_owners[sha] != event_id:
                        raise ValueError('Identical source content under different event IDs is not independent evidence')
                    source_splits[sha] = split
                    source_owners[sha] = event_id
        if verify_chunks:
            for _ in iter_features(root, header['cache_id']):
                pass
        events.append(FeatureEvent(index, header, path.relative_to(Path(root).resolve()).as_posix()))
    return events


def corpus_identity(events):
    return fingerprint(sorted((dict(index_sha256=e.index['index_sha256'], cache_id=e.cache_id,
                                    feature_content_sha256=e.header['content_sha256']) for e in events), key=lambda v: (v['index_sha256'], v['cache_id'])))


def provenance(events):
    return [dict(original_event_id=e.manifest['original_event_id'], split=e.manifest['split'], source=e.manifest['source'],
                 index_sha256=e.index['index_sha256'], cache_id=e.cache_id,
                 content_sha256=e.index['document']['content_sha256'], review=e.index['document']['review'],
                 media_hashes=sorted({c[k] for c in e.manifest['cameras'] for k in ('video_sha256', 'audio_sha256') if c[k]})) for e in events]


def save_corpus(root, entries, output, allow_synthetic=False, name=''):
    value = dict(format=CORPUS_FORMAT, name=name, entries=entries)
    events = load_corpus(root, value, allow_synthetic)
    path = artifact_path(root, str(output), False)
    if path.exists():
        raise ValueError('Corpus output already exists')
    atomic_json(path, value)
    return dict(path=path.relative_to(Path(root).resolve()).as_posix(), corpus_sha256=corpus_identity(events),
                events=len({e.manifest['original_event_id'] for e in events}), feature_space_id=feature_space_id(events[0].header),
                splits={s: sum(e.manifest['split'] == s for e in events) for s in ('train', 'validation', 'test')}, production_approved=False)
