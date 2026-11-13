"""Additive authenticated Studio endpoints and isolated P0/P1 worker specs."""
from pathlib import Path

from .annotations_v2 import EventStore, import_v1_media
from .contracts_v2 import integer, load_json
from .prepare_v2 import index_event, validate_index
from .storage import digest_file
from .v2 import artifact_path, baseline_status


class V2Studio:
    def __init__(self, manager):
        self.manager = manager
        self.root = manager.root
        self.store = EventStore(self.root)

    def get(self, route, query):
        first = lambda name, default='': query.get(name, [default])[0]
        if route == '/api/studio/v2/status':
            base = self.root / 'runs/v2'
            return dict(stage='P0/P1 research foundation', v2_model_available=False, production_approved=False,
                        indexes=[p.relative_to(self.root).as_posix() for p in sorted(base.glob('*/index.json'))[-100:]],
                        feature_reports=[p.relative_to(self.root).as_posix() for p in sorted(base.glob('*/feature-report.json'))[-100:]],
                        replays=[p.relative_to(self.root).as_posix() for p in sorted(base.glob('*/replay/replay-report.json'))[-100:]],
                        checkpoints=[p.relative_to(self.root).as_posix() for p in sorted((self.root / 'runs').glob('*/best.pt'))],
                        source_note='V1 weights remain weak-supervision baselines; no foundation weights are downloaded')
        if route == '/api/studio/v2/projects':
            return dict(projects=self.store.list())
        if route == '/api/studio/v2/project':
            revision = int(first('revision')) if first('revision') else None
            return self.store.get(first('id'), revision)
        if route == '/api/studio/v2/baseline':
            return baseline_status(self.root)
        if route == '/api/studio/v2/weights':
            path = artifact_path(self.root, first('path'))
            if path.suffix not in ('.pt', '.pth'):
                raise ValueError('Local weight file required')
            return dict(path=path.relative_to(self.root).as_posix(), sha256=digest_file(path))
        if route == '/api/studio/v2/report':
            path = artifact_path(self.root, first('path'))
            if not path.is_relative_to(self.root / 'runs/v2') or path.suffix != '.json':
                raise ValueError('Only V2 JSON reports')
            return load_json(path)
        raise ValueError('Unknown V2 endpoint')

    def post(self, route, payload):
        if not isinstance(payload, dict):
            raise ValueError('JSON object required')
        if route == '/api/studio/v2/create':
            return self.store.create(payload['manifest'])
        if route == '/api/studio/v2/import-v1':
            return self.store.create(import_v1_media(self.root, payload['manifest']))
        if route == '/api/studio/v2/save':
            return self.store.save(payload['id'], payload['revision'], payload['manifest'])
        if route == '/api/studio/v2/review':
            return self.store.review(payload['id'], payload['revision'], payload.get('reviewer'),
                                     payload.get('alignment_confirmed'), payload.get('labels_confirmed'))
        jobs = {'/api/studio/v2/' + name: 'v2_' + name for name in ('demo', 'index', 'extract', 'replay', 'quality')}
        if route in jobs:
            return self.manager.start({**payload, 'kind': jobs[route]})
        raise ValueError('Unknown V2 operation')


def job_spec(manager, payload, jobdir):
    """No V1 config/dataset required; all compute runs under the existing worker lock."""
    from .web_core import stamp
    root, kind, uid = manager.root, payload['kind'], jobdir.name
    store = EventStore(root)
    out = root / 'runs/v2' / uid
    args = ['--workspace', str(root), '--stop-file', str(jobdir / 'stop.request')]
    if kind == 'v2_demo':
        if payload.get('ack_synthetic') is not True:
            raise ValueError('Explicit synthetic acknowledgment required')
        out = root / 'data' / ('v2_synthetic_' + uid)
        args += ['demo', '--output', str(out), '--accept-synthetic']
    elif kind == 'v2_index':
        revision = integer(payload.get('revision'), 'revision', 1, 1000000)
        doc = store.get(payload.get('id'), revision)
        config = dict(decision_hz=payload.get('decision_hz', 2.), chunk_steps=payload.get('chunk_steps', 64),
                      history_s=payload.get('history_s', 8.), purpose=payload.get('purpose', 'preview'),
                      allow_synthetic=payload.get('allow_synthetic', False))
        index_event(doc, **config)
        args += ['index', '--project', doc['id'], '--revision', str(revision), '--output', str(out / 'index.json'),
                 '--decision-hz', str(config['decision_hz']), '--chunk-steps', str(config['chunk_steps']),
                 '--history-s', str(config['history_s']), '--purpose', config['purpose']]
        if config['allow_synthetic']:
            args.append('--allow-synthetic')
    elif kind == 'v2_extract':
        from .features_v2 import local_weights
        path = artifact_path(root, payload.get('index'))
        validate_index(load_json(path))
        encoder = payload.get('encoder', 'v1_visual')
        if encoder not in ('v1_visual', 'mobilenet_v3_small'):
            raise ValueError('Invalid offline encoder kind')
        weights = local_weights(root, payload.get('weights'), payload.get('weights_sha256'))
        args += ['extract', '--index', str(path), '--encoder', encoder, '--weights', str(weights),
                 '--weights-sha256', payload['weights_sha256'], '--output', str(out / 'feature-report.json')]
    elif kind == 'v2_replay':
        from .features_v2 import cache_header
        path = artifact_path(root, payload.get('index'))
        index = validate_index(load_json(path))
        cache = cache_header(root, payload.get('cache_id'))
        if cache['identity']['index_sha256'] != index['index_sha256']:
            raise ValueError('Replay feature/index identity mismatch')
        policy = payload.get('policy', 'hold_valid')
        if policy not in ('hold_valid', 'audio_rms'):
            raise ValueError('Only non-learned baselines currently available')
        from .contracts_v2 import number
        hold = number(payload.get('min_hold_s', 2.), 'min hold', 0, 60)
        args += ['replay', '--index', str(path), '--cache-id', cache['cache_id'], '--policy', policy,
                 '--min-hold-s', str(hold), '--output', str(out / 'replay')]
    elif kind == 'v2_quality':
        selection = payload.get('projects')
        if not isinstance(selection, list) or not 1 <= len(selection) <= 1000:
            raise ValueError('Select 1..1000 event revisions')
        paths = [store.path(p['id'], integer(p['revision'], 'revision', 1, 1000000)) for p in selection]
        args += ['quality', '--events'] + [str(p) for p in paths] + ['--output', str(out / 'quality.json')]
    else:
        raise ValueError('Unknown V2 job')
    return dict(id=uid, kind=kind, workspace=str(root), created_at=stamp(), status='queued', output=str(out),
                module='camix_director.v2', args=args)
