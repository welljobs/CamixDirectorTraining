"""P0/P1 command line. No model training, weight download, or live control."""
import argparse
from pathlib import Path

from .annotations_v2 import EventStore, import_v1_media
from .contracts_v2 import load_json, media_identity
from .prepare_v2 import collection_quality, prepare_index
from .runtime import atomic_json


def artifact_path(root, value, exists=True):
    root = Path(root).resolve()
    if not isinstance(value, str) or not value:
        raise ValueError('Missing artifact path')
    path = (root / value).resolve()
    if not any(path.is_relative_to(root / folder) for folder in ('runs', 'data')):
        raise ValueError('V2 artifacts must stay under workspace runs or data')
    if exists and not path.is_file():
        raise ValueError('Missing V2 artifact')
    return path


def baseline_status(root):
    from .storage import digest_file
    root = Path(root).resolve()
    path = root / 'reports/v2-foundation/v1-baseline.json'
    if not path.exists():
        return dict(available=False, preserved=None)
    baseline = load_json(path)
    changed = []
    for group in ('protected_sources', 'checkpoints'):
        for name, expected in baseline[group].items():
            source = (root / name).resolve()
            if not source.is_relative_to(root) or not source.is_file() or digest_file(source) != expected:
                changed.append(name)
    return dict(available=True, preserved=not changed, changed=changed,
                protected_sources=len(baseline['protected_sources']), checkpoints=len(baseline['checkpoints']))


def demo(root, output, accept_synthetic=False):
    if accept_synthetic is not True:
        raise ValueError('Synthetic demo requires explicit acknowledgment')
    from .raw_demo import create
    output = Path(output)
    store = EventStore(root)
    paths = create(output, seconds=6)
    projects = []
    for path in paths:
        m = import_v1_media(root, path)
        m['license'] = dict(name='Locally generated synthetic UI fixture; no director ground truth', usage_confirmed=True)
        m['shot_preferences'] = [dict(id='example-shot', start_s=1., end_s=5.,
                                      acceptable_cameras=['cam_a', 'cam_b'], cut_window_s=[2., 3.],
                                      context_program=None, intent='unknown', reason_tags=[])]
        m['events'] = [dict(id='example-event', start_s=2., end_s=4., label='SYNTHETIC test marker', visible_cameras=['cam_b'], importance=.5)]
        project = store.create(m)
        m = project['manifest']
        m['comparisons'] = [dict(id='example-comparison', start_s=1., end_s=5., source_identity=media_identity(m),
                                 a=dict(initial_program='cam_a', cuts=[]),
                                 b=dict(initial_program='cam_a', cuts=[dict(time_s=2., camera_id='cam_b')]),
                                 label='uncertain', reviewer='synthetic_example_not_human_review', reason_tags=[],
                                 notes='Example only; play and replace with your own reviewed judgment.')]
        project = store.save(project['id'], project['revision'], m)
        projects.append(dict(id=project['id'], revision=project['revision'], source='synthetic'))
    result = dict(projects=projects, source='synthetic', human_reviewed=False, production_approved=False,
                  warning='Generated workflow examples; not real data or professional quality evidence')
    atomic_json(output / 'v2-demo.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', default='.')
    parser.add_argument('--stop-file')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    command = sub.add_parser('import-v1')
    command.add_argument('--manifest', required=True)
    command = sub.add_parser('demo')
    command.add_argument('--output', required=True)
    command.add_argument('--accept-synthetic', action='store_true')
    command = sub.add_parser('index')
    command.add_argument('--project', required=True)
    command.add_argument('--revision', type=int, required=True)
    command.add_argument('--output', required=True)
    command.add_argument('--decision-hz', type=float, default=2.)
    command.add_argument('--chunk-steps', type=int, default=64)
    command.add_argument('--history-s', type=float, default=8.)
    command.add_argument('--purpose', choices=('preview', 'training'), default='preview')
    command.add_argument('--allow-synthetic', action='store_true')
    command = sub.add_parser('quality')
    command.add_argument('--events', nargs='+', required=True)
    command.add_argument('--output', required=True)
    command = sub.add_parser('extract')
    command.add_argument('--index', required=True)
    command.add_argument('--encoder', choices=('v1_visual', 'mobilenet_v3_small'), required=True)
    command.add_argument('--weights', required=True)
    command.add_argument('--weights-sha256', required=True)
    command.add_argument('--output', required=True)
    command = sub.add_parser('replay')
    command.add_argument('--index', required=True)
    command.add_argument('--cache-id', required=True)
    command.add_argument('--output', required=True)
    command.add_argument('--policy', choices=('hold_valid', 'audio_rms'), default='hold_valid')
    command.add_argument('--min-hold-s', type=float, default=2.)
    args = parser.parse_args()
    root = Path(args.workspace).resolve()
    store = EventStore(root)
    try:
        if args.command == 'status':
            result = dict(baseline=baseline_status(root), projects=store.list(), v2_model_available=False, production_approved=False)
        elif args.command == 'import-v1':
            result = store.create(import_v1_media(root, args.manifest))
        elif args.command == 'demo':
            result = demo(root, artifact_path(root, args.output, False), args.accept_synthetic)
        elif args.command == 'index':
            result = prepare_index(root, store.get(args.project, args.revision), artifact_path(root, args.output, False),
                                   decision_hz=args.decision_hz, chunk_steps=args.chunk_steps, history_s=args.history_s,
                                   purpose=args.purpose, allow_synthetic=args.allow_synthetic)
            result = dict(index_sha256=result['index_sha256'], ticks=result['ticks'], chunks=len(result['chunks']), output=args.output)
        elif args.command == 'quality':
            result = collection_quality(root, [load_json(artifact_path(root, p)) for p in args.events])
            output = artifact_path(root, args.output, False)
            if output.exists():
                raise ValueError('Quality output exists')
            atomic_json(output, result)
        elif args.command == 'extract':
            from .features_v2 import extract
            result = extract(root, load_json(artifact_path(root, args.index)), args.encoder, args.weights, args.weights_sha256, args.stop_file)
            result = dict(cache_id=result['cache_id'], feature_dim=result['feature_dim'], chunks=len(result['chunks']),
                          description=result['description'], production_approved=False,
                          index=artifact_path(root, args.index).relative_to(root).as_posix(),
                          encoder_kind=args.encoder, weights_sha256=args.weights_sha256,
                          video_valid_fraction=result['video_valid_fraction'], audio_valid_fraction=result['audio_valid_fraction'])
            output = artifact_path(root, args.output, False)
            if output.exists():
                raise ValueError('Feature report exists')
            atomic_json(output, result)
        else:
            from .replay_v2 import replay
            result = replay(root, load_json(artifact_path(root, args.index)), args.cache_id,
                            artifact_path(root, args.output, False), args.policy, args.min_hold_s, args.stop_file)
        import json
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    except InterruptedError as error:
        print(str(error), flush=True)
        raise SystemExit(130)


if __name__ == '__main__':
    main()
