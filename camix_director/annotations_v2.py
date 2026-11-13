"""Revisioned V2 events and explicit, content-bound local human review.

Reviewers are self-declared, not enterprise-verified or certified directors.
No V1 labels/review flags are implicitly promoted to V2.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
import json
import uuid

from .contracts_v2 import (EVENT_FORMAT, REVIEW_FORMAT, data_path,
                           fingerprint, integer, load_json, media_identity, object_fields, text, validate_manifest)
from .runtime import atomic_json, file_lock
from .storage import digest_file


def now():
    return datetime.now(timezone.utc).isoformat()


def bind_media(root, manifest, verify=False):
    if not isinstance(manifest, dict) or not isinstance(manifest.get('cameras'), list) or not 1 <= len(manifest['cameras']) <= 16:
        raise ValueError('V2 manifest requires 1..16 cameras')
    m = deepcopy(manifest)
    for camera in m['cameras']:
        if not isinstance(camera, dict):
            raise ValueError('Invalid camera object')
        for kind in ('video', 'audio'):
            value = camera.get(kind)
            if value is None and kind == 'audio':
                if verify and camera.get('audio_sha256') is not None:
                    raise ValueError('Missing audio identity')
                camera['audio_sha256'] = None
                continue
            path = data_path(root, value)
            digest = digest_file(path)
            if verify and camera.get(kind + '_sha256') != digest:
                raise ValueError('Source media changed after binding/review: ' + str(value))
            camera[kind] = path.relative_to(Path(root).resolve()).as_posix()
            camera[kind + '_sha256'] = digest
    return validate_manifest(m)


def validate_document(value):
    if not isinstance(value, dict) or value.get('format') != REVIEW_FORMAT:
        raise ValueError('Not a reviewed-event V2 document')
    fields = ('format', 'id', 'revision', 'updated_at', 'manifest', 'content_sha256', 'media_identity', 'review', 'production_approved', 'execute_allowed')
    object_fields(value, fields, fields)
    if not isinstance(value['id'], str) or not re.fullmatch('[0-9a-f]{32}', value['id']):
        raise ValueError('Invalid event document ID')
    integer(value['revision'], 'revision', 1, 1000000)
    text(value['updated_at'], 'document timestamp', 80)
    if value['production_approved'] is not False or value['execute_allowed'] is not False:
        raise ValueError('Research documents cannot grant production/execute permission')
    m = validate_manifest(value['manifest'])
    if value['media_identity'] != media_identity(m):
        raise ValueError('Event media identity changed')
    if fingerprint(m) != value.get('content_sha256'):
        raise ValueError('Event content identity changed')
    review = value.get('review')
    if review is not None:
        fields = ('content_sha256', 'reviewer', 'reviewed_at', 'alignment_confirmed', 'labels_confirmed', 'identity_type')
        object_fields(review, fields, fields)
        if review['identity_type'] != 'local_self_declared_not_certified':
            raise ValueError('Local review cannot certify professional identity')
        if (not isinstance(review, dict) or review.get('content_sha256') != value['content_sha256']
                or review.get('alignment_confirmed') is not True or review.get('labels_confirmed') is not True
                or not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip()):
            raise ValueError('Invalid content-bound review')
    return deepcopy(value)


class EventStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.base = self.root / 'data/v2/projects'

    def directory(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch('[0-9a-f]{32}', identifier):
            raise ValueError('Invalid V2 project ID')
        path = self.base / identifier
        if not path.resolve().is_relative_to(self.root / 'data'):
            raise ValueError('Project symlink escape')
        return path

    def list(self):
        result = []
        for path in sorted(self.base.glob('*/draft.json')):
            if len(result) >= 500:
                break
            project = self.get(path.parent.name)
            m = project['manifest']
            result.append(dict(id=project['id'], revision=project['revision'], original_event_id=m['original_event_id'],
                               source=m['source'], split=m['split'], duration_s=m['duration_s'],
                               reviewed=project['review'] is not None, content_sha256=project['content_sha256']))
        return result

    def get(self, identifier, revision=None):
        directory = self.directory(identifier)
        if revision is not None:
            integer(revision, 'revision', 1, 1000000)
        path = directory / ('draft.json' if revision is None else f'rev-{revision:06}.json')
        if not path.is_file():
            raise ValueError('Missing V2 project revision')
        project = validate_document(load_json(path))
        if project.get('id') != identifier or (revision is not None and project.get('revision') != revision):
            raise ValueError('Project identity mismatch')
        return project

    def _write(self, directory, identifier, revision, manifest, review=None):
        value = dict(format=REVIEW_FORMAT, id=identifier, revision=revision, updated_at=now(),
                     manifest=manifest, content_sha256=fingerprint(manifest), media_identity=media_identity(manifest), review=review,
                     production_approved=False, execute_allowed=False)
        if len(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')) > 8 * 1024 * 1024:
            raise ValueError('Event revision exceeds bounded JSON size; reduce annotations per revision')
        path = directory / f'rev-{revision:06}.json'
        if path.exists():
            raise ValueError('Revision already exists; refusing overwrite')
        atomic_json(path, value)
        atomic_json(directory / 'draft.json', value)
        return value

    def create(self, manifest):
        m = bind_media(self.root, manifest)
        identifier = uuid.uuid4().hex
        directory = self.directory(identifier)
        directory.mkdir(parents=True, exist_ok=False)
        with file_lock(directory / 'edit.lock'):
            return self._write(directory, identifier, 1, m)

    def save(self, identifier, revision, manifest):
        integer(revision, 'revision', 1, 1000000)
        directory = self.directory(identifier)
        with file_lock(directory / 'edit.lock'):
            current = self.get(identifier)
            if current['revision'] != revision:
                raise ValueError('Revision conflict; reload before editing')
            if not isinstance(manifest, dict) or manifest.get('source') != current['manifest']['source']:
                raise ValueError('Source provenance cannot change from synthetic to real')
            m = bind_media(self.root, manifest)
            return self._write(directory, identifier, revision + 1, m)

    def review(self, identifier, revision, reviewer, alignment_confirmed, labels_confirmed):
        integer(revision, 'revision', 1, 1000000)
        if alignment_confirmed is not True or labels_confirmed is not True:
            raise ValueError('Explicit boolean alignment and labels confirmation required')
        if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 160:
            raise ValueError('Self-declared reviewer required')
        directory = self.directory(identifier)
        with file_lock(directory / 'edit.lock'):
            current = self.get(identifier)
            if current['revision'] != revision:
                raise ValueError('Revision conflict; reload before review')
            m = bind_media(self.root, current['manifest'], verify=True)
            if not m['license']['usage_confirmed']:
                raise ValueError('Confirm authorized source usage before review')
            if not any(m[key] for key in ('shot_preferences', 'events', 'comparisons')):
                raise ValueError('No annotations to review; use preview mode')
            evidence = dict(content_sha256=fingerprint(m), reviewer=reviewer.strip(),
                            reviewed_at=now(), alignment_confirmed=True, labels_confirmed=True,
                            identity_type='local_self_declared_not_certified')
            return self._write(directory, identifier, revision + 1, m, evidence)

    def path(self, identifier, revision):
        self.get(identifier, revision)
        return self.directory(identifier) / f'rev-{revision:06}.json'


def import_v1_media(root, manifest_path):
    """Explicit media/PGM import only. Legacy point labels and review are not copied."""
    root = Path(root).resolve()
    path = data_path(root, str(manifest_path))
    old = load_json(path)
    if old.get('source') not in ('real', 'synthetic'):
        raise ValueError('Legacy manifest needs explicit source provenance')
    cameras = []
    start = float(old.get('start_seconds', 0))
    for c in old['cameras']:
        row = dict(id=c['id'], video_offset_s=start + float(c.get('video_offset', 0)),
                   audio_offset_s=start + float(c.get('audio_offset', 0)),
                   video_delay_s=c.get('video_delay', 0), audio_delay_s=c.get('audio_delay', 0))
        for kind in ('video', 'audio'):
            source = data_path(root, str(path.parent / c[kind])) if c.get(kind) else None
            row[kind] = source.relative_to(root).as_posix() if source else None
        cameras.append(row)
    cuts = []
    if old.get('program_events'):
        source = data_path(root, str(path.parent / old['program_events']))
        if source.stat().st_size > 8 * 1024 * 1024:
            raise ValueError('Legacy timeline too large')
        import json
        for line in source.read_text(encoding='utf-8').splitlines():
            if line.strip():
                cut = json.loads(line)
                cuts.append(dict(time_s=cut['time_s'], camera_id=cut['target']))
    return dict(format=EVENT_FORMAT, original_event_id=old['session_id'], source=old['source'],
                split=old['split'], duration_s=old['duration_seconds'], initial_program=old['initial_program'],
                cameras=cameras, clock=dict(mode='strict_live', delay_s=0),
                license=dict(name=old.get('license', 'Unconfirmed legacy source; fill in authorization'), usage_confirmed=False),
                scenario_tags=old.get('scenario_tags', []), style=dict(name='unspecified', notes=''),
                program_events=cuts, shot_preferences=[], events=[], comparisons=[])
