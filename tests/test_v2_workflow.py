import http.cookiejar
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from camix_director.annotations_v2 import EventStore
from camix_director.contracts_v2 import load_json
from camix_director.v2_api import V2Studio
from camix_director.web import make_server
from camix_director.web_core import CODE_ROOT, JobManager
from camix_director.web_worker import ALLOWED
from v2_helpers import encoder_weights, make_event, reviewed


@pytest.fixture
def manager(tmp_path):
    manager = JobManager(tmp_path, system=dict(os='Darwin', devices=[dict(id='cpu', available=True)], coreml_export_available=False))
    yield manager
    manager.close()


def await_job(manager, uid):
    end = time.monotonic() + 40
    while time.monotonic() < end:
        manager.refresh()
        job = manager.get(uid)
        if job['status'] in ('completed', 'failed', 'cancelled', 'interrupted'):
            assert job['status'] == 'completed', manager.logs(uid)
            if manager.active:
                manager.active[1].wait(timeout=10)
                manager.refresh()
            return job
        time.sleep(.05)
    pytest.fail('V2 worker did not finish')


def test_v2_specs_are_isolated_and_never_require_v1_tensor_packs(tmp_path, manager):
    p = reviewed(tmp_path)
    directory = tmp_path / 'spec';directory.mkdir()
    result = manager._spec(dict(kind='v2_index', id=p['id'], revision=p['revision'], purpose='preview'), directory)
    assert result['module'] == 'camix_director.v2' and result['module'] in ALLOWED
    assert '--config' not in result['args'] and '--data' not in result['args']
    assert str(tmp_path) in result['args'] and 'index' in result['args']
    with pytest.raises(ValueError, match='acknowledgment'): manager._spec(dict(kind='v2_demo', ack_synthetic='true'), directory)
    with pytest.raises(ValueError, match='review'):
        project = EventStore(tmp_path).save(p['id'], p['revision'], p['manifest'])
        manager._spec(dict(kind='v2_index', id=project['id'], revision=project['revision'], purpose='training', allow_synthetic=True), directory)


def test_cli_status_works_without_optional_backbone_download_or_training(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'camix_director.v2', '--workspace', str(tmp_path), 'status'], cwd=CODE_ROOT, capture_output=True, text=True, check=True)
    value = json.loads(result.stdout)
    assert value['projects'] == [] and not value['v2_model_available'] and not value['production_approved']


def test_worker_complete_index_features_samples_replay_pipeline(tmp_path, manager):
    from camix_director.data_v2 import EventFeatureSequences
    p = reviewed(tmp_path)
    weights, sha = encoder_weights(tmp_path)
    job = manager.start(dict(kind='v2_index', id=p['id'], revision=p['revision'], decision_hz=2,
                             chunk_steps=3, purpose='training', allow_synthetic=True))
    done = await_job(manager, job['id'])
    index_path = Path(done['output']) / 'index.json'
    assert load_json(index_path)['ticks'] == 12
    job = manager.start(dict(kind='v2_extract', index=str(index_path), encoder='v1_visual', weights=str(weights), weights_sha256=sha))
    done = await_job(manager, job['id'])
    report = load_json(Path(done['output']) / 'feature-report.json')
    assert report['encoder_kind'] == 'v1_visual' and len(report['video_valid_fraction']) == 2
    assert len(list(EventFeatureSequences(tmp_path, load_json(index_path), report['cache_id']))) == 4
    job = manager.start(dict(kind='v2_replay', index=str(index_path), cache_id=report['cache_id'], policy='audio_rms', min_hold_s=1.))
    done = await_job(manager, job['id'])
    replay = load_json(Path(done['output']) / 'replay/replay-report.json')
    assert replay['ticks'] == 12 and not replay['learned_policy'] and not replay['production_approved']
    assert EventStore(tmp_path).get(p['id'])['revision'] == p['revision']


def test_synthetic_demo_worker_never_fakes_human_review(tmp_path, manager):
    job = manager.start(dict(kind='v2_demo', ack_synthetic=True))
    done = await_job(manager, job['id'])
    result = load_json(Path(done['output']) / 'v2-demo.json')
    assert len(result['projects']) == 3 and not result['human_reviewed']
    for p in EventStore(tmp_path).list():
        assert p['source'] == 'synthetic' and not p['reviewed']
        project = EventStore(tmp_path).get(p['id'])
        assert project['manifest']['comparisons'][0]['label'] == 'uncertain'


def test_authenticated_v2_api_assets_review_and_cross_site_protection(tmp_path, manager):
    make_event(tmp_path)
    server = make_server(manager, 0, 'v2-test-token')
    thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    def request(path, body=None, headers=None):
        h = {'X-Camix-Token': 'v2-test-token', **(headers or {})}
        if body is not None: h['Content-Type'] = 'application/json'
        return urllib.request.urlopen(urllib.request.Request(base+path, data=json.dumps(body).encode() if body is not None else None, headers=h))
    try:
        for asset in ('/v2', '/v2.html', '/v2.js', '/v2.css'):
            response = request(asset)
            assert response.status == 200 and "frame-ancestors 'none'" in response.headers['Content-Security-Policy']
        with pytest.raises(urllib.error.HTTPError) as error: urllib.request.urlopen(base+'/api/studio/v2/status')
        assert error.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as error: request('/api/studio/v2/status', headers={'Origin': 'https://evil.invalid'})
        assert error.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as error: request('/v2.js', headers={'Sec-Fetch-Site':'cross-site'})
        assert error.value.code == 403
        assert request('/v2', headers={'Sec-Fetch-Site':'cross-site','Sec-Fetch-Mode':'navigate','Sec-Fetch-Dest':'document'}).status == 200
        # Use already-created test media without invoking the legacy app or V1 trainer.
        from camix_director.contracts_v2 import EVENT_FORMAT
        # Source fixture exists; obtain a manifest via a separate test store helper without recreating media.
        m = dict(format=EVENT_FORMAT, original_event_id='http-event', split='train', source='synthetic', duration_s=6., initial_program='a',
                 cameras=[dict(id=n, video=f'data/raw/{n}.mp4', audio=f'data/raw/{n}.wav', video_offset_s=0, audio_offset_s=0, video_delay_s=0, audio_delay_s=0) for n in ('a','b')],
                 clock=dict(mode='strict_live',delay_s=0),license=dict(name='synthetic test',usage_confirmed=True),scenario_tags=[],style=dict(name='test',notes=''),program_events=[],
                 shot_preferences=[dict(id='s',start_s=1,end_s=4,acceptable_cameras=['a'],cut_window_s=None,context_program=None,intent='unknown',reason_tags=[])],events=[],comparisons=[])
        p = json.load(request('/api/studio/v2/create', dict(manifest=m)))
        assert p['revision'] == 1 and p['review'] is None
        with pytest.raises(urllib.error.HTTPError) as error:
            request('/api/studio/v2/review',dict(id=p['id'],revision=1,reviewer='tester',alignment_confirmed='true',labels_confirmed=True))
        assert error.value.code == 400
        p = json.load(request('/api/studio/v2/review',dict(id=p['id'],revision=1,reviewer='synthetic tester',alignment_confirmed=True,labels_confirmed=True)))
        assert p['review'] and p['manifest']['source'] == 'synthetic'
        with pytest.raises(urllib.error.HTTPError) as error: request('/api/studio/v2/report?path=../../private.json')
        assert error.value.code == 400
    finally:
        server.shutdown();server.server_close();thread.join()


def test_ui_hooks_exist_and_javascript_parses():
    import re
    from html.parser import HTMLParser
    root = CODE_ROOT / 'camix_director/web_assets'
    html = (root / 'v2.html').read_text()
    script = (root / 'v2.js').read_text()
    class Parser(HTMLParser):
        ids = []
        def handle_starttag(self, tag, attrs):
            self.ids.extend(value for key,value in attrs if key == 'id')
    p = Parser();p.feed(html)
    assert len(p.ids) == len(set(p.ids))
    references = re.findall(r"\$\('([^']+)'\)", script)
    assert set(references) <= set(p.ids), set(references)-set(p.ids)
    import shutil
    node = shutil.which('node')
    if node: subprocess.run([node, '--check', str(root / 'v2.js')], check=True, capture_output=True)
