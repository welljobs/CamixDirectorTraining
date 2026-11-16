import json
import re
import subprocess
import shutil
import threading
import time
import urllib.request
import urllib.error

import pytest
from camix_director.contracts_v2 import load_json
from camix_director.stages_api_v2 import catalogue
from camix_director.web import make_server
from camix_director.web_core import CODE_ROOT,JobManager
from camix_director.web_worker import ALLOWED
from v2_helpers import encoder_weights


@pytest.fixture
def manager(tmp_path):
    m=JobManager(tmp_path,system=dict(os='Darwin',devices=[dict(id='cpu',available=True)],coreml_export_available=False))
    yield m
    m.close()


def await_job(manager,uid):
    end=time.monotonic()+90
    while time.monotonic()<end:
        manager.refresh();job=manager.get(uid)
        if job['status'] in ('completed','failed','cancelled','interrupted'):
            assert job['status']=='completed',manager.logs(uid)
            if manager.active:manager.active[1].wait(timeout=15);manager.refresh()
            return job
        time.sleep(.1)
    pytest.fail('Stage worker timeout')


def test_stage_smoke_runs_all_paths_in_the_real_isolated_worker(tmp_path,manager):
    weights,sha=encoder_weights(tmp_path)
    job=manager.start(dict(kind='v2_stage_smoke',weights=str(weights),weights_sha256=sha,allow_synthetic=True))
    done=await_job(manager,job['id']);out=__import__('pathlib').Path(done['output'])
    report=load_json(out/'stages-smoke-report.json')
    assert report['status']=='completed' and report['source']=='synthetic' and not report['human_director_evidence']
    assert report['research_gate_status']=='blocked' and not report['production_approved']
    assert len(report['completed_paths'])>=10
    c=catalogue(tmp_path)
    assert len(c['members'])==3 and len(c['policies'])==3 and c['corpora']
    assert c['quality_evaluations'] and c['shadows'] and c['gates']
    signature=load_json(out/'export/signature.json')
    assert signature['requires_external_encoder_and_causal_adapter']


def test_stage_specs_reject_truthy_flags_unavailable_devices_and_unsafe_paths(tmp_path,manager):
    directory=tmp_path/'job-spec';directory.mkdir()
    with pytest.raises(ValueError,match='boolean'):manager._spec(dict(kind='v2_stage_corpus',entries=[{}],allow_synthetic='true'),directory)
    with pytest.raises(ValueError,match='Unknown'):manager._spec(dict(kind='v2_stage_online_take'),directory)
    cp=tmp_path/'runs/test/best.pt';cp.parent.mkdir(parents=True);cp.write_bytes(b'not loaded in export spec')
    with pytest.raises(ValueError,match='boolean'):manager._spec(dict(kind='v2_stage_export',checkpoint=str(cp),coreml='true'),directory)
    with pytest.raises(ValueError,match='unavailable'):manager._spec(dict(kind='v2_stage_train',config=None,device='cuda'),directory)
    assert 'camix_director.stages_v2' in ALLOWED


def test_new_stage_web_page_and_api_keep_auth_origin_and_entry_protection(tmp_path,manager):
    server=make_server(manager,0,'stages-token');thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    def get(path,headers=None):return urllib.request.urlopen(urllib.request.Request(base+path,headers={'X-Camix-Token':'stages-token',**(headers or {})}))
    try:
        assert get('/v2-model').status==200 and get('/v2-model.js').status==200
        value=json.load(get('/api/studio/v2/stages/catalog'))
        assert value['policy_implementation_available'] and not value['production_approved']
        with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(base+'/api/studio/v2/stages/catalog')
        assert error.value.code==403
        with pytest.raises(urllib.error.HTTPError) as error:get('/api/studio/v2/stages/catalog',{'Origin':'https://evil.invalid'})
        assert error.value.code==403
        with pytest.raises(urllib.error.HTTPError) as error:get('/v2-model.js',{'Sec-Fetch-Site':'cross-site'})
        assert error.value.code==403
        assert get('/v2-model',{'Sec-Fetch-Site':'cross-site','Sec-Fetch-Mode':'navigate','Sec-Fetch-Dest':'document'}).status==200
    finally:server.shutdown();server.server_close();thread.join()


def test_model_ui_hooks_and_scripts_parse():
    from html.parser import HTMLParser
    root=CODE_ROOT/'camix_director/web_assets'
    class Ids(HTMLParser):
        def __init__(self):super().__init__();self.ids=[]
        def handle_starttag(self,tag,attrs):self.ids.extend(v for k,v in attrs if k=='id')
    p=Ids();p.feed((root/'v2-model.html').read_text())
    script=(root/'v2-model.js').read_text()
    assert len(p.ids)==len(set(p.ids))
    assert set(re.findall(r"\$\('([^']+)'\)",script))<=set(p.ids)
    if shutil.which('node'):subprocess.run(['node','--check',str(root/'v2-model.js')],check=True,capture_output=True)
