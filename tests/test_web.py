import io
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from camix_director.config import ModelConfig
from camix_director.synthetic import create_fixture
from camix_director.train import train, load_checkpoint, TrainingCancelled
from camix_director.runtime import file_lock
from camix_director.web_core import JobManager, scoped, pack_info, CODE_ROOT
from camix_director.web import make_server, preview, strict_json


@pytest.fixture
def config():
    return ModelConfig(max_cameras=2,image_size=16,audio_samples=256,hidden_dim=16,layers=1,memory_tokens=4,dropout=0)


@pytest.fixture
def manager(tmp_path,config):
    create_fixture(tmp_path/'data/small/prepared',config,steps=12,sessions=3)
    system=dict(os='Windows',ram_bytes=32*1024**3,devices=[dict(id='cpu',available=True),dict(id='cuda',available=False),dict(id='mps',available=False)],coreml_export_available=False)
    result=JobManager(tmp_path,CODE_ROOT,system=system)
    yield result
    result.close()


def train_payload(config,**extra):
    return dict(kind='train',data='data/small/prepared',config=config.to_dict(),epochs=2,sequence_steps=6,burn_in=1,batch_size=2,device='cpu',allow_synthetic=True,**extra)


def await_job(manager,uid,timeout=35):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        manager.refresh();job=manager.get(uid)
        if job['status'] in ('completed','failed','cancelled'):
            return job
        time.sleep(.1)
    raise AssertionError('Job timeout: '+manager.logs(uid))


def test_workspace_escape_and_symlink_are_refused(tmp_path):
    with pytest.raises(ValueError):scoped(tmp_path,'../outside')
    if hasattr(Path,'symlink_to'):
        try:(tmp_path/'escape').symlink_to(tmp_path.parent,target_is_directory=True)
        except OSError:return  # Windows may require symlink privileges.
        with pytest.raises(ValueError):scoped(tmp_path,'escape/file')


def test_catalog_metadata_and_memory_estimate(manager):
    result=manager.catalog();dataset=result['datasets'][0]
    assert dataset['splits']==dict(train=3,validation=1,test=1)
    assert dataset['raw_bytes']>0 and dataset['packs'][0]['steps']==12
    assert manager.browse('data')['entries'][0]['directory']


def test_synthetic_permission_is_explicit(manager,config):
    payload=train_payload(config);payload['allow_synthetic']=False
    with pytest.raises(ValueError,match='合成'):manager.start(payload)
    assert list(manager.jobs_dir.iterdir())==[]


def test_unavailable_cuda_and_bad_sequence_are_refused(manager,config):
    payload=train_payload(config);payload['device']='cuda'
    with pytest.raises(ValueError,match='不可用'):manager.start(payload)
    payload=train_payload(config);payload['burn_in']=6
    with pytest.raises(ValueError,match='burn-in'):manager.start(payload)
    payload=train_payload(config);payload['lr']=float('nan')
    with pytest.raises(ValueError,match='学习率'):manager.start(payload)


def test_coreml_is_explicitly_disabled_on_windows(manager):
    with pytest.raises(ValueError,match='Windows'):manager.start(dict(kind='export_coreml',checkpoint='runs/a/best.pt'))


def test_local_api_auth_origin_host_and_static_assets(manager):
    server=make_server(manager,0,token='local-secret');thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}'
    try:
        response=urllib.request.urlopen(url+'/');assert 'Camix'.encode() in response.read()
        assert "frame-ancestors 'none'" in response.headers['Content-Security-Policy']
        with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(url+'/api/state')
        assert error.value.code==403
        request=urllib.request.Request(url+'/api/state',headers={'X-Camix-Token':'local-secret'})
        assert json.load(urllib.request.urlopen(request))['system']['os']=='Windows'
        request=urllib.request.Request(url+'/api/state',headers={'X-Camix-Token':'local-secret','Origin':'https://attacker.invalid'})
        with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(request)
        assert error.value.code==403
        request=urllib.request.Request(url+'/api/state',headers={'X-Camix-Token':'local-secret','Host':'attacker.invalid'})
        with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(request)
        assert error.value.code==403
        request=urllib.request.Request(url+'/api/artifact?path=README.md',headers={'X-Camix-Token':'local-secret'})
        with pytest.raises(urllib.error.HTTPError):urllib.request.urlopen(request)
    finally:server.shutdown();server.server_close();thread.join()


def test_preview_png_audio_and_bounds(manager):
    path='data/small/prepared/train_0.npz'
    image,mime=preview(manager.root,{'path':[path],'tick':['1']})
    assert image.startswith(b'\xff\xd8') and mime=='image/jpeg'
    sound,mime=preview(manager.root,{'path':[path],'tick':['1'],'camera':['0']},True)
    assert sound.startswith(b'RIFF') and mime=='audio/wav'
    with pytest.raises(ValueError):preview(manager.root,{'path':[path],'tick':['999']})


def test_real_worker_trains_and_persists_reports(manager,config):
    job=manager.start(train_payload(config))
    with pytest.raises(ValueError,match='已有'):manager.start(train_payload(config))
    result=await_job(manager,job['id'])
    assert result['status']=='completed',manager.logs(job['id'])
    assert len(result['training_report']['history'])==2
    assert result['progress']['phase']=='completed'
    assert result['checkpoint'].endswith('best.pt')
    _,checkpoint=load_checkpoint(manager.root/result['checkpoint'])
    assert checkpoint['qualification']=='synthetic_smoke_only' and not checkpoint['production_approved']
    assert 'LOCAL EXPERIMENT ONLY' in manager.logs(job['id'])


def test_cancel_preserves_completed_checkpoint(manager,config):
    payload=train_payload(config);payload['epochs']=1000
    job=manager.start(payload);uid=job['id'];end=time.monotonic()+35
    while time.monotonic()<end:
        current=manager.get(uid)
        if current.get('checkpoint'):break
        if current['status']=='failed':pytest.fail(manager.logs(uid))
        time.sleep(.1)
    else:pytest.fail('No checkpoint appeared')
    manager.stop(uid);result=await_job(manager,uid)
    assert result['status']=='cancelled',manager.logs(uid)
    model,cp=load_checkpoint(manager.root/result['checkpoint'])
    assert cp['production_approved'] is False
    assert (Path(result['output'])/'training-report.json').exists()


def test_early_training_cancel_is_reported(tmp_path,config):
    stop=tmp_path/'stop';stop.touch()
    with pytest.raises(TrainingCancelled):train(tmp_path/'missing',tmp_path/'out',config,stop_file=stop)
    result=json.loads((tmp_path/'out/training-report.json').read_text())
    assert result['status']=='cancelled' and not result['checkpoint_available']


def test_file_lock_releases_and_excludes(tmp_path):
    path=tmp_path/'worker.lock'
    with file_lock(path,timeout=0):
        with pytest.raises(TimeoutError):
            with file_lock(path,timeout=0):pass
    with file_lock(path,timeout=0):pass


def test_windows_lock_branch_uses_msvcrt(tmp_path,monkeypatch):
    import sys,types
    import camix_director.runtime as runtime
    calls=[]
    msvcrt=types.SimpleNamespace(LK_NBLCK=1,LK_UNLCK=2,locking=lambda fd,kind,size:calls.append((kind,size)))
    monkeypatch.setitem(sys.modules,'msvcrt',msvcrt)
    monkeypatch.setattr(runtime,'os',types.SimpleNamespace(name='nt',SEEK_END=2))
    with runtime.file_lock(tmp_path/'win.lock',timeout=0):pass
    assert calls==[(1,1),(2,1)]


def test_json_refuses_nonfinite():
    with pytest.raises(ValueError):strict_json('{"lr":NaN}')


def test_windows_installer_avoids_coreml_and_supports_cpu_cuda():
    text=(CODE_ROOT/'setup_windows.bat').read_text()
    assert 'cu118' in text and 'cu126' in text and 'cu128' in text and 'whl/cpu' in text
    assert 'requirements-train.txt' in text and 'requirements.lock' not in text
    assert 'coremltools' not in (CODE_ROOT/'requirements-train.txt').read_text().splitlines()[-1]

def test_windows_stop_targets_only_owned_process_tree(manager,monkeypatch):
    import types
    import camix_director.web_core as core
    calls=[]
    monkeypatch.setattr(core,'os',types.SimpleNamespace(name='nt'))
    monkeypatch.setattr(core.subprocess,'run',lambda args,**kwargs:calls.append(args))
    manager._terminate(types.SimpleNamespace(pid=654321,poll=lambda:None))
    assert calls==[['taskkill','/PID','654321','/T','/F']]


def test_web_learning_rate_default_passes_html_step_validation():
    html=(CODE_ROOT/'camix_director/web_assets/index.html').read_text(encoding='utf-8')
    assert 'id="lr" type="number" step="any"' in html

def test_atomic_replace_retries_temporary_windows_reader(tmp_path,monkeypatch):
    import camix_director.runtime as runtime
    src=tmp_path/'new';dst=tmp_path/'old';src.write_bytes(b'new');dst.write_bytes(b'old')
    replace=runtime.os.replace;calls=[]
    def busy_once(a,b):
        calls.append(1)
        if len(calls)==1:raise PermissionError('reader is closing')
        return replace(a,b)
    monkeypatch.setattr(runtime.os,'replace',busy_once)
    runtime.atomic_replace(src,dst)
    assert dst.read_bytes()==b'new' and len(calls)==2

def test_stop_confirmation_is_an_accessible_html_dialog_not_a_browser_modal():
    html=(CODE_ROOT/'camix_director/web_assets/index.html').read_text(encoding='utf-8')
    js=(CODE_ROOT/'camix_director/web_assets/app.js').read_text(encoding='utf-8')
    assert 'id="stop-dialog"' in html and 'id="confirm-stop"' in html
    assert "$('stop-dialog').showModal()" in js and 'confirm(' not in js


@pytest.mark.parametrize('route,mode,destination,status',[
    ('/','navigate','document',200),
    ('/index.html','navigate','document',200),
    ('/','navigate','iframe',403),
    ('/','cors','empty',403),
    ('/api/state','navigate','document',403),
    ('/api/media?path=data/test.mp4','navigate','document',403),
    ('/app.js','no-cors','script',403),
])
def test_cross_site_entry_navigation_keeps_api_and_embed_protection(manager,route,mode,destination,status):
    server=make_server(manager,0,token='local-secret')
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}'+route,headers={
            'Sec-Fetch-Site':'cross-site','Sec-Fetch-Mode':mode,'Sec-Fetch-Dest':destination,
            'X-Camix-Token':'local-secret',
        })
        if status==200:
            with urllib.request.urlopen(request,timeout=5) as response:
                assert response.status==200 and b'Camix' in response.read()
        else:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request,timeout=5)
            assert error.value.code==status
    finally:
        server.shutdown();server.server_close();thread.join()
