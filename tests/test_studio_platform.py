import http.cookiejar,json,threading,urllib.request,urllib.error
from pathlib import Path
import numpy as np
import pytest
import torch
from camix_director.config import ModelConfig
from camix_director.synthetic import create_fixture
from camix_director.train import train,TrainingCancelled
from camix_director.evaluate import evaluate
from camix_director.replay import replay
from camix_director.gates import assess
from camix_director.annotations import AnnotationStore
from camix_director.media_proxy import make_proxy
from camix_director.web_core import JobManager,CODE_ROOT
from camix_director.web import make_server
from camix_director.preflight import check_training
from test_studio_data import media_fixture

@pytest.fixture
def cfg():return ModelConfig(max_cameras=2,image_size=16,audio_samples=256,hidden_dim=16,layers=1,memory_tokens=4,dropout=0)

def fit(tmp_path,cfg):
    data=tmp_path/'data/prepared';create_fixture(data,cfg,steps=12,sessions=3)
    cp,_=train(data,tmp_path/'runs/base',cfg,epochs=1,sequence_steps=6,burn_in=1,batch_size=2,allow_synthetic=True)
    return cp,data

def test_preflight_reports_all_errors_without_starting(tmp_path,cfg):
    create_fixture(tmp_path/'data/prepared',cfg,steps=12,sessions=3)
    value=check_training(tmp_path,dict(data='data/prepared',config=cfg.to_dict(),allow_synthetic=False,batch_size=0,burn_in=9,sequence_steps=6))
    fields={e['field'] for e in value['errors']}
    assert {'synthetic','batch','burn'}<=fields and value['training_started'] is False

def test_eval_counts_original_events_not_repeated_clips(tmp_path,cfg):
    import shutil
    cp,data=fit(tmp_path,cfg);shutil.copyfile(data/'test_0.npz',data/'test_episode2.npz')
    report,_=evaluate(cp,data)
    assert report['summary']['independent_sessions']==1 and report['summary']['episodes']==2
    assert report['summary']['session_bootstrap_95_interval'] is None
    assert any(r['ablation']=='all_cameras_disconnected' and r['pairs']==0 and r['finite_output_ratio']==1 and r['unavailable_selection_mask_violations']==0 for r in report['evaluations'])

def test_replay_and_gate_do_not_claim_production(tmp_path,cfg):
    cp,data=fit(tmp_path,cfg);report,_=evaluate(cp,data)
    evaluation=tmp_path/'evaluation.json';evaluation.write_text(json.dumps(report))
    result=replay(cp,data,tmp_path/'replay',min_hold=1)
    assert result['execute_allowed'] is False and all(c['execute_allowed'] is False for s in result['sessions'] for c in s['cuts'])
    gate=assess(cp,evaluation,tmp_path/'replay/replay-report.json')
    assert gate['status']=='blocked' and not gate['production_approved']
    assert any(c['name']=='reviewed_evaluation' and not c['passed'] for c in gate['checks'])

def test_proxy_binds_source_clock_and_invalidates_on_offset_change(tmp_path):
    store=AnnotationStore(tmp_path);project=store.create(media_fixture(tmp_path))
    result=make_proxy(tmp_path,project['id'],0,tmp_path/'data/proxies/test')
    assert result['video_frames']>0 and result['audio_coverage']>.95
    session=project['session'];camera=session['cameras'][0];camera.update(preview_video=result['preview_video'],preview_audio=result['preview_audio'],proxy_signature=result['source_signature'])
    saved=store.save(project['id'],project['revision'],session);assert saved['session']['cameras'][0].get('preview_video')
    session=saved['session'];session['cameras'][0]['video_offset']=.2
    updated=store.save(saved['id'],saved['revision'],session)
    assert not updated['session']['cameras'][0].get('preview_video')

def test_authenticated_media_range_and_csrf(tmp_path):
    media_fixture(tmp_path)
    system=dict(os='Darwin',ram_bytes=34359738368,devices=[dict(id='cpu',available=True)],coreml_export_available=True)
    manager=JobManager(tmp_path,system=system);server=make_server(manager,0,'media-test');thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}';path='/api/media?path=data/raw/a.mp4'
    try:
        with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(base+path)
        assert error.value.code==403
        opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        opener.open(urllib.request.Request(base+'/api/state',headers={'X-Camix-Token':'media-test'})).read()
        response=opener.open(urllib.request.Request(base+path,headers={'Range':'bytes=0-31'}))
        assert response.status==206 and len(response.read())==32 and response.headers['Accept-Ranges']=='bytes'
        with pytest.raises(urllib.error.HTTPError) as error:opener.open(urllib.request.Request(base+path,headers={'Origin':'https://evil.invalid'}))
        assert error.value.code==403
        for asset in ('studio.js','studio.css'):assert urllib.request.urlopen(base+'/'+asset).status==200
    finally:server.shutdown();server.server_close();manager.close();thread.join()

def test_resume_refuses_training_code_mismatch(tmp_path,cfg):
    data=tmp_path/'data';create_fixture(data,cfg,steps=12,sessions=3)
    kwargs=dict(config=cfg,epochs=2,sequence_steps=6,burn_in=1,batch_size=2,allow_synthetic=True)
    with pytest.raises(TrainingCancelled):train(data,tmp_path/'partial',max_updates=1,**kwargs)
    p=tmp_path/'partial/last.pt';state=torch.load(p,weights_only=True);state['training_code']['model.py']='changed';torch.save(state,p)
    with pytest.raises(ValueError,match='Training code changed'):train(data,tmp_path/'resume',resume_checkpoint=p,**kwargs)

def test_new_job_specs_do_not_require_an_unrelated_dataset(tmp_path):
    from camix_director.annotations import AnnotationStore
    store=AnnotationStore(tmp_path);project=store.create(media_fixture(tmp_path))
    cp=tmp_path/'runs/a/best.pt';cp.parent.mkdir(parents=True);cp.write_bytes(b'not loaded in spec validation')
    evaluation=tmp_path/'runs/a/evaluation.json';evaluation.write_text('{}')
    manager=JobManager(tmp_path,system={'devices':[{'id':'cpu','available':True}],'coreml_export_available':True})
    cases=[({'kind':'raw_demo','ack_synthetic':True},'camix_director.raw_demo'),({'kind':'proxy','project_id':project['id'],'camera_index':0},'camix_director.media_proxy'),({'kind':'gate','checkpoint':'runs/a/best.pt','evaluation':'runs/a/evaluation.json'},'camix_director.gates')]
    for index,(payload,module) in enumerate(cases):
        directory=tmp_path/f'job-{index}';directory.mkdir()
        spec=manager._spec(payload,directory)
        assert spec['module']==module and '--data' not in spec['args']
