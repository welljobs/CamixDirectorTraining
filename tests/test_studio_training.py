import json
from pathlib import Path
import numpy as np
import pytest
import torch
from camix_director.config import ModelConfig
from camix_director.synthetic import create_fixture
from camix_director.train import train,load_checkpoint,TrainingCancelled
from camix_director.data import SessionDataset
from camix_director.storage import MappedPack

@pytest.fixture
def cfg():return ModelConfig(max_cameras=2,image_size=16,audio_samples=256,hidden_dim=16,layers=1,memory_tokens=4,dropout=.1)

def arguments(cfg):return dict(config=cfg,epochs=3,sequence_steps=6,burn_in=1,batch_size=2,allow_synthetic=True,checkpoint_steps=1)

def test_media_arrays_are_mmap_not_in_memory(tmp_path,cfg):
    create_fixture(tmp_path,cfg,steps=12,sessions=3)
    dataset=SessionDataset(tmp_path,'train',cfg,6,1,True)
    assert isinstance(dataset.packs[0],MappedPack)
    assert isinstance(dataset.packs[0]['audio'],np.memmap)
    sample=dataset[0];assert sample['video_frames'].shape==(6,2,3,16,16)

def test_full_resume_matches_uninterrupted_weights(tmp_path,cfg):
    data=tmp_path/'data';create_fixture(data,cfg,steps=12,sessions=3)
    kw=arguments(cfg)
    path,_=train(data,tmp_path/'complete',**kw)
    with pytest.raises(TrainingCancelled):train(data,tmp_path/'partial',max_updates=2,**kw)
    checkpoint=tmp_path/'partial/last.pt'
    state=torch.load(checkpoint,weights_only=True)
    assert 'optimizer_state' in state and 'rng' in state and state['position']['next_batch']>0
    resumed,_=train(data,tmp_path/'resumed',resume_checkpoint=checkpoint,**kw)
    expected=torch.load(tmp_path/'complete/last.pt',weights_only=True)
    actual=torch.load(tmp_path/'resumed/last.pt',weights_only=True)
    for name,value in expected['model_state'].items():torch.testing.assert_close(value,actual['model_state'][name],rtol=0,atol=0)
    assert expected['history']==actual['history']

def test_resume_rejects_data_and_setting_changes(tmp_path,cfg):
    data=tmp_path/'data';create_fixture(data,cfg,steps=12,sessions=3)
    kw=arguments(cfg)
    with pytest.raises(TrainingCancelled):train(data,tmp_path/'partial',max_updates=1,**kw)
    with pytest.raises(ValueError,match='batch_size'):train(data,tmp_path/'bad_setting',resume_checkpoint=tmp_path/'partial/last.pt',**{**kw,'batch_size':1})
    p=data/'train_0.npz'
    with np.load(p,allow_pickle=False) as z:values={k:z[k] for k in z.files}
    values['audio'][0,0,0]+=.001;np.savez_compressed(p,**values)
    with pytest.raises(ValueError,match='Dataset'):train(data,tmp_path/'bad_data',resume_checkpoint=tmp_path/'partial/last.pt',**kw)

def test_weight_initialization_and_early_stopping(tmp_path,cfg):
    data=tmp_path/'data';create_fixture(data,cfg,steps=12,sessions=3)
    initial,_=train(data,tmp_path/'first',**{**arguments(cfg),'epochs':1})
    new,report=train(data,tmp_path/'fine',init_checkpoint=initial,**{**arguments(cfg),'epochs':5,'patience':1,'min_delta':100.})
    assert report['status']=='early_stopped' and len(report['history'])==2
    manifest=json.loads((tmp_path/'fine/run-manifest.json').read_text())
    assert manifest['parent']['mode']=='fine_tune' and manifest['environment']['torch']
    with pytest.raises(ValueError,match='weights only'):train(data,tmp_path/'bad_resume',resume_checkpoint=initial,**arguments(cfg))

def test_gradient_accumulation_checkpoint_boundary(tmp_path,cfg):
    data=tmp_path/'data';create_fixture(data,cfg,steps=12,sessions=3)
    kw={**arguments(cfg),'accumulation_steps':2}
    with pytest.raises(TrainingCancelled):train(data,tmp_path/'partial',max_updates=1,**kw)
    state=torch.load(tmp_path/'partial/last.pt',weights_only=True)
    assert state['position']['next_batch']==2
    train(data,tmp_path/'resumed',resume_checkpoint=tmp_path/'partial/last.pt',**kw)

def test_saved_best_state_does_not_alias_latest_mutable_weights(tmp_path,cfg):
    data=tmp_path/'data';create_fixture(data,cfg,steps=12,sessions=3)
    train(data,tmp_path/'run',**arguments(cfg))
    state=torch.load(tmp_path/'run/last.pt',weights_only=True)
    name=next(k for k,v in state['model_state'].items() if v.numel()>10)
    assert state['model_state'][name].data_ptr()!=state['best_state']['model_state'][name].data_ptr()

def test_resume_during_validation_restarts_validation_without_changing_result(tmp_path,cfg,monkeypatch):
    import camix_director.engine as engine
    data=tmp_path/'data';create_fixture(data,cfg,steps=12,sessions=3);kwargs=arguments(cfg)
    train(data,tmp_path/'full',**kwargs)
    stop=tmp_path/'stop';original=engine.atomic_json
    def observe(path,payload):
        original(path,payload)
        if Path(path).name=='progress.json' and payload.get('phase')=='validation' and payload.get('batch')==1:stop.touch()
    monkeypatch.setattr(engine,'atomic_json',observe)
    with pytest.raises(TrainingCancelled):train(data,tmp_path/'partial',stop_file=stop,progress_file=tmp_path/'progress.json',**kwargs)
    monkeypatch.setattr(engine,'atomic_json',original)
    train(data,tmp_path/'resume',resume_checkpoint=tmp_path/'partial/last.pt',**kwargs)
    a=torch.load(tmp_path/'full/last.pt',weights_only=True);b=torch.load(tmp_path/'resume/last.pt',weights_only=True)
    assert a['history']==b['history']
    for key in a['model_state']:torch.testing.assert_close(a['model_state'][key],b['model_state'][key],rtol=0,atol=0)


def test_completed_early_stop_does_not_silently_run_more_updates(tmp_path,cfg):
    data=tmp_path/'data';create_fixture(data,cfg,steps=12,sessions=3);kwargs={**arguments(cfg),'epochs':5,'patience':1,'min_delta':100.}
    train(data,tmp_path/'first',**kwargs)
    last=torch.load(tmp_path/'first/last.pt',weights_only=True)
    _,report=train(data,tmp_path/'restore',resume_checkpoint=tmp_path/'first/last.pt',**kwargs)
    assert report['status']=='early_stopped' and len(report['history'])==len(last['history'])

def test_one_prefetch_worker_uses_disk_backed_sequences(tmp_path,cfg):
    data=tmp_path/'data';create_fixture(data,cfg,steps=12,sessions=3)
    _,report=train(data,tmp_path/'worker',**{**arguments(cfg),'epochs':1,'num_workers':1})
    assert len(report['history'])==1
