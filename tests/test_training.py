import json
import numpy as np
import pytest
import torch
from camix_director.config import ModelConfig
from camix_director.synthetic import create_fixture
from camix_director.data import SessionDataset,read_pack,validate_split_separation
from camix_director.train import train,load_checkpoint,preference_loss
from camix_director.stream import StreamingPolicy
from camix_director.export import example_inputs

@pytest.fixture
def config():
    return ModelConfig(max_cameras=2,image_size=16,audio_samples=256,hidden_dim=16,layers=1,memory_tokens=4,dropout=0)

def test_synthetic_is_not_silently_real(tmp_path,config):
    create_fixture(tmp_path,config,steps=12,sessions=3)
    with pytest.raises(ValueError,match='Synthetic'):SessionDataset(tmp_path,'train',config,6,1)
    data=SessionDataset(tmp_path,'train',config,6,1,allow_synthetic=True)
    assert data[0]['video_frames'].shape==(6,2,3,16,16)

def test_no_unlabelled_implicit_keep():
    value=torch.zeros(1,3,requires_grad=True)
    loss,pairs,_=preference_loss(value,torch.full_like(value,float('nan')),torch.ones(1,2),torch.tensor([[1.,0.]]))
    assert pairs==0 and torch.isfinite(loss)

def test_weak_labels_require_opt_in(tmp_path,config):
    create_fixture(tmp_path,config,steps=12,sessions=3)
    p=tmp_path/'train_0.npz'
    with np.load(p,allow_pickle=False) as z:payload={k:z[k] for k in z.files}
    m=json.loads(str(payload['metadata'].item()));m.update(source='real',supervision='weak_test',human_reviewed=False,alignment_verified=True)
    payload['metadata']=np.array(json.dumps(m));np.savez_compressed(p,**payload)
    with pytest.raises(ValueError,match='reviewed'):read_pack(p,config)
    _,meta=read_pack(p,config,allow_weak=True);assert not meta['human_reviewed']

def test_session_separation(tmp_path,config):
    create_fixture(tmp_path,config,steps=12,sessions=3)
    trainset=SessionDataset(tmp_path,'train',config,6,1,True)
    valset=SessionDataset(tmp_path,'validation',config,6,1,True)
    validate_split_separation(trainset,valset)
    valset.metadata[0]['session_id']=trainset.metadata[0]['session_id']
    with pytest.raises(ValueError,match='leakage'):validate_split_separation(trainset,valset)

def test_train_checkpoint_and_stream_state(tmp_path,config):
    create_fixture(tmp_path/'data',config,steps=12,sessions=3)
    path,report=train(tmp_path/'data',tmp_path/'run',config,epochs=2,sequence_steps=6,burn_in=1,batch_size=2,allow_synthetic=True)
    model,checkpoint=load_checkpoint(path)
    assert checkpoint['qualification']=='synthetic_smoke_only' and not checkpoint['production_approved']
    assert len(report['history'])==2
    stream=StreamingPolicy(path)
    features=dict(zip(('video_frames','audio_pcm','modality_mask','camera_mask','media_age','current_program','held_seconds'),example_inputs(config)[:-2]))
    result=stream.step(features,1.,'room-a');assert not result['execute_allowed']
    with pytest.raises(ValueError,match='timestamp'):stream.step(features,1.,'room-a')
    with pytest.raises(ValueError,match='timestamp'):stream.step(features,float('nan'),'room-a')
    stream.step(features,0.,'room-b');assert stream.generation=='room-b'

def test_existing_checkpoint_is_not_overwritten(tmp_path,config):
    out=tmp_path/'run';out.mkdir();(out/'best.pt').write_bytes(b'keep me')
    with pytest.raises(ValueError,match='fresh output'):
        train(tmp_path/'missing-data',out,config)
    assert (out/'best.pt').read_bytes()==b'keep me'

def test_multiple_episodes_of_one_event_stay_in_one_split(tmp_path,config):
    import shutil
    create_fixture(tmp_path,config,steps=12,sessions=3)
    shutil.copyfile(tmp_path/'train_0.npz',tmp_path/'train_episode2.npz')
    data=SessionDataset(tmp_path,'train',config,6,1,allow_synthetic=True)
    assert len(data.metadata)==4
    assert sum(m["session_id"]=="SYNTHETIC-train-0" for m in data.metadata)==2
    val=SessionDataset(tmp_path,'validation',config,6,1,allow_synthetic=True)
    validate_split_separation(data,val)

def test_importance_weighted_pairs_do_not_cancel_per_step():
    scores=torch.zeros(1,3)
    _,count,_=preference_loss(scores,torch.tensor([[1.,1.,0.]]),torch.ones(1,2),torch.tensor([[1.,0.]]))
    plain,_,_=preference_loss(scores,torch.tensor([[1.,1.,0.]]),torch.ones(1,2),torch.tensor([[1.,0.]]))
    weighted,_,_=preference_loss(scores,torch.tensor([[1.,1.,0.]]),torch.ones(1,2),torch.tensor([[1.,0.]]),torch.tensor([3.,.5,.5]))
    assert count==1 and torch.allclose(weighted,plain*3)

def test_pair_diagnostics_separates_keep_and_take():
    from camix_director.evaluate import pair_diagnostics
    current=torch.tensor([[1.,0.],[1.,0.]])
    labels=torch.tensor([[1.,1.,0.],[0.,0.,1.]])
    values=torch.tensor([[1.,1.,0.],[1.,1.,0.]])
    assert pair_diagnostics(values,labels,torch.ones(2,2),current).tolist()==[1,1,1,0]
