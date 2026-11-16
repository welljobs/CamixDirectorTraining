from copy import deepcopy
import numpy as np
import pytest
import torch
from camix_director.contracts_v2 import INTENTS
from camix_director.corpus_v2 import corpus_identity, load_corpus
from camix_director.data_v2 import EventFeatureSequences
from camix_director.losses_v2 import policy_losses
from camix_director.model_v2 import FeatureDirector, PolicyConfig, action_log_probs
from camix_director.policy_io_v2 import load_policy
from camix_director.policy_runtime_v2 import CausalSignals, tensor_targets, tensors_for_step
from camix_director.train_v2 import train
from camix_director.training_state import TrainingCancelled
from stages_v2_helpers import stage_fixture


@pytest.fixture(scope='module')
def stage(tmp_path_factory):
    return stage_fixture(tmp_path_factory.mktemp('policy-v2'))


def inputs(config):
    n = config.max_cameras
    return [torch.rand(1,n,config.visual_dim),torch.rand(1,n),torch.ones(1,n),torch.ones(1,n),
            torch.zeros(1,n),torch.zeros(1,n),torch.tensor([[1.]+[0.]*(n-1)]),torch.tensor([[2.]]),
            torch.zeros(1,1),torch.ones(1,1),torch.zeros(1,1),torch.zeros(1,3)]


def test_policy_shapes_permutation_and_all_invalid():
    cfg = PolicyConfig(visual_dim=8,max_cameras=3,hidden_dim=16,layers=1,short_tokens=4,shot_tokens=3,slow_tokens=2,dropout=0)
    model = FeatureDirector(cfg).eval();x=inputs(cfg);state=model.initial_state()
    expected=model(*x,*state)
    assert len(expected)==10 and expected[1].shape==(1,3) and expected[2].shape==(1,len(INTENTS))
    permutation=torch.tensor([2,0,1]);y=list(x)
    for i in range(7):y[i]=x[i].index_select(1,permutation)
    actual=model(*y,*state)
    torch.testing.assert_close(expected[0],actual[0],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(expected[1].index_select(1,permutation),actual[1],rtol=1e-5,atol=1e-6)
    x[2].zero_();x[3].zero_()
    outputs=model(*x,*state);logp,legal=action_log_probs(outputs[0],outputs[1],x[2],x[6])
    assert all(torch.isfinite(v).all() for v in outputs) and torch.isfinite(logp).all() and not legal.any()


def test_memories_update_only_from_committed_cut_and_slow_signal():
    cfg=PolicyConfig(visual_dim=8,max_cameras=2,hidden_dim=16,layers=1,short_tokens=4,shot_tokens=3,slow_tokens=2,dropout=0)
    model=FeatureDirector(cfg).eval();x=inputs(cfg);state=model.initial_state()
    output=model(*x,*state)
    assert output[5].sum()==1 and output[7].sum()==0 and output[9].sum()==0
    x[8].fill_(1);x[10].fill_(1)
    updated=model(*x,*output[4:])
    assert updated[7].sum()==1 and updated[9].sum()==1
    assert updated[4].shape==state[0].shape


def test_future_features_do_not_change_prior_state():
    cfg=PolicyConfig(visual_dim=8,max_cameras=2,hidden_dim=16,layers=1,short_tokens=3,shot_tokens=2,slow_tokens=2,dropout=0)
    model=FeatureDirector(cfg).eval();x=inputs(cfg);state=model.initial_state()
    a=model(*x,*state);saved=[t.clone() for t in a]
    x[0].fill_(100);model(*x,*a[4:])
    for before,after in zip(saved,a):torch.testing.assert_close(before,after,rtol=0,atol=0)


def test_cut_windows_do_not_invent_keep_or_force_switch_when_already_acceptable(stage):
    from camix_director.data_v2 import policy_targets
    root,corpus,cfg,projects=stage;m=projects[0]['manifest'];row=deepcopy(m['shot_preferences'][1]);row['cut_window_s']=[2.,3.]
    value=policy_targets(m,1.,'cam_a',np.ones(2),[row],[])
    assert not value['action_annotated'] and not value['switch_mask']
    value=policy_targets(m,2.,'cam_a',np.ones(2),[row],[])
    assert value['action_annotated'] and value['switch_target']==1
    row['acceptable_cameras']=['cam_a','cam_b']
    assert not policy_targets(m,2.,'cam_a',np.ones(2),[row],[])['switch_mask']


def test_reviewed_targets_are_separate_from_policy_observations(stage):
    root,corpus,cfg,_=stage;event=load_corpus(root,corpus,True)[0]
    sample=next(iter(EventFeatureSequences(root,event.index,event.cache_id)))
    x,observed=tensors_for_step(sample['observations'],0,cfg,'cpu',CausalSignals(),event.manifest['clock'])
    assert not {'intent_distribution','visibility_target','shot_preferences'} & set(observed)
    target=tensor_targets(sample['targets'],0,cfg,'cpu');model=FeatureDirector(cfg)
    loss,metrics=policy_losses(model(*x,*model.initial_state()),observed,target)
    assert loss is not None and torch.isfinite(loss) and metrics['action_count']==1
    loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_set_valued_loss_all_unknown_has_no_implicit_keep():
    cfg=PolicyConfig(visual_dim=8,max_cameras=2,hidden_dim=16,layers=1,short_tokens=3,shot_tokens=2,slow_tokens=2)
    model=FeatureDirector(cfg);x=inputs(cfg);out=model(*x,*model.initial_state())
    from camix_director.model_v2 import OBS_NAMES
    obs=dict(zip(OBS_NAMES,x));target=dict(acceptable_cameras=torch.zeros(1,2),annotated=torch.zeros(1,dtype=torch.bool),switch_target=torch.zeros(1),switch_mask=torch.zeros(1,dtype=torch.bool),intent_distribution=torch.zeros(1,len(INTENTS)),intent_mask=torch.zeros(1,dtype=torch.bool),visibility_target=torch.zeros(1,2),visibility_mask=torch.zeros(1,2))
    loss,metrics=policy_losses(out,obs,target)
    assert loss is None and metrics['action_count']==0


def test_train_policy_checkpoint_and_independent_validation(stage):
    root,corpus,cfg,_=stage
    cp,report=train(root,corpus,root/'runs/sft',cfg,epochs=2,allow_synthetic=True)
    model,payload=load_policy(cp)
    assert report['status']=='completed' and len(report['history'])==2 and payload['qualification']=='synthetic_smoke_only'
    assert payload['feature_space_id'] and not payload['production_approved']
    assert all(p['split']!='test' for p in payload['seen_data'])
    assert report['history'][-1]['train']['action_count']>0


def test_epoch_boundary_resume_matches_uninterrupted_weights(stage):
    root,corpus,cfg,_=stage
    expected,_=train(root,corpus,root/'runs/complete',cfg,epochs=2,allow_synthetic=True)
    with pytest.raises(TrainingCancelled):train(root,corpus,root/'runs/partial',cfg,epochs=2,allow_synthetic=True,max_epochs_this_call=1)
    actual,_=train(root,corpus,root/'runs/resumed',cfg,epochs=2,allow_synthetic=True,resume_checkpoint=root/'runs/partial/last.pt')
    a=torch.load(root/'runs/complete/last.pt',weights_only=True);b=torch.load(root/'runs/resumed/last.pt',weights_only=True)
    assert a['history']==b['history']
    for name,t in a['model_state'].items():torch.testing.assert_close(t,b['model_state'][name],rtol=0,atol=0)


def test_policy_refuses_synthetic_or_preview_bypass(stage):
    root,corpus,cfg,_=stage
    with pytest.raises(ValueError,match='synthetic'):train(root,corpus,root/'runs/no-optin',cfg,epochs=1)
    with pytest.raises(ValueError,match='booleans'):load_corpus(root,corpus,'false')
    with pytest.raises(ValueError,match='Duplicate'):load_corpus(root,{**corpus,'entries':corpus['entries']+[corpus['entries'][0]]},True)


def test_causal_teacher_student_distillation_is_an_executable_training_mode(stage):
    from dataclasses import replace
    root,corpus,cfg,_=stage
    teacher,_=train(root,corpus,root/'runs/teacher',cfg,epochs=1,allow_synthetic=True)
    student=replace(cfg,hidden_dim=8,heads=2,short_tokens=2)
    cp,report=train(root,corpus,root/'runs/student',student,epochs=1,allow_synthetic=True,mode='distill',teacher_checkpoint=teacher)
    _,p=load_policy(cp)
    assert p['config']['hidden_dim']==8 and p['settings']['teacher_sha256']
    assert report['history'][0]['train']['totals']['distillation']['count']>0


def test_style_axes_are_real_conditions_not_a_fixed_scene_classifier():
    torch.manual_seed(7)
    cfg=PolicyConfig(visual_dim=8,max_cameras=2,hidden_dim=16,layers=1,short_tokens=3,shot_tokens=2,slow_tokens=2,dropout=0)
    model=FeatureDirector(cfg).eval();x=inputs(cfg);state=model.initial_state()
    a=model(*x,*state);x[11]=torch.tensor([[.8,-.5,.2]])
    b=model(*x,*state)
    assert not torch.allclose(a[0],b[0]) and not torch.allclose(a[2],b[2])


def test_same_split_renamed_source_cannot_inflate_independent_events(stage):
    from camix_director.annotations_v2 import EventStore
    from camix_director.prepare_v2 import prepare_index
    from camix_director.features_v2 import extract
    from camix_director.storage import digest_file
    root,corpus,cfg,projects=stage
    m=deepcopy(projects[0]['manifest']);m['original_event_id']+='-renamed';m['comparisons']=[]
    store=EventStore(root);p=store.create(m);p=store.review(p['id'],p['revision'],'synthetic alias test',True,True)
    path=root/'runs/alias/index.json';index=prepare_index(root,p,path,decision_hz=2,chunk_steps=3,purpose='training',allow_synthetic=True)
    weights=root/'runs/test-encoder/best.pt';cache=extract(root,index,'v1_visual',str(weights),digest_file(weights))
    renamed=dict(index=path.relative_to(root).as_posix(),cache_id=cache['cache_id'])
    with pytest.raises(ValueError,match='different event IDs'):
        load_corpus(root,{**corpus,'entries':corpus['entries']+[renamed]},True)
