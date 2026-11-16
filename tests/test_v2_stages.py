from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
import torch

from camix_director.corpus_v2 import feature_space_id,load_corpus
from camix_director.corrections_v2 import create_correction
from camix_director.evaluate_v2 import evaluate
from camix_director.export_v2 import export_policy
from camix_director.gates_v2 import assess
from camix_director.policy_io_v2 import load_policy
from camix_director.policy_replay_v2 import feature_packet,replay_policy
from camix_director.policy_runtime_v2 import StreamingDirector
from camix_director.preference_v2 import trajectory_log_likelihood
from camix_director.quality_v2 import comparison_loss,train_quality,evaluate_quality,clip_descriptor
from camix_director.shadow_v2 import packets_from_event,run_shadow
from camix_director.train_v2 import train
from camix_director.features_v2 import iter_features
from stages_v2_helpers import stage_fixture


@pytest.fixture(scope='module')
def stage(tmp_path_factory):
    root,corpus,cfg,projects=stage_fixture(tmp_path_factory.mktemp('stages-v2'))
    cp,_=train(root,corpus,root/'runs/v2/sft',cfg,epochs=2,allow_synthetic=True)
    events=load_corpus(root,corpus,True)
    return root,corpus,cfg,projects,cp,events


def test_reference_relative_preference_training_is_executable(stage):
    root,corpus,cfg,_,cp,_=stage
    output,report=train(root,corpus,root/'runs/v2/preference',cfg,epochs=1,allow_synthetic=True,mode='preference',teacher_checkpoint=cp)
    _,payload=load_policy(output)
    assert report['history'][0]['train']['comparisons']==1
    assert payload['settings']['mode']=='preference' and payload['settings']['teacher_sha256']
    assert not payload['production_approved'] and payload['qualification']=='synthetic_smoke_only'


def test_variants_have_their_own_pgm_and_are_differentiable(stage):
    root,_,_,_,cp,events=stage;model,_=load_policy(cp)
    event=events[0];row=event.manifest['comparisons'][0]
    captures=[];original=model.forward
    def capture(*args):captures.append(args[6].detach().clone());return original(*args)
    model.forward=capture
    a=trajectory_log_likelihood(root,model,event,row,row['a']);left=list(captures);captures.clear()
    b=trajectory_log_likelihood(root,model,event,row,row['b']);right=list(captures)
    assert any(not torch.equal(x,y) for x,y in zip(left,right))
    loss=torch.nn.functional.softplus(a-b);loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_offline_quality_ties_uncertainty_and_independent_test(stage):
    root,corpus,_,_,_,events=stage
    assert comparison_loss(torch.tensor([1.]),torch.tensor([1.]),'tie')==0
    assert comparison_loss(torch.tensor([1.]),torch.tensor([0.]),'uncertain') is None
    assert comparison_loss(torch.tensor([1.]),torch.tensor([0.]),'a')<comparison_loss(torch.tensor([0.]),torch.tensor([1.]),'a')
    cp,report=train_quality(root,corpus,root/'runs/v2/quality',epochs=1,allow_synthetic=True)
    result=evaluate_quality(root,cp,corpus,root/'runs/v2/quality-test.json',allow_synthetic=True)
    assert result['summary']['ranked_comparisons']==1 and not result['professional_evidence']
    with pytest.raises(ValueError,match='seen'):evaluate_quality(root,cp,corpus,root/'runs/v2/not-independent.json',split='train',allow_synthetic=True)
    row=events[0].manifest['comparisons'][0]
    descriptor=clip_descriptor(root,events[0],row,row['a'])
    assert descriptor.shape[-1]==events[0].header['feature_dim']*2+9


def test_evaluation_uses_independent_events_ablations_and_matched_replays(stage):
    root,corpus,_,_,cp,_=stage
    report=evaluate(root,cp,corpus,root/'runs/v2/evaluation',allow_synthetic=True,min_hold_s=1.)
    assert report['independent'] and report['original_events']==1
    assert report['event_bootstrap_95_interval'] is None and len(report['teacher_context'])==4
    assert {r['policy'] for r in report['closed_loop']}=={'learned','hold_valid','audio_rms'}
    assert all(not r['execute_allowed'] and r['report_sha256'] for r in report['closed_loop'])
    with pytest.raises(ValueError,match='used'):evaluate(root,cp,corpus,root/'runs/v2/seen',split='train',allow_synthetic=True)


def test_learned_replay_corrections_are_unreviewed_and_drop_relative_labels(stage):
    root,_,_,_,cp,events=stage;event=events[-1]
    out=root/'runs/v2/correction-source'
    report=replay_policy(root,cp,event.index,event.cache_id,out,min_hold_s=1.)
    correction=create_correction(root,event.index,out/'replay-report.json');p=correction['project']
    assert p['review'] is None and p['manifest']['shot_preferences']==[] and p['manifest']['comparisons']==[]
    assert p['manifest']['trajectory_origin']['kind']=='offline_policy'
    assert p['manifest']['original_event_id']==event.manifest['original_event_id'] and p['manifest']['source']=='synthetic'
    assert not correction['execute_allowed']
    with (out/'value-trace.jsonl').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='changed'):create_correction(root,event.index,out/'replay-report.json')


def packet_fixture(stage):
    root,_,_,_,cp,events=stage;event=events[-1]
    arrays=next(iter_features(root,event.cache_id))[1]
    return cp,feature_packet(event.index,event.header,arrays,0,'cam_a',0.,'room-generation-one'),arrays,event


def test_streaming_rejects_wrong_encoder_future_and_unannounced_slots(stage):
    cp,packet,arrays,event=packet_fixture(stage)
    actor=StreamingDirector(cp)
    result=actor.step(packet)
    assert not result['execute_allowed'] and not result['probability_calibrated']
    with pytest.raises(ValueError,match='Duplicate'):actor.step(packet)
    bad=deepcopy(packet);bad['event_time_s']=.5;bad['feature_space_id']='0'*64
    with pytest.raises(ValueError,match='feature-space'):actor.step(bad)
    bad=deepcopy(packet);bad['event_time_s']=.5;bad['video_available_s'][0]=1.
    with pytest.raises(ValueError,match='Future'):actor.step(bad)
    bad=deepcopy(packet);bad['event_time_s']=.5;bad['camera_ids'].reverse()
    with pytest.raises(ValueError,match='slots'):actor.step(bad)
    bad=deepcopy(packet);bad['event_time_s']=.5;bad['style_features']=[.5,0,0]
    with pytest.raises(ValueError,match='Style'):actor.step(bad)
    bad['generation']='new-style-generation'
    assert actor.step(bad)['reset_reason']=='generation'


def test_streaming_all_invalid_abstains_and_gap_resets_state(stage):
    cp,packet,_,_=packet_fixture(stage);actor=StreamingDirector(cp,max_gap_s=2.)
    for key in ('video_mask','audio_mask','audio_rms'):packet[key]=[0.,0.]
    packet['visual']=[[0.]*actor.model.config.visual_dim for _ in range(2)]
    assert actor.step(packet)['selected_camera'] is None
    packet['event_time_s']=3.
    assert actor.step(packet)['reset_reason']=='gap'
    assert actor.step({**packet,'generation':'fresh','event_time_s':0.})['execute_allowed'] is False


def test_policy_step_export_has_independent_rolling_state_parity(stage):
    root,_,_,_,cp,_=stage
    report=export_policy(cp,root/'runs/v2/export',parity_steps=6)
    assert report['parity_passed'] and report['torchscript_parity_passed']
    assert report['coreml_parity_passed'] is None and not report['production_approved']
    assert (root/'runs/v2/export/FeatureDirectorStep.pt').is_file()
    signature=json.loads((root/'runs/v2/export/signature.json').read_text())
    assert signature['requires_external_encoder_and_causal_adapter'] and 'style_features' in signature['inputs']


def test_recorded_shadow_and_gates_never_grant_live_execution(stage):
    root,corpus,_,_,cp,events=stage;event=events[-1]
    path=root/'data/shadow-input.jsonl';packets_from_event(root,event.index,event.cache_id,path)
    shadow=run_shadow(root,cp,path,root/'runs/v2/shadow')
    assert shadow['packets']==event.index['ticks'] and not shadow['live_rtc_verified']
    assert not shadow['execute_allowed']
    if not (root/'runs/v2/gate-eval/evaluation.json').exists():evaluate(root,cp,corpus,root/'runs/v2/gate-eval',allow_synthetic=True)
    if not (root/'runs/v2/export/export-report.json').exists():export_policy(cp,root/'runs/v2/export',parity_steps=4)
    if not (root/'runs/v2/quality-test.json').exists():
        quality,_=train_quality(root,corpus,root/'runs/v2/gate-quality',epochs=1,allow_synthetic=True)
        evaluate_quality(root,quality,corpus,root/'runs/v2/quality-test.json',allow_synthetic=True)
    result=assess(root,cp,root/'runs/v2/gate-eval/evaluation.json',root/'runs/v2/gate-report.json',
                  root/'runs/v2/export/export-report.json',root/'runs/v2/shadow/shadow-report.json',root/'runs/v2/quality-test.json')
    assert result['status']=='blocked' and not result['production_approved']
    assert any(c['name']=='real_reviewed_data' and not c['passed'] for c in result['checks'])


def test_shadow_json_rejects_nonfinite_and_duplicate_keys():
    from camix_director.shadow_v2 import _strict_line
    for raw in ('{"a":1,"a":2}','{"a":NaN}'):
        with pytest.raises(ValueError):_strict_line(raw)


def test_policy_and_quality_checkpoints_are_not_interchangeable(stage):
    from camix_director.quality_v2 import load_quality
    root,corpus,_,_,cp,_=stage
    if not (root/'runs/v2/quality/quality-best.pt').exists():train_quality(root,corpus,root/'runs/v2/quality',epochs=1,allow_synthetic=True)
    with pytest.raises(ValueError,match='quality'):load_quality(cp)
    with pytest.raises(ValueError,match='policy'):load_policy(root/'runs/v2/quality/quality-best.pt')
