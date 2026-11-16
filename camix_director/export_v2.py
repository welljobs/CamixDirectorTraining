"""Feature-policy step export and independent rolling-state parity, not device acceptance."""
from pathlib import Path
import json
import platform
import numpy as np
import torch

from .model_v2 import INPUT_NAMES,OUTPUT_NAMES,OBS_NAMES
from .memory_v2 import initial_state
from .policy_io_v2 import checkpoint_hash,load_policy
from .runtime import atomic_json


def example_inputs(config,seed=1):
    g=torch.Generator().manual_seed(seed);n=config.max_cameras;d=config.visual_dim
    observed=[torch.randn(1,n,d,generator=g),torch.rand(1,n,generator=g)*.3,torch.ones(1,n),torch.ones(1,n),
              torch.rand(1,n,generator=g)*5,torch.rand(1,n,generator=g)*5,torch.nn.functional.one_hot(torch.tensor([seed%n]),n).float(),
              torch.ones(1,1)*(seed%13+.1),torch.zeros(1,1),torch.ones(1,1)*(seed%9+.2),torch.zeros(1,1),torch.rand(1,3,generator=g)*2-1]
    return tuple(observed)+initial_state(config)


def export_policy(checkpoint,output,coreml=False,parity_steps=12):
    if type(coreml) is not bool or isinstance(parity_steps,bool) or not isinstance(parity_steps,int) or not 4<=parity_steps<=100:
        raise ValueError('Export requires explicit Core ML flag and 4..100 parity steps')
    model,payload=load_policy(checkpoint);torch.set_num_threads(2)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Choose empty export output')
    examples=example_inputs(model.config)
    with torch.no_grad():traced=torch.jit.trace(model,examples,check_trace=True)
    traced.save(str(output/'FeatureDirectorStep.pt'))
    signature=dict(format='camix_feature_policy_signature_v2',config=model.config.to_dict(),feature_space=payload['feature_space'],
                   feature_space_id=payload['feature_space_id'],clock_profiles=payload['clock_profiles'],
                   inputs={name:list(t.shape) for name,t in zip(INPUT_NAMES,examples)},outputs=list(OUTPUT_NAMES),
                   input_type='encoded features; NOT raw RGB/PCM',requires_external_encoder_and_causal_adapter=True,
                   probability_calibrated=False,production_approved=False,execute_allowed=False)
    atomic_json(output/'signature.json',signature)
    mlmodel=None
    try:
        if coreml:
            if platform.system()!='Darwin':raise ValueError('Core ML conversion and predict parity require a Mac')
            import coremltools as ct
            mlmodel=ct.convert(traced,convert_to='mlprogram',
                inputs=[ct.TensorType(name=name,shape=tuple(t.shape),dtype=np.float32) for name,t in zip(INPUT_NAMES,examples)],
                outputs=[ct.TensorType(name=name,dtype=np.float32) for name in OUTPUT_NAMES],
                minimum_deployment_target=ct.target.iOS17,compute_precision=ct.precision.FLOAT32,compute_units=ct.ComputeUnit.CPU_ONLY)
            mlmodel.user_defined_metadata.update({'model_family':payload['model_family'],'feature_space_id':payload['feature_space_id'],
                                                  'production_approved':'false','shadow_only':'true','config_json':json.dumps(payload['config'])})
            mlmodel.save(str(output/'FeatureDirectorStep.mlpackage'))
        eager_state=model.initial_state();script_state=tuple(t.clone() for t in eager_state)
        core_state=tuple(t.numpy().copy() for t in eager_state);maximum_script=0.;maximum_core=0.;records=[]
        for step in range(parity_steps):
            values=list(example_inputs(model.config,step+50))
            values[8].fill_(float(step%3==1));values[10].fill_(float(step%4==0))
            if step%5==1:values[3][:,-1]=0;values[1][:,-1]=0
            if step%5==2:values[2][:,-1]=0;values[0][:,-1]=0
            if step%5==3:values[2].zero_();values[3].zero_();values[0].zero_();values[1].zero_()
            values[4]=torch.where(values[2]>.5,values[4],torch.full_like(values[4],30))
            values[5]=torch.where(values[3]>.5,values[5],torch.full_like(values[5],30))
            with torch.no_grad():
                expected=model(*values[:len(OBS_NAMES)],*eager_state);actual=traced(*values[:len(OBS_NAMES)],*script_state)
            ds=max(float(torch.max(torch.abs(a-b))) for a,b in zip(expected,actual));maximum_script=max(maximum_script,ds)
            row=dict(step=step,torchscript_max_absolute_difference=ds)
            eager_state=expected[4:];script_state=actual[4:]
            if mlmodel:
                feed={name:t.numpy() for name,t in zip(INPUT_NAMES[:len(OBS_NAMES)],values[:len(OBS_NAMES)])};feed.update(dict(zip(INPUT_NAMES[len(OBS_NAMES):],core_state)))
                predicted=mlmodel.predict(feed)
                dc=max(float(np.max(np.abs(t.numpy()-np.asarray(predicted[name])))) for name,t in zip(OUTPUT_NAMES,expected))
                maximum_core=max(maximum_core,dc);core_state=tuple(np.asarray(predicted[name],np.float32).copy() for name in OUTPUT_NAMES[4:])
                row['coreml_max_absolute_difference']=dc
            records.append(row)
        passed=maximum_script<=.002 and (not coreml or maximum_core<=.002)
        report=dict(format='camix_policy_export_v2',checkpoint_sha256=checkpoint_hash(checkpoint),feature_space_id=payload['feature_space_id'],
                    torchscript_parity_passed=maximum_script<=.002,torchscript_max_absolute_difference=maximum_script,
                    coreml_requested=coreml,coreml_parity_passed=maximum_core<=.002 if coreml else None,
                    coreml_max_absolute_difference=maximum_core if coreml else None,parity_passed=passed,steps=records,tolerance=.002,
                    qualification=payload['qualification'],production_approved=False,execute_allowed=False,
                    parity_cases=['signed_visual_features','nonzero_style_conditions','changing_program_camera','varying_hold_and_completed_shot','short_shot_slow_state_updates','single_and_all_modality_missing'],
                    warning='Policy feature-step only. CPU conversion/independent memory parity is not raw-media deployment, iPad latency, Neural Engine, thermal stability or production certification.')
        atomic_json(output/'export-report.json',report)
        if not passed:raise ValueError('V2 export parity failed; do not deploy')
        return report
    except Exception as error:
        if not (output/'export-report.json').exists():atomic_json(output/'export-report.json',dict(format='camix_policy_export_v2',checkpoint_sha256=checkpoint_hash(checkpoint),parity_passed=False,coreml_requested=coreml,error=str(error),production_approved=False))
        raise
