import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import torch
from .config import INPUT_NAMES,OUTPUT_NAMES,SCHEMA_VERSION
from .train import load_checkpoint


def example_inputs(config,seed=9):
    generator=torch.Generator().manual_seed(seed);n=config.max_cameras
    memory=torch.randn(1,config.memory_tokens,config.hidden_dim,generator=generator)*.1
    return (torch.rand(1,n,3,config.image_size,config.image_size,generator=generator)*2-1,
            torch.rand(1,n,config.audio_samples,generator=generator)*.2-.1,
            torch.ones(1,n,2),torch.ones(1,n),torch.zeros(1,n,2),
            torch.nn.functional.one_hot(torch.zeros(1,dtype=torch.long),n).float(),
            torch.ones(1,1)*5,memory,torch.ones(1,config.memory_tokens))


def export(checkpoint_path,output,precision='fp32',parity_steps=8):
    if platform.system()!='Darwin':raise RuntimeError('Run Core ML conversion + numerical parity on a Mac')
    import coremltools as ct
    torch.set_num_threads(2)
    model,checkpoint=load_checkpoint(checkpoint_path);model.eval()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model_path=output/'ExperimentalDirectorStep.mlpackage'
    if model_path.exists():raise ValueError('Choose fresh export directory; existing model is not overwritten')
    example=example_inputs(model.config)
    with torch.no_grad():traced=torch.jit.trace(model,example,check_trace=True)
    mlmodel=ct.convert(traced,convert_to='mlprogram',
        inputs=[ct.TensorType(name=name,shape=tuple(value.shape),dtype=np.float32) for name,value in zip(INPUT_NAMES,example)],
        outputs=[ct.TensorType(name=name,dtype=np.float32) for name in OUTPUT_NAMES],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32 if precision=='fp32' else ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY)
    mlmodel.short_description='EXPERIMENT ONLY: raw audiovisual value model with bounded causal memory. NOT approved for live programme control.'
    mlmodel.user_defined_metadata.update({'schema_version':str(SCHEMA_VERSION),'qualification':checkpoint['qualification'],
        'production_approved':'false','shadow_only':'true','config_json':json.dumps(model.config.to_dict())})
    mlmodel.save(str(model_path))
    torch_memory,torch_mask=model.initial_memory();core_memory=torch_memory.numpy().copy();core_mask=torch_mask.numpy().copy()
    maximum=0.;records=[]
    for step in range(parity_steps):
        values=list(example_inputs(model.config,step+20))
        # Include missing audio, a disconnected camera and all-invalid input in parity checks.
        if step%4==1:values[2][:,-1,1]=0
        if step%4==2:values[2][:,-1,:]=0;values[3][:,-1]=0
        if step%4==3:values[2][:]=0;values[3][:]=0
        values[-2:]=[torch_memory,torch_mask]
        with torch.no_grad():expected=model(*values)
        payload={name:value.numpy().copy() for name,value in zip(INPUT_NAMES,values)}
        payload['memory']=core_memory;payload['memory_mask']=core_mask
        started=time.perf_counter();actual=mlmodel.predict(payload);elapsed=(time.perf_counter()-started)*1000
        differences={name:float(np.max(np.abs(value.numpy()-np.asarray(actual[name])))) for name,value in zip(OUTPUT_NAMES,expected)}
        maximum=max(maximum,max(differences.values()))
        records.append(dict(step=step,max_abs_differences=differences,mac_coreml_cpu_ms=elapsed))
        torch_memory,torch_mask=expected[-2:]
        core_memory=np.asarray(actual['next_memory'],np.float32).copy();core_mask=np.asarray(actual['next_memory_mask'],np.float32).copy()
    tolerance=.002 if precision=='fp32' else .1
    report=dict(qualification=checkpoint['qualification'],production_approved=False,shadow_only=True,precision=precision,
        parity_passed=maximum<=tolerance,maximum_absolute_difference=maximum,tolerance=tolerance,steps=records,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),config=model.config.to_dict(),
        input_names=list(INPUT_NAMES),output_names=list(OUTPUT_NAMES),
        warning='Conversion/numerical verification only. Not iPad performance, calibrated gains, director aesthetics or production qualification.')
    (output/'export-report.json').write_text(json.dumps(report,indent=2), encoding="utf-8")
    if not report['parity_passed']:raise RuntimeError(f'Core ML numerical parity failed: {maximum} > {tolerance}; do not deploy')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--precision',choices=['fp32','fp16'],default='fp32');p.add_argument('--parity-steps',type=int,default=8)
    a=p.parse_args();print(json.dumps(export(a.checkpoint,a.output,a.precision,a.parity_steps),indent=2))
