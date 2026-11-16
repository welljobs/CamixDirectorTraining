"""Executable P2-P5 research workflows. No production control or automatic downloads."""
import argparse
from pathlib import Path
import json

from .contracts_v2 import load_json,data_path
from .runtime import atomic_json
from .training_state import TrainingCancelled
from .v2 import artifact_path


def input_json(root,path):
    p=(Path(root)/path).resolve()
    if p.is_relative_to(Path(root).resolve()/'.web/jobs') and p.is_file():return load_json(p)
    return load_json(artifact_path(root,str(path)))


def smoke(root,output,data_output,encoder_weights,encoder_sha,coreml=False,stop_file=None):
    """Deterministic generated media/labels; AUTOMATED fixtures, never expert evidence."""
    import wave
    import numpy as np
    from .annotations_v2 import EventStore, import_v1_media
    from .contracts_v2 import media_identity
    from .corpus_v2 import save_corpus, load_corpus
    from .features_v2 import extract
    from .model_v2 import PolicyConfig
    from .prepare_v2 import prepare_index
    from .raw_demo import create
    from .train_v2 import train
    from .quality_v2 import train_quality,evaluate_quality
    from .evaluate_v2 import evaluate
    from .policy_replay_v2 import replay_policy
    from .corrections_v2 import create_correction
    from .export_v2 import export_policy
    from .shadow_v2 import packets_from_event,run_shadow
    from .gates_v2 import assess
    root=Path(root).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Choose empty smoke output')
    data_output=data_path(root,str(data_output),exists=False)
    store=EventStore(root);entries=[]
    for ordinal,path in enumerate(create(data_output,seconds=6)):
        if stop_file and Path(stop_file).exists():raise InterruptedError('Synthetic pipeline stopped')
        for audio in Path(path).parent.glob('*.wav'):
            with wave.open(str(audio),'rb') as f:params=f.getparams();values=np.frombuffer(f.readframes(f.getnframes()),'<i2').copy()
            values=(values.astype(np.float64)*(1-ordinal*.13)).astype('<i2')
            with wave.open(str(audio),'wb') as f:f.setparams(params);f.writeframes(values.tobytes())
        m=import_v1_media(root,path);m['original_event_id']+='-'+Path(data_output).name
        m['license']=dict(name='Locally generated synthetic ENGINEERING fixture, not expert evidence',usage_confirmed=True)
        m['program_events']=[dict(time_s=2.,camera_id='cam_b'),dict(time_s=4.,camera_id='cam_a')]
        m['shot_preferences']=[dict(id=f's{i}',start_s=a,end_s=b,acceptable_cameras=[c],cut_window_s=None,context_program=None,intent='subject',reason_tags=[]) for i,(a,b,c) in enumerate([(0.,2.,'cam_a'),(2.,4.,'cam_b'),(4.,6.,'cam_a')])]
        m['events']=[dict(id='marker',start_s=2.,end_s=4.,label='generated marker NOT professional narrative',visible_cameras=['cam_b'],importance=.7)]
        p=store.create(m);m=p['manifest']
        m['comparisons']=[dict(id='generated-comparison',start_s=1.,end_s=5.,source_identity=media_identity(m),
            a=dict(initial_program='cam_a',cuts=[]),b=dict(initial_program='cam_a',cuts=[dict(time_s=2.,camera_id='cam_b'),dict(time_s=4.,camera_id='cam_a')]),
            label='b',reviewer='AUTOMATED SYNTHETIC fixture NOT a human director',reason_tags=['coverage'],notes='fabricated workflow label; not a real judgment')]
        p=store.save(p['id'],p['revision'],m);p=store.review(p['id'],p['revision'],'AUTOMATED SYNTHETIC fixture NOT a human director',True,True)
        path=output/f'event-{ordinal}'/'index.json';index=prepare_index(root,p,path,chunk_steps=4,decision_hz=2,purpose='training',allow_synthetic=True)
        features=extract(root,index,'v1_visual',str(encoder_weights),encoder_sha,stop_file)
        atomic_json(path.parent/'feature-report.json',dict(index=path.relative_to(root).as_posix(),cache_id=features['cache_id'],feature_dim=features['feature_dim'],description=features['description']))
        entries.append(dict(index=path.relative_to(root).as_posix(),cache_id=features['cache_id']))
    corpus_path=output/'corpus.json';save_corpus(root,entries,corpus_path,True,'generated engineering smoke only')
    events=load_corpus(root,corpus_path,True);dim=events[0].header['feature_dim']
    cfg=PolicyConfig(visual_dim=dim,max_cameras=4,hidden_dim=32,layers=1,short_tokens=8,shot_tokens=4,slow_tokens=4,slow_stride_s=1.,dropout=0.)
    cp,sft=train(root,corpus_path,output/'supervised',cfg,epochs=2,allow_synthetic=True,stop_file=stop_file)
    pref,pref_report=train(root,corpus_path,output/'preference',cfg,epochs=1,allow_synthetic=True,mode='preference',teacher_checkpoint=cp,stop_file=stop_file)
    from dataclasses import replace
    student,student_report=train(root,corpus_path,output/'student',replace(cfg,hidden_dim=16,short_tokens=4),epochs=1,allow_synthetic=True,mode='distill',teacher_checkpoint=pref,stop_file=stop_file)
    quality,quality_report=train_quality(root,corpus_path,output/'quality',epochs=2,allow_synthetic=True,stop_file=stop_file)
    quality_eval=output/'quality-test.json';evaluate_quality(root,quality,corpus_path,quality_eval,allow_synthetic=True)
    evaluation=evaluate(root,student,corpus_path,output/'evaluation',allow_synthetic=True,min_hold_s=1.,stop_file=stop_file)
    test=[e for e in events if e.manifest['split']=='test'][0]
    loop=replay_policy(root,student,test.index,test.cache_id,output/'correction-source',min_hold_s=1.,stop_file=stop_file)
    correction=create_correction(root,test.index,output/'correction-source/replay-report.json');atomic_json(output/'correction.json',correction)
    export=export_policy(student,output/'export',coreml=coreml)
    packets=Path(data_output)/'recorded-packets.jsonl';packets_from_event(root,test.index,test.cache_id,packets)
    shadow=run_shadow(root,student,packets,output/'shadow',min_hold_s=1.,stop_file=stop_file)
    gate=assess(root,student,output/'evaluation/evaluation.json',output/'gate-report.json',output/'export/export-report.json',output/'shadow/shadow-report.json',quality_eval)
    result=dict(format='camix_p2_p5_engineering_smoke_v2',status='completed',source='synthetic',human_director_evidence=False,
                checkpoint=str(student.relative_to(root)),corpus=str(corpus_path.relative_to(root)),
                completed_paths=['supervised_training','short_shot_slow_memories','intent_visibility_auxiliary_tasks',
                    'reviewed_reference_relative_preferences','causal_teacher_student_distillation','offline_clip_quality',
                    'independent_event_ablations_and_matched_replays','unreviewed_correction_draft','policy_step_export','recorded_feature_packet_shadow','research_gates'],
                research_gate_status=gate['status'],coreml_parity_passed=export.get('coreml_parity_passed'),
                production_approved=False,execute_allowed=False,
                warning='All examples, labels and reviews in this smoke are automated synthetic fixtures; they prove executable engineering paths, NOT aesthetics, narrative, generalization or professional quality.')
    atomic_json(output/'stages-smoke-report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workspace',default='.');parser.add_argument('--stop-file')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('corpus');p.add_argument('--entries',required=True);p.add_argument('--output',required=True);p.add_argument('--allow-synthetic',action='store_true')
    p=sub.add_parser('train');p.add_argument('--corpus',required=True);p.add_argument('--output',required=True);p.add_argument('--config')
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--lr',type=float,default=.0003);p.add_argument('--seed',type=int,default=42);p.add_argument('--patience',type=int,default=0)
    p.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu');p.add_argument('--mode',choices=('supervised','distill','preference'),default='supervised')
    p.add_argument('--init-checkpoint');p.add_argument('--resume-checkpoint');p.add_argument('--teacher-checkpoint');p.add_argument('--beta',type=float,default=.2);p.add_argument('--distill-weight',type=float,default=.5)
    p.add_argument('--allow-synthetic',action='store_true');p.add_argument('--progress-file')
    p=sub.add_parser('evaluate');p.add_argument('--checkpoint',required=True);p.add_argument('--corpus',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split',choices=('train','validation','test'),default='test');p.add_argument('--allow-synthetic',action='store_true');p.add_argument('--allow-seen',action='store_true');p.add_argument('--min-hold-s',type=float,default=2.)
    p=sub.add_parser('replay');p.add_argument('--checkpoint',required=True);p.add_argument('--index',required=True);p.add_argument('--cache-id',required=True);p.add_argument('--output',required=True)
    p.add_argument('--min-hold-s',type=float,default=2.);p.add_argument('--min-probability',type=float,default=0.)
    p=sub.add_parser('quality-train');p.add_argument('--corpus',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=5);p.add_argument('--lr',type=float,default=.001);p.add_argument('--allow-synthetic',action='store_true')
    p=sub.add_parser('quality-evaluate');p.add_argument('--checkpoint',required=True);p.add_argument('--corpus',required=True);p.add_argument('--output',required=True);p.add_argument('--allow-synthetic',action='store_true')
    p=sub.add_parser('correct');p.add_argument('--index',required=True);p.add_argument('--replay-report',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('export');p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--coreml',action='store_true');p.add_argument('--steps',type=int,default=12)
    p=sub.add_parser('packets');p.add_argument('--index',required=True);p.add_argument('--cache-id',required=True);p.add_argument('--output',required=True);p.add_argument('--report',required=True)
    p=sub.add_parser('shadow');p.add_argument('--checkpoint',required=True);p.add_argument('--packets',required=True);p.add_argument('--output',required=True);p.add_argument('--min-hold-s',type=float,default=2.);p.add_argument('--min-probability',type=float,default=0.)
    p=sub.add_parser('gate');p.add_argument('--checkpoint',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True);p.add_argument('--export-report');p.add_argument('--shadow-report');p.add_argument('--quality-evaluation')
    p=sub.add_parser('smoke');p.add_argument('--output',required=True);p.add_argument('--data-output',required=True);p.add_argument('--weights',required=True);p.add_argument('--weights-sha256',required=True);p.add_argument('--allow-synthetic',action='store_true');p.add_argument('--coreml',action='store_true')
    a=parser.parse_args();root=Path(a.workspace).resolve()
    path=lambda v:artifact_path(root,str(v))
    output=lambda v:artifact_path(root,str(v),False)
    try:
        if a.command=='corpus':
            from .corpus_v2 import save_corpus
            value=input_json(root,a.entries);result=save_corpus(root,value['entries'] if isinstance(value,dict) else value,output(a.output),a.allow_synthetic)
        elif a.command=='train':
            from .train_v2 import train
            cfg=input_json(root,a.config) if a.config else None
            cp,result=train(root,path(a.corpus),output(a.output),cfg,a.epochs,a.lr,a.seed,a.device,a.patience,a.allow_synthetic,a.mode,
                            path(a.init_checkpoint) if a.init_checkpoint else None,path(a.resume_checkpoint) if a.resume_checkpoint else None,
                            path(a.teacher_checkpoint) if a.teacher_checkpoint else None,a.beta,a.distill_weight,a.stop_file,a.progress_file)
        elif a.command=='evaluate':
            from .evaluate_v2 import evaluate
            result=evaluate(root,path(a.checkpoint),path(a.corpus),output(a.output),a.split,a.allow_synthetic,a.allow_seen,a.min_hold_s,a.stop_file)
        elif a.command=='replay':
            from .policy_replay_v2 import replay_policy
            result=replay_policy(root,path(a.checkpoint),load_json(path(a.index)),a.cache_id,output(a.output),a.min_hold_s,a.min_probability,stop_file=a.stop_file)
        elif a.command=='quality-train':
            from .quality_v2 import train_quality
            cp,result=train_quality(root,path(a.corpus),output(a.output),a.epochs,a.lr,allow_synthetic=a.allow_synthetic,stop_file=a.stop_file)
        elif a.command=='quality-evaluate':
            from .quality_v2 import evaluate_quality
            result=evaluate_quality(root,path(a.checkpoint),path(a.corpus),output(a.output),allow_synthetic=a.allow_synthetic)
        elif a.command=='correct':
            from .corrections_v2 import create_correction
            target=output(a.output)
            if target.exists():raise ValueError('Correction report output exists')
            result=create_correction(root,load_json(path(a.index)),path(a.replay_report));atomic_json(target,result)
        elif a.command=='export':
            from .export_v2 import export_policy
            result=export_policy(path(a.checkpoint),output(a.output),a.coreml,a.steps)
        elif a.command=='packets':
            from .shadow_v2 import packets_from_event
            target=output(a.report)
            if target.exists():raise ValueError('Packet report output exists')
            result=packets_from_event(root,load_json(path(a.index)),a.cache_id,output(a.output));atomic_json(target,result)
        elif a.command=='shadow':
            from .shadow_v2 import run_shadow
            result=run_shadow(root,path(a.checkpoint),path(a.packets),output(a.output),a.min_hold_s,a.min_probability,a.stop_file)
        elif a.command=='gate':
            from .gates_v2 import assess
            result=assess(root,path(a.checkpoint),path(a.evaluation),output(a.output),path(a.export_report) if a.export_report else None,
                          path(a.shadow_report) if a.shadow_report else None,path(a.quality_evaluation) if a.quality_evaluation else None)
        else:
            if not a.allow_synthetic:raise ValueError('Explicit synthetic smoke acknowledgment required')
            from .features_v2 import local_weights
            weights=local_weights(root,a.weights,a.weights_sha256)
            result=smoke(root,output(a.output),output(a.data_output),weights,a.weights_sha256,a.coreml,a.stop_file)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2))
    except TrainingCancelled as error:
        print(str(error),flush=True);raise SystemExit(130)
    except InterruptedError as error:
        print(str(error),flush=True);raise SystemExit(130)


if __name__=='__main__':main()
