"""Authenticated stage catalogue and strict background job arguments. No shell strings."""
from pathlib import Path
from .contracts_v2 import boolean,integer,load_json,number
from .runtime import atomic_json
from .v2 import artifact_path

STAGE_COMMANDS=('corpus','train','evaluate','replay','quality-train','quality-evaluate','correct','export','packets','shadow','gate','smoke')


def catalogue(root):
    root=Path(root).resolve();base=root/'runs/v2';members=[];warnings=[]
    for path in sorted(base.rglob('feature-report.json'))[-200:]:
        try:
            report=load_json(path);index_path=artifact_path(root,report['index']);index=load_json(index_path);m=index['document']['manifest']
            members.append(dict(index=index_path.relative_to(root).as_posix(),cache_id=report['cache_id'],
                original_event_id=m['original_event_id'],split=m['split'],source=m['source'],
                eligible=index['config']['purpose']=='training' and index['document']['review'] is not None,
                feature_dim=report['feature_dim'],label=f"{m['original_event_id']} · {m['split']} · {m['source']}"))
        except (KeyError,ValueError,OSError) as error:warnings.append(path.name+': '+str(error))
    def paths(pattern):return [p.relative_to(root).as_posix() for p in sorted(base.rglob(pattern))[-200:]]
    return dict(stage='P2-P5 executable research paths; no professional approval',policy_implementation_available=True,
                production_approved=False,members=members,warnings=warnings,
                corpora=paths('corpus.json'),training_reports=paths('training-report.json'),quality_reports=paths('quality-training-report.json'),
                gates=paths('gate-report.json'),smokes=paths('stages-smoke-report.json'),packet_reports=paths('packets-report.json'),policies=paths('best.pt'),resume_checkpoints=paths('last.pt'),
                qualities=paths('quality-best.pt'),evaluations=paths('evaluation.json'),replays=paths('replay-report.json'),
                exports=paths('export-report.json'),shadows=paths('shadow-report.json'),quality_evaluations=paths('quality-test.json'),
                corpus_note='Only reviewed research indexes can train; use the P0/P1 page to label/review/re-extract first')


def make_job(manager,payload,jobdir):
    from .web_core import stamp
    root=manager.root;uid=jobdir.name;command=payload['kind'].removeprefix('v2_stage_')
    if command not in STAGE_COMMANDS:raise ValueError('Unknown V2 stage command')
    out=root/'runs/v2'/uid
    args=['--workspace',str(root),'--stop-file',str(jobdir/'stop.request'),command]
    def path(field):return str(artifact_path(root,payload.get(field)))
    def add_path(field,flag=None,optional=False):
        if optional and not payload.get(field):return
        args.extend([flag or '--'+field.replace('_','-'),path(field)])
    def flag(field):
        value=payload.get(field,False);boolean(value,field)
        if value:args.append('--'+field.replace('_','-'))
        return value
    if command=='corpus':
        entries=payload.get('entries')
        if not isinstance(entries,list) or not 1<=len(entries)<=1000:raise ValueError('Select 1..1000 reviewed index/cache pairs')
        selection=jobdir/'members.json';atomic_json(selection,entries);args+=['--entries',str(selection),'--output',str(out/'corpus.json')];flag('allow_synthetic')
    elif command=='train':
        from .model_v2 import PolicyConfig
        config=payload.get('config')
        if config is not None:
            if not isinstance(config,dict):raise ValueError('Policy config must be an object')
            PolicyConfig(**config);path_config=jobdir/'policy-config.json';atomic_json(path_config,config);args+=['--config',str(path_config)]
        mode=payload.get('mode','supervised')
        if mode not in ('supervised','distill','preference'):raise ValueError('Unknown policy learning mode')
        device=payload.get('device','cpu')
        if device not in ('cpu','mps','cuda') or not any(d['id']==device and d['available'] for d in manager.system['devices']):raise ValueError('Training device unavailable')
        epochs=integer(payload.get('epochs',10),'epochs',1,10000);lr=number(payload.get('lr',.0003),'lr',1e-7,.1)
        patience=integer(payload.get('patience',0),'patience',0,10000);seed=integer(payload.get('seed',42),'seed',0,2**31-1)
        beta=number(payload.get('beta',.2),'beta',.01,10);weight=number(payload.get('distill_weight',.5),'distill_weight',0,10)
        add_path('corpus');args+=['--output',str(out),'--mode',mode,'--epochs',str(epochs),'--lr',str(lr),'--device',device,'--seed',str(seed),
            '--patience',str(patience),'--beta',str(beta),'--distill-weight',str(weight),'--progress-file',str(jobdir/'progress.json')]
        add_path('init_checkpoint',optional=True);add_path('resume_checkpoint',optional=True);add_path('teacher_checkpoint',optional=True);flag('allow_synthetic')
        if mode in ('distill','preference') and not payload.get('teacher_checkpoint'):raise ValueError('Select a causal V2 teacher/reference checkpoint')
        if payload.get('init_checkpoint') and payload.get('resume_checkpoint'):raise ValueError('Do not combine fine-tuning and resume')
    elif command=='evaluate':
        add_path('checkpoint');add_path('corpus');split=payload.get('split','test')
        if split not in ('train','validation','test'):raise ValueError('Unknown evaluation split')
        hold=number(payload.get('min_hold_s',2.),'min hold',0,60);args+=['--output',str(out),'--split',split,'--min-hold-s',str(hold)];flag('allow_synthetic');flag('allow_seen')
    elif command=='replay':
        add_path('checkpoint');add_path('index');args+=['--cache-id',str(payload.get('cache_id','')),'--output',str(out),
            '--min-hold-s',str(number(payload.get('min_hold_s',2.),'min hold',0,60)),
            '--min-probability',str(number(payload.get('min_probability',0.),'uncalibrated threshold',0,1))]
    elif command=='quality-train':
        add_path('corpus');args+=['--output',str(out),'--epochs',str(integer(payload.get('epochs',5),'epochs',1,1000)),'--lr',str(number(payload.get('lr',.001),'lr',1e-7,.1))];flag('allow_synthetic')
    elif command=='quality-evaluate':
        add_path('checkpoint');add_path('corpus');args+=['--output',str(out/'quality-test.json')];flag('allow_synthetic')
    elif command=='correct':
        add_path('index');add_path('replay_report');args+=['--output',str(out/'correction.json')]
    elif command=='export':
        add_path('checkpoint');args+=['--output',str(out),'--steps',str(integer(payload.get('steps',12),'parity steps',4,100))];flag('coreml')
    elif command=='packets':
        add_path('index');args+=['--cache-id',str(payload.get('cache_id','')),'--output',str(root/'data/v2_packets'/f'{uid}.jsonl'),'--report',str(out/'packets-report.json')]
    elif command=='shadow':
        add_path('checkpoint');add_path('packets');args+=['--output',str(out),'--min-hold-s',str(number(payload.get('min_hold_s',2.),'min hold',0,60)),
            '--min-probability',str(number(payload.get('min_probability',0.),'uncalibrated threshold',0,1))]
    elif command=='gate':
        add_path('checkpoint');add_path('evaluation');add_path('export_report',optional=True);add_path('shadow_report',optional=True);add_path('quality_evaluation',optional=True);args+=['--output',str(out/'gate-report.json')]
    else:
        if not flag('allow_synthetic'):raise ValueError('Synthetic smoke requires explicit acknowledgment')
        from .features_v2 import local_weights
        weights=local_weights(root,payload.get('weights'),payload.get('weights_sha256'))
        args+=['--weights',str(weights),'--weights-sha256',payload['weights_sha256'],'--output',str(out),'--data-output',str(root/'data'/f'v2_stage_smoke_{uid}')];flag('coreml')
    return dict(id=uid,kind=payload['kind'],workspace=str(root),created_at=stamp(),status='queued',output=str(out),
                module='camix_director.stages_v2',args=args)
