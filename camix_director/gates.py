"""Evidence-based offline candidate gates. Never grants production approval."""
from pathlib import Path
from datetime import datetime,timezone
import json
from .storage import digest_file
from .runtime import atomic_json


def assess(checkpoint,evaluation,replay_report=None,min_sessions=3):
    cp=Path(checkpoint);evaluation=json.loads(Path(evaluation).read_text(encoding='utf-8'));sha=digest_file(cp)
    rows=[r for r in evaluation.get('evaluations',[]) if r.get('ablation')=='full'];sessions={r['session_id'] for r in rows}
    evidence=evaluation.get('data_provenance',[])
    checks=[]
    def add(name,passed,detail):checks.append(dict(name=name,passed=bool(passed),detail=detail))
    add('checkpoint_identity',evaluation.get('checkpoint_sha256')==sha,'评估必须与该检查点哈希匹配；旧报告需重新评估')
    add('independent_sessions',len(sessions)>=min_sessions,f'{len(sessions)}个独立场次，要求至少{min_sessions}（本项目建议门槛，不是行业统一标准）')
    add('reviewed_evaluation',bool(evidence) and all(p.get('source')=='real' and p.get('human_reviewed') and p.get('review') for p in evidence),'需真实素材、版本化本地人审证据；弱监督/合成不能冒充专业验收')
    def aggregate(rows):
        out={}
        for row in rows:
            if row.get('pairs',0) and row.get('pair_accuracy') is not None:
                entry=out.setdefault(row['session_id'],[0.,0]);entry[0]+=row['pair_accuracy']*row['pairs'];entry[1]+=row['pairs']
        return {sid:dict(pair_accuracy=values[0]/values[1]) for sid,values in out.items()}
    full=aggregate(rows)
    for ablation,key in [('video_content_muted','visual_contribution'),('memory_reset_each_step','memory_contribution')]:
        others=aggregate([r for r in evaluation.get('evaluations',[]) if r.get('ablation')==ablation])
        differences=[r['pair_accuracy']-others[s]['pair_accuracy'] for s,r in full.items() if s in others]
        mean=sum(differences)/len(differences) if differences else None
        add(key,mean is not None and mean>0,'场次平均准确率贡献：'+str(mean)+'；短样本的正差也不是统计显著性证明')
    replay=json.loads(Path(replay_report).read_text(encoding='utf-8')) if replay_report else None
    add('closed_loop_replay',bool(replay) and replay.get('checkpoint_sha256')==sha and replay.get('execute_allowed') is False and not any(p.get('seen_training_event') for p in replay.get('data_provenance',[])),'需要同一检查点的离线闭环回放，不是直播操作')
    passed=all(c['passed'] for c in checks)
    return dict(format='camix_offline_gate_v1',created_at=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=sha,status='offline_candidate' if passed else 'blocked',checks=checks,production_approved=False,missing_production_evidence=['Independent blind expert evaluation','Real device end-to-end latency, thermal and fault tests','Live shadow-mode acceptance and rollback review'],warning='本地建议门槛不是大厂认证。即使offline_candidate，也不批准真实节目控制。')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--replay');p.add_argument('--output',required=True);a=p.parse_args()
    result=assess(a.checkpoint,a.evaluation,a.replay);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);atomic_json(out/'gate-report.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
